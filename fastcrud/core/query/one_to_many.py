"""
Separate query handling for one-to-many relationships with SQL-level limiting.

This module provides functions to fetch one-to-many related data using separate
queries with window functions, enabling proper SQL-level limiting per parent record.
"""

import logging
from typing import Any, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import BinaryExpression
from sqlalchemy.inspection import inspect as sa_inspect

if TYPE_CHECKING:  # pragma: no cover
    from ..config import JoinConfig
    from ...types import ModelType

logger = logging.getLogger(__name__)

_LARGE_PARENT_SET_WARNING_THRESHOLD = 1000


def extract_foreign_key_info(
    join_on: BinaryExpression,
    related_model: "ModelType",
    primary_model: "ModelType",
) -> tuple[Any, Any]:
    """
    Extract foreign key column and parent primary key column from join condition.

    Args:
        join_on: SQLAlchemy join condition (e.g., Article.author_id == Author.id)
        related_model: The related model (e.g., Article)
        primary_model: The primary/parent model (e.g., Author)

    Returns:
        Tuple of (foreign_key_column, parent_pk_column)

    Raises:
        ValueError: If cannot extract foreign key info from join condition
    """
    try:
        left = join_on.left
        right = join_on.right

        related_table = related_model.__table__
        primary_table = primary_model.__table__

        if hasattr(left, "table") and left.table == related_table:
            fk_column = left
            pk_column = right
        elif hasattr(right, "table") and right.table == related_table:
            fk_column = right
            pk_column = left
        else:
            left_key = getattr(left, "key", None)
            right_key = getattr(right, "key", None)

            related_cols = {c.key for c in related_table.columns}
            primary_cols = {c.key for c in primary_table.columns}

            if left_key in related_cols and right_key in primary_cols:
                fk_column = left
                pk_column = right
            elif right_key in related_cols and left_key in primary_cols:
                fk_column = right
                pk_column = left
            else:
                raise ValueError(
                    f"Cannot determine foreign key relationship from join condition: {join_on}"
                )

        return fk_column, pk_column

    except Exception as e:
        raise ValueError(
            f"Cannot extract foreign key info from join condition: {join_on}. Error: {e}"
        )


async def fetch_one_to_many_with_limit(
    db: AsyncSession,
    related_model: "ModelType",
    fk_column: Any,
    parent_ids: list[Any],
    limit_per_parent: int,
    sort_columns: Optional[Union[str, list[str]]] = None,
    sort_orders: Optional[Union[str, list[str]]] = None,
    schema_to_select: Optional[type[BaseModel]] = None,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Fetch one-to-many related records with SQL-level limiting using window functions.

    This function executes a query that returns at most `limit_per_parent` records
    for each parent ID, using ROW_NUMBER() window function for efficient SQL-level
    limiting.

    Args:
        db: AsyncSession for database operations
        related_model: The related SQLAlchemy model
        fk_column: Foreign key column on the related model
        parent_ids: List of parent primary key values to fetch related records for
        limit_per_parent: Maximum number of records to return per parent
        sort_columns: Optional column(s) to sort by within each parent group
        sort_orders: Optional sort orders ("asc" or "desc") for sort_columns
        schema_to_select: Optional Pydantic schema for column selection
        filters: Optional dict of column_name: value filters to apply

    Returns:
        List of dictionaries, each containing related record data with the FK value

    Example:
        >>> # Fetch top 10 published articles per author
        >>> results = await fetch_one_to_many_with_limit(
        ...     db=session,
        ...     related_model=Article,
        ...     fk_column=Article.author_id,
        ...     parent_ids=[1, 2, 3],
        ...     limit_per_parent=10,
        ...     sort_columns=["created_at"],
        ...     sort_orders=["desc"],
        ...     filters={"is_published": True},
        ... )
    """
    if not parent_ids:
        return []

    mapper = sa_inspect(related_model).mapper
    if schema_to_select:
        columns = [
            getattr(related_model, field)
            for field in schema_to_select.model_fields.keys()
            if hasattr(related_model, field) and field not in mapper.relationships
        ]
    else:
        columns = [getattr(related_model, prop.key) for prop in mapper.column_attrs]

    order_by_clauses = []
    if sort_columns:
        if isinstance(sort_columns, str):
            sort_columns = [sort_columns]
        if isinstance(sort_orders, str):
            sort_orders = [sort_orders]
        sort_orders = sort_orders or ["asc"] * len(sort_columns)

        for col, order in zip(sort_columns, sort_orders):
            if hasattr(related_model, col):
                col_attr = getattr(related_model, col)
                if order.lower() == "desc":
                    order_by_clauses.append(col_attr.desc())
                else:
                    order_by_clauses.append(col_attr.asc())
    else:
        pk_cols = [pk for pk in mapper.primary_key]
        for pk in pk_cols:
            order_by_clauses.append(getattr(related_model, pk.key).asc())

    row_number = (
        func.row_number()
        .over(partition_by=fk_column, order_by=order_by_clauses)
        .label("_rn")
    )
    base_query = select(*columns, row_number).where(fk_column.in_(parent_ids))
    if filters:
        for col_name, value in filters.items():
            if hasattr(related_model, col_name):
                col_attr = getattr(related_model, col_name)
                base_query = base_query.where(col_attr == value)

    subquery = base_query.subquery()
    final_stmt = select(subquery).where(subquery.c._rn <= limit_per_parent)

    result = await db.execute(final_stmt)
    rows = result.fetchall()
    data = []
    for row in rows:
        row_dict = dict(row._mapping)
        row_dict.pop("_rn", None)
        data.append(row_dict)

    return data


def split_join_configs(
    joins_config: list["JoinConfig"],
) -> tuple[list["JoinConfig"], list["JoinConfig"]]:
    """
    Split join configs into regular joins and one-to-many with limits.

    Args:
        joins_config: List of JoinConfig objects

    Returns:
        Tuple of (regular_joins, one_to_many_with_limits)
        - regular_joins: Joins to include in the main query
        - one_to_many_with_limits: One-to-many joins with nested_limit that
          should use separate queries
    """
    regular_joins = []
    one_to_many_with_limits = []

    for config in joins_config:
        if (
            config.relationship_type == "one-to-many"
            and config.nested_limit is not None
        ):
            one_to_many_with_limits.append(config)
        else:
            regular_joins.append(config)

    return regular_joins, one_to_many_with_limits


async def fetch_and_merge_one_to_many(
    db: AsyncSession,
    primary_model: "ModelType",
    main_results: list[dict[str, Any]],
    one_to_many_configs: list["JoinConfig"],
    pk_column_name: str,
) -> list[dict[str, Any]]:
    """
    Fetch one-to-many data and merge into main results.

    Args:
        db: AsyncSession for database operations
        primary_model: The primary SQLAlchemy model
        main_results: Results from the main query
        one_to_many_configs: List of JoinConfig for one-to-many relationships
        pk_column_name: Name of the primary key column on the primary model

    Returns:
        Main results with one-to-many data merged in (preserves original order)
    """
    if not main_results or not one_to_many_configs:
        return main_results

    parent_ids = [
        r[pk_column_name]
        for r in main_results
        if pk_column_name in r and r[pk_column_name] is not None
    ]
    if not parent_ids:
        return main_results

    if len(parent_ids) > _LARGE_PARENT_SET_WARNING_THRESHOLD:
        logger.warning(
            "Fetching one-to-many relationships for %d parent records. "
            "Consider using pagination (limit/offset) in the main query for better "
            "performance. The IN clause for this many IDs may be slow on some databases.",
            len(parent_ids),
        )

    results_by_pk = {
        r[pk_column_name]: r
        for r in main_results
        if pk_column_name in r and r[pk_column_name] is not None
    }

    for config in one_to_many_configs:
        nested_key = (
            config.join_prefix.rstrip("_")
            if config.join_prefix
            else config.model.__tablename__
        )

        try:
            fk_column, _ = extract_foreign_key_info(
                config.join_on, config.model, primary_model
            )
            for pk in parent_ids:
                if pk in results_by_pk:
                    results_by_pk[pk][nested_key] = []
            assert (
                config.nested_limit is not None
            ), "nested_limit should not be None for one-to-many configs with limits"

            related_data = await fetch_one_to_many_with_limit(
                db=db,
                related_model=config.model,
                fk_column=fk_column,
                parent_ids=parent_ids,
                limit_per_parent=config.nested_limit,
                sort_columns=config.sort_columns,
                sort_orders=config.sort_orders,
                schema_to_select=config.schema_to_select,
                filters=config.filters,
            )
            fk_name = fk_column.key
            for item in related_data:
                parent_id = item.get(fk_name)
                if parent_id in results_by_pk:
                    results_by_pk[parent_id][nested_key].append(item)

        except Exception as e:
            logger.warning(
                "Failed to fetch one-to-many data for %s: %s. "
                "Falling back to empty list.",
                config.model.__name__,
                str(e),
            )
            for result in main_results:
                if nested_key not in result:
                    result[nested_key] = []

    return main_results
