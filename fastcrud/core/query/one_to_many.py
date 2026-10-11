"""
Separate query handling for one-to-many relationships with SQL-level limiting.

This module provides functions to fetch one-to-many related data using separate
queries with window functions, enabling proper SQL-level limiting per parent record.
"""

import logging
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import select, func, literal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BooleanClauseList, ColumnElement
from sqlalchemy.sql.expression import BinaryExpression
from sqlalchemy.sql.selectable import FromClause
from sqlalchemy.inspection import inspect as sa_inspect

from .builder import describe_from_clause

if TYPE_CHECKING:  # pragma: no cover
    from ..config import JoinConfig
    from ...types import ModelType

logger = logging.getLogger(__name__)

_LARGE_PARENT_SET_WARNING_THRESHOLD = 1000


def split_join_condition(
    join_on: ColumnElement[bool],
    related_model: Any,
    primary_model: "ModelType",
) -> tuple[Any, list[ColumnElement[bool]]]:
    """
    Split a one-to-many join condition into its foreign key and its other conditions.

    A ``join_on`` such as ``(Author.id == Article.author_id) & (Article.published)``
    links the models by the equality between their columns, and keeps only the
    related rows that meet the rest. The rest is applied when the related rows
    are fetched in their own query.

    When more than one equality compares the two models, such as
    ``Article.lang == Author.lang`` beside the foreign key, the link is the one
    whose related column has a foreign key to the primary model.

    Args:
        join_on: The join condition, one comparison or several joined by AND
        related_model: The related model or its alias (e.g., Article)
        primary_model: The primary/parent model (e.g., Author)

    Returns:
        Tuple of (foreign_key_column, conditions on the related model)

    Raises:
        ValueError: If no equality links the two models, or another condition
            reads a table other than the related model's, which the related rows'
            own query can't see
    """
    is_and = (
        isinstance(join_on, BooleanClauseList) and join_on.operator is operators.and_
    )
    terms = list(join_on.clauses) if is_and else [join_on]

    related = sa_inspect(related_model)
    related_name = related.mapper.class_.__name__
    related_tables = _tables_of(related_model)
    primary_tables = _tables_of(primary_model)

    links = [
        (term, columns)
        for term in terms
        if (columns := _linked_columns(term, related_tables, primary_tables))
    ]
    if not links:
        raise ValueError(
            f"Cannot find the equality between {primary_model.__name__} and "
            f"{related_name} columns in the join condition: {join_on}"
        )
    link, (fk_column, _) = next(
        (
            (term, columns)
            for term, columns in links
            if _references(columns[0], primary_tables)
        ),
        links[0],
    )

    conditions = [term for term in terms if term is not link]
    for condition in conditions:
        others = [
            from_clause
            for from_clause in select(literal(1))
            .select_from(related.selectable)
            .where(condition)
            .get_final_froms()
            if from_clause is not related.selectable
        ]
        if others:
            names = ", ".join(
                primary_model.__name__
                if from_clause in primary_tables
                else describe_from_clause(from_clause)
                for from_clause in others
            )
            raise ValueError(
                f"A one-to-many join with nested_limit can only add conditions on "
                f"{related_name}, not on {names}: {condition}"
            )
    return fk_column, conditions


def _tables_of(model: Any) -> set[FromClause]:
    """The tables a model's columns belong to: its alias, or its mapped tables."""
    inspected = sa_inspect(model)
    if inspected.is_aliased_class:
        return {inspected.selectable}
    return set(inspected.mapper.tables)


def _linked_columns(
    term: ColumnElement[bool],
    related_tables: set[FromClause],
    primary_tables: set[FromClause],
) -> tuple[Any, Any] | None:
    """The (related, primary) columns an equality between the two models compares."""
    if not (isinstance(term, BinaryExpression) and term.operator is operators.eq):
        return None
    left_table = getattr(term.left, "table", None)
    right_table = getattr(term.right, "table", None)
    if left_table in related_tables and right_table in primary_tables:
        return term.left, term.right
    if right_table in related_tables and left_table in primary_tables:
        return term.right, term.left
    return None


def _references(column: Any, tables: set[FromClause]) -> bool:
    return any(
        foreign_key.column.table in tables
        for foreign_key in getattr(column, "foreign_keys", ())
    )


async def fetch_one_to_many_with_limit(
    db: AsyncSession,
    related_model: "ModelType",
    fk_column: Any,
    parent_ids: list[Any],
    limit_per_parent: int,
    sort_columns: str | list[str] | None = None,
    sort_orders: str | list[str] | None = None,
    schema_to_select: type[BaseModel] | None = None,
    filters: dict[str, Any] | None = None,
    conditions: Sequence[ColumnElement[bool]] = (),
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
        conditions: SQLAlchemy conditions on the related model the records must meet

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
    base_query = select(*columns, row_number).where(
        fk_column.in_(parent_ids), *conditions
    )
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

    Raises:
        ValueError: If a join condition can't be applied to the related rows' own query
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

        fk_column, conditions = split_join_condition(
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
            conditions=conditions,
        )
        fk_name = fk_column.key
        for item in related_data:
            parent_id = item.get(fk_name)
            if parent_id in results_by_pk:
                results_by_pk[parent_id][nested_key].append(item)

    return main_results
