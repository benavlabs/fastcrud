"""
SQL Query Builder.

This module provides the main SQLQueryBuilder class that coordinates
query construction with support for filtering, sorting, pagination, and joins.
"""

from typing import Any, TYPE_CHECKING
from sqlalchemy import Select, select, func
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.ext.asyncio import AsyncSession

from .sorting import SortProcessor
from .joins import JoinBuilder
from ...types import ModelType
from ..introspection import get_primary_key_columns
from ..field_management import extract_matching_columns_from_schema

if TYPE_CHECKING:  # pragma: no cover
    from ...types import SelectSchemaType


class SQLQueryBuilder:
    """Builds and modifies SQLAlchemy SELECT statements."""

    def __init__(self, model: type[ModelType]):
        """
        Initialize query builder for a specific model.

        Args:
            model: SQLAlchemy model class
        """
        self.model = model
        self.sort_processor = SortProcessor(model)
        self.join_builder = JoinBuilder(model)

    def build_base_select(self, columns: list[Any] | None = None) -> Select:
        """
        Create base SELECT statement.

        Args:
            columns: Optional list of specific columns to select.
                    If None, selects all columns from the model.

        Returns:
            SQLAlchemy SELECT statement

        Example:
            >>> builder = SQLQueryBuilder(User)
            >>> stmt = builder.build_base_select()  # SELECT * FROM users
            >>> stmt = builder.build_base_select([User.id, User.name])  # SELECT id, name FROM users
        """
        if columns:
            return select(*columns)
        else:
            return select(self.model)

    def apply_filters(self, stmt: Select, filters: list[ColumnElement]) -> Select:
        """
        Apply WHERE conditions to statement.

        Args:
            stmt: SQLAlchemy SELECT statement to modify
            filters: List of SQLAlchemy filter conditions

        Returns:
            Modified SELECT statement with WHERE clauses

        Example:
            >>> builder = SQLQueryBuilder(User)
            >>> stmt = builder.build_base_select()
            >>> filters = [User.age > 18, User.is_active == True]
            >>> filtered_stmt = builder.apply_filters(stmt, filters)
        """
        if filters:
            return stmt.where(*filters)
        return stmt

    def apply_sorting(
        self,
        stmt: Select,
        sort_columns: str | list[str],
        sort_orders: str | list[str] | None = None,
    ) -> Select:
        """
        Apply ORDER BY to statement.

        Args:
            stmt: SQLAlchemy SELECT statement to modify
            sort_columns: Column name(s) to sort by
            sort_orders: Sort direction(s) - 'asc' or 'desc'

        Returns:
            Modified SELECT statement with ORDER BY clause

        Example:
            >>> builder = SQLQueryBuilder(User)
            >>> stmt = builder.build_base_select()
            >>> sorted_stmt = builder.apply_sorting(stmt, ['name', 'created_at'], ['asc', 'desc'])
        """
        return self.sort_processor.apply_sorting_to_statement(
            stmt, sort_columns, sort_orders
        )

    def apply_pagination(
        self, stmt: Select, offset: int = 0, limit: int | None = None
    ) -> Select:
        """
        Apply OFFSET and LIMIT to statement.

        Args:
            stmt: SQLAlchemy SELECT statement to modify
            offset: Number of rows to skip (default: 0)
            limit: Maximum number of rows to return (default: None - no limit)

        Returns:
            Modified SELECT statement with OFFSET and/or LIMIT clauses

        Example:
            >>> builder = SQLQueryBuilder(User)
            >>> stmt = builder.build_base_select()
            >>> paginated_stmt = builder.apply_pagination(stmt, offset=20, limit=10)
        """
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def prepare_joins(
        self,
        stmt: Select,
        joins_config: list[Any],
        use_temporary_prefix: bool = False,
        select_joined_columns: bool = True,
    ) -> Select:
        """
        Apply joins to statement.

        Args:
            stmt: SQLAlchemy SELECT statement to modify
            joins_config: List of join configurations
            use_temporary_prefix: Whether to use temporary prefixes for joins
            select_joined_columns: Whether to add joined columns to SELECT

        Returns:
            Modified SELECT statement with JOIN clauses

        Example:
            >>> builder = SQLQueryBuilder(User)
            >>> stmt = builder.build_base_select()
            >>> joined_stmt = builder.prepare_joins(stmt, [join_config])
        """
        return self.join_builder.prepare_joins(
            stmt, joins_config, use_temporary_prefix, select_joined_columns
        )


def build_joined_query(
    model: type[ModelType],
    query_builder: "SQLQueryBuilder",
    filter_processor: Any,
    config: dict[str, Any],
    schema_to_select: type["SelectSchemaType"] | None = None,
    nest_joins: bool = False,
    **kwargs: Any,
) -> Select:
    """
    Build SELECT statement with joins using core utilities.

    Args:
        model: Primary SQLAlchemy model class
        query_builder: SQLQueryBuilder instance
        filter_processor: FilterProcessor instance
        config: Configuration dictionary with join_definitions and counts_config
        schema_to_select: Pydantic schema for column selection
        nest_joins: Whether to use temporary prefixes for nesting
        **kwargs: Additional filter parameters

    Returns:
        SQLAlchemy SELECT statement with joins and filters
    """
    primary_select = extract_matching_columns_from_schema(
        model=model, schema=schema_to_select
    )
    stmt = query_builder.build_base_select(primary_select)

    join_definitions = config["join_definitions"]
    stmt = query_builder.prepare_joins(stmt, join_definitions, nest_joins)

    if config["counts_config"]:
        for count in config["counts_config"]:
            count_model = count.model
            count_alias = count.alias or f"{count_model.__tablename__}_count"

            count_primary_keys = get_primary_key_columns(count_model)
            if not count_primary_keys:  # pragma: no cover
                raise ValueError(
                    f"The model '{count_model.__name__}' does not have a primary key defined, which is required for counting."
                )

            count_subquery = select(func.count()).where(count.join_on)
            if count.filters:
                count_filters = filter_processor.parse_filters(
                    model=count_model, **count.filters
                )
                if count_filters:
                    count_subquery = count_subquery.filter(*count_filters)

            stmt = stmt.add_columns(count_subquery.scalar_subquery().label(count_alias))

    non_filter_params = {
        "schema_to_select",
        "join_model",
        "join_on",
        "join_prefix",
        "join_schema_to_select",
        "join_type",
        "alias",
        "join_filters",
        "nest_joins",
        "offset",
        "limit",
        "sort_columns",
        "sort_orders",
        "return_as_model",
        "joins_config",
        "counts_config",
        "return_total_count",
        "relationship_type",
        "nested_schema_to_select",
    }
    filter_kwargs = {k: v for k, v in kwargs.items() if k not in non_filter_params}
    primary_filters = filter_processor.parse_filters(**filter_kwargs)
    if primary_filters:
        stmt = query_builder.apply_filters(stmt, primary_filters)

    return stmt


async def execute_joined_query(
    db: AsyncSession,
    stmt: Select,
    query_builder: "SQLQueryBuilder",
    limit: int | None = None,
    offset: int = 0,
    sort_columns: str | list[str] | None = None,
    sort_orders: str | list[str] | None = None,
) -> list[dict]:
    """
    Execute query and return raw results.

    Args:
        db: Database session
        stmt: SQLAlchemy SELECT statement
        query_builder: SQLQueryBuilder instance for sorting/pagination
        limit: Maximum number of records to return
        offset: Number of records to skip
        sort_columns: Columns to sort by
        sort_orders: Sort order for columns

    Returns:
        List of dictionaries containing query results
    """
    if sort_columns:
        stmt = query_builder.apply_sorting(stmt, sort_columns, sort_orders)
    stmt = query_builder.apply_pagination(stmt, offset, limit)

    result = await db.execute(stmt)
    return [dict(row) for row in result.mappings().all()]
