"""
CRUD-specific validation utilities.

This module contains validation logic that is specific to CRUD operations
but generic enough to be reused across different CRUD classes.
"""

from typing import Any, Callable, Awaitable, TYPE_CHECKING
from sqlalchemy.orm import aliased as sa_aliased
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import NoResultFound, MultipleResultsFound

from ..core.field_management import has_table_overlap

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm.util import AliasedClass
    from ..types import ModelType, SelectSchemaType
    from ..core import JoinConfig, CountConfig


async def validate_update_delete_operation(
    count_func: Callable[..., Awaitable[int]],
    db: AsyncSession,
    allow_multiple: bool,
    operation_name: str,
    **kwargs: Any,
) -> int:
    """
    Validate that update/delete operations have valid target records.

    Args:
        count_func: The count function to use (e.g., self.count)
        db: Database session
        allow_multiple: Whether to allow operations on multiple records
        operation_name: Name of the operation for error messages (e.g., "update", "delete")
        **kwargs: Filter arguments to count matching records

    Returns:
        Total count of matching records

    Raises:
        NoResultFound: If no records match the filters
        MultipleResultsFound: If multiple records match but allow_multiple is False
    """
    total_count = await count_func(db, **kwargs)
    if total_count == 0:
        raise NoResultFound(f"No record found to {operation_name}.")
    if not allow_multiple and total_count > 1:
        raise MultipleResultsFound(
            f"Expected exactly one record to {operation_name}, found {total_count}."
        )
    return total_count


def validate_pagination_params(offset: int, limit: int | None) -> None:
    """
    Validate pagination parameters.

    Args:
        offset: Number of records to skip
        limit: Maximum number of records to return (None for no limit)

    Raises:
        ValueError: If offset or limit are negative
    """
    if (limit is not None and limit < 0) or offset < 0:
        raise ValueError("Limit and offset must be non-negative.")


def validate_joined_query_params(
    primary_model: Any,
    joins_config: list["JoinConfig"] | None = None,
    join_model: type["ModelType"] | None = None,
    join_prefix: str | None = None,
    join_on: Any | None = None,
    join_schema_to_select: type["SelectSchemaType"] | None = None,
    alias: "AliasedClass[Any] | None" = None,
    relationship_type: str | None = None,
    join_type: str = "left",
    join_filters: dict | None = None,
    counts_config: list["CountConfig"] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Validate and organize joined query parameters.

    Args:
        primary_model: The primary SQLAlchemy model class
        joins_config: List of join configurations
        join_model: Single join model (if not using joins_config)
        join_prefix: Prefix for joined columns
        join_on: Join condition
        join_schema_to_select: Schema for joined data
        alias: Table alias for the join
        relationship_type: Type of relationship (one-to-one, one-to-many)
        join_type: SQL join type (left, inner, etc.)
        join_filters: Filters for the joined table
        counts_config: Configuration for count operations
        limit: Maximum number of records to return
        offset: Number of records to skip

    Returns:
        Dictionary with validated configuration

    Raises:
        ValueError: If parameters are invalid or conflicting
    """
    from ..core import JoinConfig, auto_detect_join_condition

    if joins_config and (
        join_model
        or join_prefix
        or join_on
        or join_schema_to_select
        or alias
        or relationship_type
    ):
        raise ValueError(
            "Cannot use both single join parameters and joins_config simultaneously."
        )
    elif not joins_config and not join_model and not counts_config:
        raise ValueError("You need one of join_model, joins_config, or counts_config.")

    validate_pagination_params(offset, limit)

    if relationship_type is None:
        relationship_type = "one-to-one"

    join_definitions = joins_config if joins_config else []
    if join_model:
        if alias is None and join_on is None:
            if has_table_overlap(primary_model, join_model):
                alias = sa_aliased(join_model, flat=True)

        try:
            join_definitions.append(
                JoinConfig(
                    model=join_model,
                    join_on=join_on
                    if join_on is not None
                    else auto_detect_join_condition(
                        primary_model, join_model, join_alias=alias
                    ),
                    join_prefix=join_prefix,
                    schema_to_select=join_schema_to_select,
                    join_type=join_type,
                    alias=alias,
                    filters=join_filters,
                    relationship_type=relationship_type,
                )
            )
        except ValueError as e:
            raise ValueError(f"Could not configure join: {str(e)}")

    return {
        "join_definitions": join_definitions,
        "counts_config": counts_config,
        "relationship_type": relationship_type,
    }
