"""
Response formatting functions that use join_processing (highest dependency level).

This module contains functions that require join_processing utilities and create
the potential for circular dependencies. These are separated to maintain
clean dependency hierarchy.
"""

from typing import Any, Callable, TYPE_CHECKING, cast

from ...types import SelectSchemaType
from ..introspection import get_primary_key_names
from .nesting import nest_join_data
from .transforms import format_multi_response

if TYPE_CHECKING:  # pragma: no cover
    from ..config import JoinConfig
    from sqlalchemy.ext.asyncio import AsyncSession


def process_joined_data(
    data_list: list[dict],
    join_definitions: list["JoinConfig"],
    nest_joins: bool,
    primary_model: Any,
) -> dict[str, Any] | None:
    """
    Process joined data using core utilities for nesting and relationships.

    Args:
        data_list: List of flat dictionaries containing joined data
        join_definitions: List of join configurations
        nest_joins: Whether to nest the joined data
        primary_model: Primary SQLAlchemy model class

    Returns:
        Processed nested data dictionary or None if no data
    """
    if not data_list:
        return None

    if not nest_joins:
        return data_list[0]

    one_to_many_count = sum(
        1 for join in join_definitions if join.relationship_type == "one-to-many"
    )

    if one_to_many_count > 1:
        pre_nested_data = []
        for row_data in data_list:
            nested_row = nest_join_data(
                data=row_data,
                join_definitions=join_definitions,
                get_primary_key_func=lambda model: get_primary_key_names(model)[0],
            )
            pre_nested_data.append(nested_row)

        from ..join_processing import JoinProcessor

        processor = JoinProcessor(primary_model)
        nested_results = processor.process_multi_join(
            data=pre_nested_data,
            joins_config=join_definitions,
            return_as_model=False,
            schema_to_select=None,
            nested_schema_to_select={
                (
                    join.join_prefix.rstrip("_")
                    if join.join_prefix
                    else join.model.__tablename__
                ): join.schema_to_select
                for join in join_definitions
                if join.schema_to_select
            },
        )
        return dict(nested_results[0]) if nested_results else {}
    else:
        nested_data: dict = {}
        for data in data_list:
            nested_data = nest_join_data(
                data,
                join_definitions,
                lambda model: get_primary_key_names(model)[0],
                nested_data=nested_data,
            )
        return nested_data


async def format_joined_response(
    primary_model: Any,
    raw_data: list[dict],
    config: dict[str, Any],
    schema_to_select: type[SelectSchemaType] | None = None,
    return_as_model: bool = False,
    nest_joins: bool = False,
    return_total_count: bool = True,
    db: "AsyncSession | None" = None,
    nested_schema_to_select: dict[str, type[SelectSchemaType]] | None = None,
    count_func: Callable | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Format response using core utilities.

    Args:
        primary_model: Primary SQLAlchemy model class
        raw_data: Raw query results
        config: Configuration dictionary with join_definitions
        schema_to_select: Pydantic schema for response formatting
        return_as_model: Whether to return as Pydantic model
        nest_joins: Whether joins are nested
        return_total_count: Whether to include total count
        db: Database session (for count queries)
        nested_schema_to_select: Schemas for nested data
        count_func: Function to get total count
        **kwargs: Additional filter parameters

    Returns:
        Formatted response dictionary
    """
    join_definitions = config["join_definitions"]

    processed_data = []
    for row_dict in raw_data:
        if nest_joins:
            row_dict = nest_join_data(
                data=row_dict,
                join_definitions=join_definitions,
                get_primary_key_func=lambda model: get_primary_key_names(model)[0],
            )
        processed_data.append(row_dict)

    nested_data: list[dict[str, Any] | SelectSchemaType]
    if nest_joins and any(
        join.relationship_type == "one-to-many" for join in join_definitions
    ):
        from ..join_processing import JoinProcessor

        processor = JoinProcessor(primary_model)
        nested_result = processor.process_multi_join(
            data=processed_data,
            joins_config=join_definitions,
            return_as_model=return_as_model,
            schema_to_select=schema_to_select if return_as_model else None,
            nested_schema_to_select=nested_schema_to_select
            or {
                (
                    join.join_prefix.rstrip("_")
                    if join.join_prefix
                    else join.model.__tablename__
                ): join.schema_to_select
                for join in join_definitions
                if join.schema_to_select
            },
        )
        nested_data = list(nested_result)
    else:
        from ..join_processing import handle_null_primary_key_multi_join

        nested_data = handle_null_primary_key_multi_join(
            cast(list[dict[str, Any] | SelectSchemaType], processed_data),
            join_definitions,
        )

    formatted_data: list[Any] = format_multi_response(
        nested_data, schema_to_select, return_as_model
    )
    response: dict[str, Any] = {"data": formatted_data}

    if return_total_count and db and count_func:
        distinct_on_primary = bool(
            nest_joins
            and any(j.relationship_type == "one-to-many" for j in join_definitions)
        )
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
        total_count: int = await count_func(
            db=db,
            joins_config=join_definitions,
            distinct_on_primary=distinct_on_primary,
            **filter_kwargs,
        )
        response["total_count"] = total_count

    return response
