"""
CRUD-specific execution utilities.

This module contains execution logic for CRUD operations,
including statement execution and response handling.
"""

from typing import Any, Union, TYPE_CHECKING, cast
from sqlalchemy import column
from sqlalchemy.ext.asyncio import AsyncSession

from ..core import format_single_response, format_multi_response
from ..core.protocols import CRUDInstance

if TYPE_CHECKING:  # pragma: no cover
    from ..types import SelectSchemaType, GetMultiResponseModel, GetMultiResponseDict


async def execute_update_and_return_response(
    db: AsyncSession,
    stmt: Any,
    commit: bool,
    return_columns: list[str] | None,
    schema_to_select: type["SelectSchemaType"] | None,
    return_as_model: bool,
    allow_multiple: bool,
    one_or_none: bool,
) -> Union[dict, "SelectSchemaType"] | None:
    """
    Execute update statement and format response using core utilities.

    Args:
        db: Database session
        stmt: SQLAlchemy statement to execute
        commit: Whether to commit the transaction
        return_columns: Columns to return from the update
        schema_to_select: Pydantic schema for response formatting
        return_as_model: Whether to return as Pydantic model
        allow_multiple: Whether multiple records are expected
        one_or_none: Whether to use one_or_none() vs first()

    Returns:
        Formatted response data or None
    """
    if return_columns:
        stmt = stmt.returning(*[column(name) for name in return_columns])
        db_row = await db.execute(stmt)
        if commit:
            await db.commit()

        if allow_multiple:
            multi_data = [dict(row) for row in db_row.mappings()]
            formatted_data = format_multi_response(
                multi_data, schema_to_select, return_as_model
            )
            return {"data": formatted_data}

        result_row = db_row.one_or_none() if one_or_none else db_row.first()
        if not result_row:
            return None
        single_data = dict(result_row._mapping)
        return format_single_response(single_data, schema_to_select, return_as_model)
    else:
        await db.execute(stmt)
        if commit:
            await db.commit()
        return None


async def handle_joined_filters_delegation(
    crud_instance: CRUDInstance,
    joined_filters_info: dict[str, Any],
    db: AsyncSession,
    offset: int,
    limit: int | None,
    schema_to_select: type["SelectSchemaType"] | None,
    sort_columns: str | list[str] | None,
    sort_orders: str | list[str] | None,
    return_as_model: bool,
    return_total_count: bool,
    **regular_filters: Any,
) -> Union["GetMultiResponseModel[SelectSchemaType]", "GetMultiResponseDict"]:
    """
    Handle delegation to get_multi_joined when joined filters are detected.

    Args:
        crud_instance: The FastCRUD instance (for accessing model and get_multi_joined)
        joined_filters_info: Information about joined filters
        db: Database session
        offset: Number of records to skip
        limit: Maximum number of records to return
        schema_to_select: Pydantic schema for response formatting
        sort_columns: Columns to sort by
        sort_orders: Sort order for columns
        return_as_model: Whether to return as Pydantic model
        return_total_count: Whether to include total count
        **regular_filters: Regular filter arguments

    Returns:
        Response from get_multi_joined

    Raises:
        ValueError: If relationship not found or multiple joined filters provided
    """
    if len(joined_filters_info) == 1:
        relationship_name = list(joined_filters_info.keys())[0]
        relationship_filters = joined_filters_info[relationship_name]

        relationship = getattr(crud_instance.model, relationship_name, None)
        if relationship is None:
            raise ValueError(
                f"Relationship '{relationship_name}' not found in model '{crud_instance.model.__name__}'"
            )

        if hasattr(relationship.property, "mapper"):
            join_model = relationship.property.mapper.class_
        else:
            raise ValueError(
                f"Invalid relationship '{relationship_name}' in model '{crud_instance.model.__name__}'"
            )

        result: (
            "GetMultiResponseModel[SelectSchemaType]" | "GetMultiResponseDict"
        ) = await crud_instance.get_multi_joined(
            db=db,
            offset=offset,
            limit=limit,
            schema_to_select=schema_to_select,
            join_model=join_model,
            join_filters=relationship_filters,
            sort_columns=sort_columns,
            sort_orders=sort_orders,
            return_as_model=return_as_model,
            return_total_count=return_total_count,
            **regular_filters,
        )
        return cast(
            Union["GetMultiResponseModel[SelectSchemaType]", "GetMultiResponseDict"],
            result,
        )
    else:
        raise ValueError(
            "Multiple joined filters are not supported in get_multi. "
            "Use get_multi_joined with explicit joins_config instead."
        )
