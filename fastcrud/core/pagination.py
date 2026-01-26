"""
Pagination utilities for FastCRUD.

This module consolidates all pagination-related functionality including:
- Pagination parameter schemas
- Pagination response formatting
- Offset calculation helpers
- Dynamic response model creation
"""

from typing import Generic, TypeVar, Optional, Type, Any, Union

from pydantic import BaseModel, create_model, Field, field_validator

from ..types import GetMultiResponseDict

SchemaType = TypeVar("SchemaType", bound=BaseModel)


# ------------- Helper Functions -------------
def compute_offset(page: int, items_per_page: int) -> int:
    """Calculate the offset for pagination based on the given page number and items per page.

    The offset represents the starting point in a dataset for the items on a given page.
    For example, if each page displays 10 items and you want to display page 3, the offset will be 20,
    meaning the display should start with the 21st item.

    Args:
        page: The current page number. Page numbers should start from 1.
        items_per_page: The number of items to be displayed on each page.

    Returns:
        The calculated offset.

    Examples:
        >>> compute_offset(1, 10)
        0
        >>> compute_offset(3, 10)
        20
    """
    return (page - 1) * items_per_page


def paginated_response(
    crud_data: GetMultiResponseDict | dict[str, Any],
    page: int,
    items_per_page: int,
    multi_response_key: str = "data",
) -> dict[str, Any]:
    """Create a paginated response based on the provided data and pagination parameters.

    Args:
        crud_data: Data to be paginated, including the list of items and total count.
        page: Current page number.
        items_per_page: Number of items per page.
        multi_response_key: Key to use for the items list in the response (defaults to "data").

    Returns:
        A structured paginated response dict containing the list of items, total count, pagination flags, and numbers.

    Note:
        The function does not actually paginate the data but formats the response to indicate pagination metadata.
    """
    items = crud_data.get(multi_response_key, [])
    total_count = crud_data.get("total_count", 0)

    response = {
        multi_response_key: items,
        "total_count": total_count,
        "has_more": (page * items_per_page) < total_count,
        "page": page,
        "items_per_page": items_per_page,
    }

    return response


# ------------- Request Query Schemas -------------
class PaginatedRequestQuery(BaseModel):
    """
    Pydantic model for paginated query parameters.

    This model encapsulates all query parameters used for pagination and sorting
    in read_items endpoints. It can be used with FastAPI's Depends() to inject
    these parameters into endpoints, making it easy to reuse in custom endpoints.

    Supports two pagination modes:
    - Page-based: Using 'page' and 'items_per_page' (or 'itemsPerPage' alias)
    - Offset-based: Using 'offset' and 'limit'

    Attributes:
        offset: Offset for unpaginated queries (used with limit)
        limit: Limit for unpaginated queries (used with offset)
        page: Page number for paginated queries
        items_per_page: Number of items per page for paginated queries
        sort: Sort results by one or more fields. Format: 'field1,-field2' where '-' prefix
              indicates descending order. Example: 'name' (ascending), '-age' (descending),
              'name,-age' (name ascending, then age descending).

    Example:
        ```python
        from typing import Annotated
        from fastapi import Depends
        from fastcrud import PaginatedRequestQuery

        async def custom_endpoint(
            query: Annotated[PaginatedRequestQuery, Depends()]
        ):
            # Use query.page, query.items_per_page, query.sort, etc.
            pass
        ```
    """

    offset: int | None = Field(None, description="Offset for unpaginated queries")
    limit: int | None = Field(None, description="Limit for unpaginated queries")
    page: int | None = Field(None, alias="page", description="Page number")
    items_per_page: int | None = Field(
        None, alias="itemsPerPage", description="Number of items per page"
    )
    sort: str | None = Field(
        None,
        description="Sort results by one or more fields. Format: 'field1,-field2' where '-' prefix indicates descending order. Example: 'name' (ascending), '-age' (descending), 'name,-age' (name ascending, then age descending).",
    )

    model_config = {"populate_by_name": True}


class CursorPaginatedRequestQuery(BaseModel):
    """
    Pydantic model for cursor-based pagination query parameters.

    This model encapsulates all query parameters used for cursor-based pagination
    in endpoints. It can be used with FastAPI's Depends() to inject these parameters
    into endpoints, making it easy to reuse in custom endpoints.

    Cursor-based pagination is ideal for large datasets and infinite scrolling
    features, as it provides consistent results even when data is being modified.

    Attributes:
        cursor: Cursor value for pagination (typically the ID of the last item from previous page)
        limit: Maximum number of items to return per page
        sort_column: Column name to sort by (defaults to 'id')
        sort_order: Sort order, either 'asc' or 'desc' (defaults to 'asc')

    Example:
        ```python
        from typing import Annotated
        from fastapi import Depends
        from fastcrud import CursorPaginatedRequestQuery

        async def custom_cursor_endpoint(
            query: Annotated[CursorPaginatedRequestQuery, Depends()]
        ):
            # Use query.cursor, query.limit, query.sort_column, query.sort_order
            pass
        ```
    """

    cursor: int | str | None = Field(
        None,
        description="Cursor value for pagination (typically the ID of the last item from previous page). Supports int, datetime (ISO format), or UUID string.",
    )
    limit: int | None = Field(
        100, description="Maximum number of items to return per page", gt=0, le=1000
    )
    sort_column: str | None = Field("id", description="Column name to sort by")
    sort_order: str | None = Field(
        "asc",
        description="Sort order: 'asc' for ascending, 'desc' for descending",
        pattern="^(asc|desc)$",
    )

    model_config = {"populate_by_name": True}

    @field_validator("cursor", mode="before")
    @classmethod
    def coerce_cursor(cls, v):
        """Try to coerce string cursor to int if it represents a valid integer."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Try to parse as integer first
            try:
                return int(v)
            except ValueError:
                # Keep as string for datetime/UUID cursors
                return v
        return v


# ------------- Response Schema Factories -------------
def create_list_response(
    schema: type[SchemaType], response_key: str = "data"
) -> type[BaseModel]:
    """Creates a dynamic ListResponse model with the specified response key."""
    return create_model("DynamicListResponse", **{response_key: (list[schema], ...)})  # type: ignore


def create_paginated_response(
    schema: type[SchemaType], response_key: str = "data"
) -> type[BaseModel]:
    """Creates a dynamic PaginatedResponse model with the specified response key."""
    fields = {
        response_key: (list[schema], ...),  # type: ignore
        "total_count": (int, ...),
        "has_more": (bool, ...),
        "page": (int | None, None),
        "items_per_page": (int | None, None),
    }
    return create_model("DynamicPaginatedResponse", **fields)  # type: ignore


# ------------- Response Schema Classes -------------
class ListResponse(BaseModel, Generic[SchemaType]):
    data: list[SchemaType]


class PaginatedListResponse(ListResponse[SchemaType]):
    total_count: int
    has_more: bool
    page: int | None = None
    items_per_page: int | None = None
