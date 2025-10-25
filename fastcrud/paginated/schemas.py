from typing import Generic, TypeVar, Optional, Type

from pydantic import BaseModel, create_model, Field

SchemaType = TypeVar("SchemaType", bound=BaseModel)


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
        from fastcrud.paginated import PaginatedRequestQuery

        async def custom_endpoint(
            query: Annotated[PaginatedRequestQuery, Depends()]
        ):
            # Use query.page, query.items_per_page, query.sort, etc.
            pass
        ```
    """

    offset: Optional[int] = Field(
        None,
        description="Offset for unpaginated queries"
    )
    limit: Optional[int] = Field(
        None,
        description="Limit for unpaginated queries"
    )
    page: Optional[int] = Field(
        None,
        alias="page",
        description="Page number"
    )
    items_per_page: Optional[int] = Field(
        None,
        alias="itemsPerPage",
        description="Number of items per page"
    )
    sort: Optional[str] = Field(
        None,
        description="Sort results by one or more fields. Format: 'field1,-field2' where '-' prefix indicates descending order. Example: 'name' (ascending), '-age' (descending), 'name,-age' (name ascending, then age descending)."
    )

    model_config = {"populate_by_name": True}


def create_list_response(
    schema: Type[SchemaType], response_key: str = "data"
) -> Type[BaseModel]:
    """Creates a dynamic ListResponse model with the specified response key."""
    return create_model("DynamicListResponse", **{response_key: (list[schema], ...)})  # type: ignore


def create_paginated_response(
    schema: Type[SchemaType], response_key: str = "data"
) -> Type[BaseModel]:
    """Creates a dynamic PaginatedResponse model with the specified response key."""
    fields = {
        response_key: (list[schema], ...),  # type: ignore
        "total_count": (int, ...),
        "has_more": (bool, ...),
        "page": (Optional[int], None),
        "items_per_page": (Optional[int], None),
    }
    return create_model("DynamicPaginatedResponse", **fields)  # type: ignore


class ListResponse(BaseModel, Generic[SchemaType]):
    data: list[SchemaType]


class PaginatedListResponse(ListResponse[SchemaType]):
    total_count: int
    has_more: bool
    page: Optional[int] = None
    items_per_page: Optional[int] = None
