"""
Tests for CursorPaginatedRequestQuery reusability in custom endpoints (SQLModel).

This test file demonstrates how the CursorPaginatedRequestQuery schema can be
reused in custom endpoints for cursor-based pagination with SQLModel.
"""

import pytest
from typing import Annotated
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from fastcrud.core import CursorPaginatedRequestQuery


@pytest.fixture
def cursor_app(async_session) -> FastAPI:
    """Create a FastAPI app with custom endpoints using CursorPaginatedRequestQuery."""
    app = FastAPI()

    # Custom endpoint that reuses CursorPaginatedRequestQuery
    @app.get("/cursor/items")
    async def custom_cursor_items(
        query: Annotated[CursorPaginatedRequestQuery, Depends()],
        db: AsyncSession = Depends(lambda: async_session),
    ):
        """Custom endpoint demonstrating CursorPaginatedRequestQuery reusability."""
        return {
            "cursor": query.cursor,
            "limit": query.limit,
            "sort_column": query.sort_column,
            "sort_order": query.sort_order,
        }

    # Another custom endpoint with cursor-based logic
    @app.get("/cursor/filtered-items")
    async def custom_cursor_filtered_items(
        query: Annotated[CursorPaginatedRequestQuery, Depends()],
        db: AsyncSession = Depends(lambda: async_session),
    ):
        """Custom endpoint with cursor-based pagination logic."""
        # Simulate cursor-based pagination logic
        current_cursor = query.cursor or 0
        limit = query.limit or 100
        has_next = current_cursor < 1000
        next_cursor = current_cursor + limit if has_next else None

        return {
            "current_cursor": query.cursor,
            "limit": query.limit,
            "sort_column": query.sort_column,
            "sort_order": query.sort_order,
            "next_cursor": next_cursor,
            "has_more": has_next,
        }

    return app


@pytest.fixture
def cursor_client(cursor_app):
    """Create a test client for the cursor app."""
    return TestClient(cursor_app)


def test_cursor_endpoint_with_all_params(cursor_client):
    """Test cursor endpoint receives all parameters correctly."""
    response = cursor_client.get(
        "/cursor/items?cursor=50&limit=20&sort_column=name&sort_order=desc"
    )

    assert response.status_code == 200
    data = response.json()

    assert data["cursor"] == 50
    assert data["limit"] == 20
    assert data["sort_column"] == "name"
    assert data["sort_order"] == "desc"


def test_cursor_endpoint_with_defaults(cursor_client):
    """Test cursor endpoint with default values."""
    response = cursor_client.get("/cursor/items")

    assert response.status_code == 200
    data = response.json()

    assert data["cursor"] is None
    assert data["limit"] == 100  # Default value
    assert data["sort_column"] == "id"  # Default value
    assert data["sort_order"] == "asc"  # Default value


def test_cursor_endpoint_with_partial_params(cursor_client):
    """Test cursor endpoint with only some parameters."""
    response = cursor_client.get("/cursor/items?cursor=25&sort_order=desc")

    assert response.status_code == 200
    data = response.json()

    assert data["cursor"] == 25
    assert data["limit"] == 100  # Default
    assert data["sort_column"] == "id"  # Default
    assert data["sort_order"] == "desc"


def test_cursor_endpoint_with_zero_cursor(cursor_client):
    """Test cursor endpoint with cursor=0 (should be preserved)."""
    response = cursor_client.get("/cursor/items?cursor=0&limit=50")

    assert response.status_code == 200
    data = response.json()

    assert data["cursor"] == 0  # Should preserve 0, not treat as None
    assert data["limit"] == 50


def test_cursor_filtered_endpoint_logic(cursor_client):
    """Test cursor endpoint with pagination logic."""
    response = cursor_client.get("/cursor/filtered-items?cursor=100&limit=50")

    assert response.status_code == 200
    data = response.json()

    assert data["current_cursor"] == 100
    assert data["limit"] == 50
    assert data["next_cursor"] == 150  # 100 + 50
    assert data["has_more"] is True


def test_cursor_filtered_endpoint_no_cursor(cursor_client):
    """Test cursor endpoint without cursor (first page)."""
    response = cursor_client.get("/cursor/filtered-items?limit=25")

    assert response.status_code == 200
    data = response.json()

    assert data["current_cursor"] is None
    assert data["limit"] == 25
    assert data["next_cursor"] == 25  # 0 + 25
    assert data["has_more"] is True


def test_cursor_endpoint_validation_errors(cursor_client):
    """Test cursor endpoint with invalid parameters."""
    # Test invalid sort_order
    response = cursor_client.get("/cursor/items?sort_order=invalid")
    assert response.status_code == 422  # Validation error

    # Test invalid limit (too large)
    response = cursor_client.get("/cursor/items?limit=2000")
    assert response.status_code == 422  # Validation error

    # Test invalid limit (zero or negative)
    response = cursor_client.get("/cursor/items?limit=0")
    assert response.status_code == 422  # Validation error

    response = cursor_client.get("/cursor/items?limit=-5")
    assert response.status_code == 422  # Validation error


def test_openapi_schema_includes_cursor_params(cursor_client):
    """Test that OpenAPI schema includes all cursor pagination parameters."""
    response = cursor_client.get("/openapi.json")

    assert response.status_code == 200
    openapi_schema = response.json()

    # Check that the cursor endpoint is in the schema
    assert "/cursor/items" in openapi_schema["paths"]

    # Get the parameters for the GET endpoint
    params = openapi_schema["paths"]["/cursor/items"]["get"]["parameters"]

    # Extract parameter names
    param_names = [p["name"] for p in params]

    # Verify all expected parameters are present
    assert "cursor" in param_names
    assert "limit" in param_names
    assert "sort_column" in param_names
    assert "sort_order" in param_names

    # Check that limit has validation constraints
    limit_param = next(p for p in params if p["name"] == "limit")
    assert "schema" in limit_param
    schema = limit_param["schema"]

    # The schema might use anyOf due to Optional type
    if "anyOf" in schema:
        # Find the integer constraint in anyOf
        integer_schema = next(
            (s for s in schema["anyOf"] if s.get("type") == "integer"), None
        )
        assert integer_schema is not None
        assert integer_schema.get("maximum") == 1000
        assert integer_schema.get("exclusiveMinimum") == 0
    else:
        assert schema.get("maximum") == 1000
        assert schema.get("exclusiveMinimum") == 0 or schema.get("minimum") == 1


def test_subclassing_cursor_paginated_request_query() -> None:
    """Test that CursorPaginatedRequestQuery can be subclassed for custom extensions."""
    from typing import Optional
    from pydantic import Field

    class CustomCursorQuery(CursorPaginatedRequestQuery):
        """Extended cursor query with custom filter."""

        status: str | None = Field(None, description="Filter by status")
        category: str | None = Field(None, description="Filter by category")

    # Test instantiation
    query = CustomCursorQuery(
        cursor=100,
        limit=50,
        sort_column="created_at",
        sort_order="desc",
        status="active",
        category="books",
    )

    assert query.cursor == 100
    assert query.limit == 50
    assert query.sort_column == "created_at"
    assert query.sort_order == "desc"
    assert query.status == "active"
    assert query.category == "books"


def test_cursor_pagination_model_validation():
    """Test CursorPaginatedRequestQuery model validation directly."""
    # Valid model
    valid_query = CursorPaginatedRequestQuery(
        cursor=25, limit=50, sort_column="name", sort_order="desc"
    )
    assert valid_query.cursor == 25
    assert valid_query.limit == 50
    assert valid_query.sort_column == "name"
    assert valid_query.sort_order == "desc"

    # Test with defaults
    default_query = CursorPaginatedRequestQuery()
    assert default_query.cursor is None
    assert default_query.limit == 100
    assert default_query.sort_column == "id"
    assert default_query.sort_order == "asc"

    # Test validation errors
    import pytest
    from pydantic import ValidationError

    # Invalid sort_order
    with pytest.raises(ValidationError):
        CursorPaginatedRequestQuery(sort_order="invalid")

    # Invalid limit (too large)
    with pytest.raises(ValidationError):
        CursorPaginatedRequestQuery(limit=2000)

    # Invalid limit (zero)
    with pytest.raises(ValidationError):
        CursorPaginatedRequestQuery(limit=0)
