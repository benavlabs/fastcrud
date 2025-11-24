"""
Tests for PaginatedRequestQuery reusability in custom endpoints (SQLModel).

This test file demonstrates how the PaginatedRequestQuery schema can be
reused in custom endpoints, which is the main benefit of issue #258.
"""

import pytest
from typing import Annotated
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from fastcrud.core import PaginatedRequestQuery


@pytest.fixture
def custom_app(async_session) -> FastAPI:
    """Create a FastAPI app with custom endpoints using PaginatedRequestQuery."""
    app = FastAPI()

    # Custom endpoint that reuses PaginatedRequestQuery
    @app.get("/custom/items")
    async def custom_get_items(
        query: Annotated[PaginatedRequestQuery, Depends()],
        db: AsyncSession = Depends(lambda: async_session),
    ):
        """Custom endpoint demonstrating PaginatedRequestQuery reusability."""
        # This endpoint can now easily access all pagination parameters
        # through the query object
        return {
            "offset": query.offset,
            "limit": query.limit,
            "page": query.page,
            "items_per_page": query.items_per_page,
            "sort": query.sort,
        }

    # Another custom endpoint with additional custom logic
    @app.get("/custom/filtered-items")
    async def custom_filtered_items(
        query: Annotated[PaginatedRequestQuery, Depends()],
        db: AsyncSession = Depends(lambda: async_session),
    ):
        """Custom endpoint with additional filtering logic."""
        # Calculate offset from page if needed
        offset = query.offset
        limit = query.limit

        if query.page is not None or query.items_per_page is not None:
            page = query.page if query.page else 1
            items_per_page = query.items_per_page if query.items_per_page else 10
            offset = (page - 1) * items_per_page
            limit = items_per_page

        return {
            "calculated_offset": offset,
            "calculated_limit": limit,
            "original_page": query.page,
            "original_items_per_page": query.items_per_page,
        }

    return app


@pytest.fixture
def custom_client(custom_app):
    """Create a test client for the custom app."""
    return TestClient(custom_app)


def test_custom_endpoint_with_pagination_params(custom_client):
    """Test custom endpoint receives pagination parameters correctly."""
    response = custom_client.get("/custom/items?page=2&itemsPerPage=20")

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 2
    assert data["items_per_page"] == 20
    assert data["offset"] is None
    assert data["limit"] is None
    assert data["sort"] is None


def test_custom_endpoint_with_offset_limit(custom_client):
    """Test custom endpoint receives offset/limit parameters correctly."""
    response = custom_client.get("/custom/items?offset=10&limit=50")

    assert response.status_code == 200
    data = response.json()

    assert data["offset"] == 10
    assert data["limit"] == 50
    assert data["page"] is None
    assert data["items_per_page"] is None
    assert data["sort"] is None


def test_custom_endpoint_with_sort(custom_client):
    """Test custom endpoint receives sort parameter correctly."""
    response = custom_client.get("/custom/items?sort=name,-age")

    assert response.status_code == 200
    data = response.json()

    assert data["sort"] == "name,-age"
    assert data["page"] is None
    assert data["items_per_page"] is None
    assert data["offset"] is None
    assert data["limit"] is None


def test_custom_endpoint_with_all_params(custom_client):
    """Test custom endpoint receives all parameters correctly."""
    response = custom_client.get(
        "/custom/items?page=3&itemsPerPage=15&sort=id&offset=5&limit=25"
    )

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 3
    assert data["items_per_page"] == 15
    assert data["sort"] == "id"
    assert data["offset"] == 5
    assert data["limit"] == 25


def test_custom_endpoint_with_no_params(custom_client):
    """Test custom endpoint with no parameters (all defaults to None)."""
    response = custom_client.get("/custom/items")

    assert response.status_code == 200
    data = response.json()

    assert data["page"] is None
    assert data["items_per_page"] is None
    assert data["offset"] is None
    assert data["limit"] is None
    assert data["sort"] is None


def test_custom_endpoint_calculates_offset_from_page(custom_client):
    """Test custom endpoint can calculate offset from page parameters."""
    response = custom_client.get("/custom/filtered-items?page=3&itemsPerPage=10")

    assert response.status_code == 200
    data = response.json()

    # Page 3 with 10 items per page should have offset of 20
    assert data["calculated_offset"] == 20
    assert data["calculated_limit"] == 10
    assert data["original_page"] == 3
    assert data["original_items_per_page"] == 10


def test_custom_endpoint_uses_offset_limit_directly(custom_client):
    """Test custom endpoint uses offset/limit when provided."""
    response = custom_client.get("/custom/filtered-items?offset=15&limit=25")

    assert response.status_code == 200
    data = response.json()

    assert data["calculated_offset"] == 15
    assert data["calculated_limit"] == 25
    assert data["original_page"] is None
    assert data["original_items_per_page"] is None


def test_custom_endpoint_with_partial_pagination(custom_client):
    """Test custom endpoint with only page parameter (items_per_page defaults to 10)."""
    response = custom_client.get("/custom/filtered-items?page=2")

    assert response.status_code == 200
    data = response.json()

    # Page 2 with default 10 items per page should have offset of 10
    assert data["calculated_offset"] == 10
    assert data["calculated_limit"] == 10
    assert data["original_page"] == 2
    assert data["original_items_per_page"] is None


def test_openapi_schema_includes_query_params(custom_client):
    """Test that OpenAPI schema includes all query parameters."""
    response = custom_client.get("/openapi.json")

    assert response.status_code == 200
    openapi_schema = response.json()

    # Check that the custom endpoint is in the schema
    assert "/custom/items" in openapi_schema["paths"]

    # Get the parameters for the GET endpoint
    params = openapi_schema["paths"]["/custom/items"]["get"]["parameters"]

    # Extract parameter names
    param_names = [p["name"] for p in params]

    # Verify all expected parameters are present
    assert "offset" in param_names
    assert "limit" in param_names
    assert "page" in param_names
    assert "itemsPerPage" in param_names  # Note: this is the alias
    assert "sort" in param_names

    # Verify that parameters have proper schema definitions
    for param in params:
        assert "schema" in param or "content" in param


def test_subclassing_paginated_request_query() -> None:
    """Test that PaginatedRequestQuery can be subclassed for custom extensions."""
    from typing import Optional
    from pydantic import Field

    class CustomPaginatedQuery(PaginatedRequestQuery):
        """Extended query with custom filter."""

        custom_filter: Optional[str] = Field(
            None, description="Custom filter parameter"
        )

    # Test instantiation
    query = CustomPaginatedQuery(
        page=1,
        itemsPerPage=10,
        offset=None,
        limit=None,
        sort=None,
        custom_filter="active",
    )

    assert query.page == 1
    assert query.items_per_page == 10
    assert query.custom_filter == "active"

    # Test that base fields still work
    assert query.offset is None
    assert query.limit is None
    assert query.sort is None


def test_custom_endpoint_with_zero_values(custom_client):
    """Test that zero values for offset and limit are handled correctly (not replaced with defaults)."""
    # Test offset=0 (should be preserved, not replaced with default)
    response = custom_client.get("/custom/items?offset=0&limit=50")

    assert response.status_code == 200
    data = response.json()

    assert data["offset"] == 0  # Should preserve 0, not replace with default
    assert data["limit"] == 50

    # Test limit=0 (should be preserved, not replaced with default)
    response = custom_client.get("/custom/items?offset=10&limit=0")

    assert response.status_code == 200
    data = response.json()

    assert data["offset"] == 10
    assert data["limit"] == 0  # Should preserve 0, not replace with default
