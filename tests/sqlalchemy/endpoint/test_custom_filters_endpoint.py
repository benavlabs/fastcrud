"""Tests for custom filters at the endpoint/router level."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Column

from fastcrud import FastCRUD, crud_router, FilterConfig, FilterCallable


@pytest.fixture
def custom_filter_client(
    test_model, create_schema, update_schema, delete_schema, async_session
):
    """Client with custom filters defined at router level."""
    app = FastAPI()

    # Custom filter that checks if value is in a range (between min and max)
    def in_range(col: Column) -> FilterCallable:
        def filter_fn(value):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return (col >= value[0]) & (col <= value[1])
            return col == value

        return filter_fn

    # Don't pass crud= so that crud_router creates one with custom_filters
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            custom_filters={"in_range": in_range},
            filter_config=FilterConfig(tier_id__in_range=None),
            path="/test",
            tags=["test"],
        )
    )

    return TestClient(app)


@pytest.fixture
def override_filter_client(
    test_model, create_schema, update_schema, delete_schema, async_session
):
    """Client with custom filter that overrides a built-in operator."""
    app = FastAPI()

    # Override 'eq' to be case-insensitive for strings
    def case_insensitive_eq(col: Column) -> FilterCallable:
        def filter_fn(value):
            from sqlalchemy import func

            if isinstance(value, str):
                return func.lower(col) == func.lower(value)
            return col == value

        return filter_fn

    # Don't pass crud= so that crud_router creates one with custom_filters
    # Use name__eq to explicitly use the eq operator (not simple equality)
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            custom_filters={"eq": case_insensitive_eq},
            filter_config=FilterConfig(name__eq=None),
            path="/test",
            tags=["test"],
        )
    )

    return TestClient(app)


@pytest.mark.asyncio
async def test_crud_router_custom_filter(
    custom_filter_client, async_session, test_model, test_data
):
    """Test custom filter works through crud_router endpoints."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    # This test verifies the custom filter is registered correctly
    # The actual filtering via query params would require list support in query params
    # For now, just verify the endpoint is created with custom filter config
    response = custom_filter_client.get("/test/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_crud_router_override_builtin_filter(
    override_filter_client, async_session, test_model, test_data
):
    """Test overriding built-in filter through crud_router."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    # With case-insensitive eq, "alice" should match "Alice"
    # Using name__eq to explicitly use the overridden eq operator
    response = override_filter_client.get("/test/", params={"name__eq": "alice"})
    assert response.status_code == 200
    data = response.json()
    # Should find Alice records (case-insensitive match)
    assert len(data["data"]) > 0
    assert all(item["name"].lower() == "alice" for item in data["data"])


@pytest.mark.asyncio
async def test_custom_filter_validation_in_filter_config(
    test_model, create_schema, update_schema, delete_schema, async_session
):
    """Test that custom operators are recognized in FilterConfig validation."""
    app = FastAPI()

    def custom_op(col: Column) -> FilterCallable:
        def filter_fn(value):
            return col == value

        return filter_fn

    # This should NOT raise an error because custom_op is provided
    # Don't pass crud= so that crud_router creates one with custom_filters
    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        delete_schema=delete_schema,
        custom_filters={"custom_op": custom_op},
        filter_config=FilterConfig(tier_id__custom_op=None),
        path="/test",
        tags=["test"],
    )

    app.include_router(router)
    client = TestClient(app)

    # Verify the endpoint works
    response = client.get("/test/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_custom_filter_unknown_operator_without_registration(
    test_model, create_schema, update_schema, delete_schema, async_session
):
    """Test that unknown operators raise an error if not registered as custom filter."""
    from fastcrud import EndpointCreator

    # This SHOULD raise an error because 'unknown_op' is not registered
    with pytest.raises(ValueError, match="Invalid filter op"):
        EndpointCreator(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            filter_config=FilterConfig(tier_id__unknown_op=None),
            path="/test",
            tags=["test"],
        )
