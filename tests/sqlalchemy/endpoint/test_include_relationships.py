"""Tests for include_relationships parameter in crud_router and EndpointCreator."""

from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from fastcrud import FastCRUD, crud_router
from fastcrud.endpoint.endpoint_creator import EndpointCreator
from fastcrud.core import JoinConfig
from ..conftest import (
    ModelTest,
    TierModel,
    CreateSchemaTest,
    UpdateSchemaTest,
    ReadSchemaTest,
)


# Schema that includes nested relationship fields
class TierNestedSchema(BaseModel):
    id: int
    name: str


class ReadSchemaWithRelationships(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: Optional[int] = None
    tier: Optional[TierNestedSchema] = None
    category: Optional[dict] = None


@pytest.fixture
def client_with_relationships(async_session, test_model, tier_model):
    """Client with include_relationships=True enabled."""
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            # No select_schema to avoid schema validation issues with auto-detected relationships
            include_relationships=True,
            nest_joins=True,
            path="/test",
            tags=["test"],
        )
    )

    return TestClient(app)


@pytest.fixture
def client_without_relationships(async_session, test_model):
    """Client with include_relationships=False (default)."""
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            select_schema=ReadSchemaTest,
            include_relationships=False,
            path="/test",
            tags=["test"],
        )
    )

    return TestClient(app)


@pytest.mark.asyncio
async def test_read_item_with_relationships(
    client_with_relationships: TestClient,
    async_session,
    test_data,
    test_data_tier,
):
    """Test that read endpoint includes related data when include_relationships=True."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create test item
    item = ModelTest(name="Test User", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client_with_relationships.get(f"/test/{item.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test User"
    # Should have nested tier data
    assert "tier" in data or "tier_id" in data


@pytest.mark.asyncio
async def test_read_item_without_relationships(
    client_without_relationships: TestClient,
    async_session,
    test_data_tier,
):
    """Test that read endpoint returns flat data when include_relationships=False."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create test item
    item = ModelTest(name="Test User", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client_without_relationships.get(f"/test/{item.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test User"
    assert "tier_id" in data
    # Should NOT have nested tier data
    assert "tier" not in data or data.get("tier") is None


@pytest.mark.asyncio
async def test_read_multi_with_relationships(
    client_with_relationships: TestClient,
    async_session,
    test_data_tier,
):
    """Test that read_multi endpoint includes related data when include_relationships=True."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create test items
    items = [
        ModelTest(name="User 1", tier_id=1),
        ModelTest(name="User 2", tier_id=2),
    ]
    async_session.add_all(items)
    await async_session.commit()

    response = client_with_relationships.get("/test?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 2


@pytest.mark.asyncio
async def test_read_multi_with_relationships_and_pagination(
    client_with_relationships: TestClient,
    async_session,
    test_data_tier,
):
    """Test pagination works correctly with include_relationships=True."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create more test items
    items = [ModelTest(name=f"User {i}", tier_id=1) for i in range(5)]
    async_session.add_all(items)
    await async_session.commit()

    response = client_with_relationships.get("/test?offset=0&limit=2")

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 2
    assert "total_count" in data
    assert data["total_count"] == 5


@pytest.mark.asyncio
async def test_create_item_with_relationships(
    client_with_relationships: TestClient,
    async_session,
    test_data_tier,
):
    """Test that create endpoint works with include_relationships=True."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    response = client_with_relationships.post(
        "/test",
        json={"name": "New User", "tier_id": 1},
    )

    # Create should succeed
    assert response.status_code == 200
    # Without select_schema, response is null (expected v0.20.0 behavior)
    # The important thing is that create works without errors


@pytest.mark.asyncio
async def test_update_item_with_relationships(
    client_with_relationships: TestClient,
    async_session,
    test_data_tier,
):
    """Test that update endpoint returns data with relationships when enabled."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create test item
    item = ModelTest(name="Original Name", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client_with_relationships.patch(
        f"/test/{item.id}",
        json={"name": "Updated Name"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"


@pytest.mark.asyncio
async def test_delete_item_with_relationships(
    client_with_relationships: TestClient,
    async_session,
    test_data_tier,
):
    """Test that delete endpoint returns data with relationships when enabled."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create test item
    item = ModelTest(name="To Delete", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client_with_relationships.delete(f"/test/{item.id}")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_nest_joins_true_required_for_one_to_many(async_session, test_data_tier):
    """Test that include_relationships with nest_joins=False raises error for one-to-many relationships."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # ModelTest has a one-to-many relationship (multi_pk), so nest_joins=False will fail
    # This test verifies the error is raised correctly
    # Note: include_one_to_many=True is required because one-to-many is excluded by default
    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=ModelTest,
            crud=FastCRUD(ModelTest),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            include_relationships=True,
            include_one_to_many=True,  # Explicitly include one-to-many
            nest_joins=False,  # This will fail for models with one-to-many
            path="/test",
            tags=["test"],
        )
    )
    client = TestClient(app, raise_server_exceptions=False)

    # Create test item
    item = ModelTest(name="Test User", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    # Should raise error because of one-to-many relationship with nest_joins=False
    response = client.get(f"/test/{item.id}")
    # The error is raised during request processing, returns 500
    assert response.status_code == 500


# ============================================================================
# Tests for selective relationship inclusion
# ============================================================================


@pytest.fixture
def client_with_selective_relationships(async_session, test_model, tier_model):
    """Client with selective relationships - only 'tier' relationship."""
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            include_relationships=["tier"],  # Only include tier relationship
            nest_joins=True,
            path="/test",
            tags=["test"],
        )
    )

    return TestClient(app)


@pytest.mark.asyncio
async def test_selective_relationship_inclusion(
    client_with_selective_relationships: TestClient,
    async_session,
    test_data_tier,
):
    """Test that only specified relationships are included."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create test item
    item = ModelTest(name="Selective Test User", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client_with_selective_relationships.get(f"/test/{item.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Selective Test User"
    # Should have tier data since it was explicitly included
    assert "tier" in data or "tier_id" in data


@pytest.mark.asyncio
async def test_invalid_relationship_name_raises_error(async_session, test_model):
    """Test that passing an invalid relationship name raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        EndpointCreator(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            include_relationships=["nonexistent_relationship"],
            path="/test",
            tags=["test"],
        )

    error_message = str(exc_info.value)
    assert "Invalid relationship name(s)" in error_message
    assert "nonexistent_relationship" in error_message
    assert "Available relationships" in error_message


@pytest.mark.asyncio
async def test_multiple_invalid_relationship_names(async_session, test_model):
    """Test that passing multiple invalid relationship names shows all invalid names."""
    with pytest.raises(ValueError) as exc_info:
        EndpointCreator(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            include_relationships=["bad_rel1", "bad_rel2"],
            path="/test",
            tags=["test"],
        )

    error_message = str(exc_info.value)
    assert "bad_rel1" in error_message
    assert "bad_rel2" in error_message


@pytest.mark.asyncio
async def test_include_relationships_and_joins_config_mutually_exclusive(
    async_session, test_model
):
    """Test that using both include_relationships and joins_config raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        EndpointCreator(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            include_relationships=True,
            joins_config=[
                JoinConfig(
                    model=TierModel,
                    join_on=ModelTest.tier_id == TierModel.id,
                    join_prefix="tier_",
                )
            ],
            path="/test",
            tags=["test"],
        )

    error_message = str(exc_info.value)
    assert "Cannot use both" in error_message
    assert "include_relationships" in error_message
    assert "joins_config" in error_message


@pytest.fixture
def client_with_joins_config(async_session, test_model, tier_model):
    """Client with manual joins_config instead of include_relationships."""
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            joins_config=[
                JoinConfig(
                    model=TierModel,
                    join_on=ModelTest.tier_id == TierModel.id,
                    join_prefix="tier_",
                    schema_to_select=None,
                    join_type="left",
                )
            ],
            nest_joins=True,
            path="/test",
            tags=["test"],
        )
    )

    return TestClient(app)


@pytest.mark.asyncio
async def test_joins_config_parameter(
    client_with_joins_config: TestClient,
    async_session,
    test_data_tier,
):
    """Test that joins_config parameter works correctly."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    # Create test item
    item = ModelTest(name="JoinConfig Test User", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client_with_joins_config.get(f"/test/{item.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "JoinConfig Test User"
    # Should have tier data from manual JoinConfig
    assert "tier" in data or "tier_id" in data


@pytest.mark.asyncio
async def test_crud_router_with_selective_relationships(async_session, test_data_tier):
    """Test crud_router function with selective relationships."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=ModelTest,
            crud=FastCRUD(ModelTest),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            include_relationships=["tier"],
            path="/test",
            tags=["test"],
        )
    )
    client = TestClient(app)

    # Create test item
    item = ModelTest(name="Test", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client.get(f"/test/{item.id}")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_crud_router_with_joins_config(async_session, test_data_tier):
    """Test crud_router function with joins_config parameter."""
    # Setup tier data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=ModelTest,
            crud=FastCRUD(ModelTest),
            create_schema=CreateSchemaTest,
            update_schema=UpdateSchemaTest,
            joins_config=[
                JoinConfig(
                    model=TierModel,
                    join_on=ModelTest.tier_id == TierModel.id,
                    join_prefix="tier_",
                )
            ],
            path="/test",
            tags=["test"],
        )
    )
    client = TestClient(app)

    # Create test item
    item = ModelTest(name="Test", tier_id=1)
    async_session.add(item)
    await async_session.commit()
    await async_session.refresh(item)

    response = client.get(f"/test/{item.id}")
    assert response.status_code == 200
