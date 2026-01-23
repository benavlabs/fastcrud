"""
Tests for CreateConfig, UpdateConfig, and DeleteConfig functionality.
Tests auto-injection of fields via callables/dependencies.
"""

from datetime import datetime
from typing import Optional
from fastapi.testclient import TestClient
import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from fastcrud import crud_router, CreateConfig, UpdateConfig, DeleteConfig


# Test schemas
class CreateSchemaWithoutTierId(BaseModel):
    """Schema for creating items without tier_id (will be auto-injected)."""

    model_config = ConfigDict(extra="forbid")
    name: str


class UpdateSchemaWithoutTierId(BaseModel):
    """Schema for updating items without tier_id."""

    name: str | None = None


class EmptyCreateSchema(BaseModel):
    """Empty schema for testing multiple auto_fields."""

    pass


class EmptyUpdateSchema(BaseModel):
    """Empty schema for update."""

    pass


# Mock functions that will be used as auto_field callables
def get_mock_user_id() -> int:
    """Mock function that returns a user_id."""
    return 42


def get_mock_timestamp() -> datetime:
    """Mock function that returns a fixed timestamp."""
    return datetime(2024, 1, 15, 10, 30, 0)


def get_mock_status() -> str:
    """Mock function that returns a status."""
    return "active"


@pytest.mark.asyncio
async def test_create_with_auto_fields(
    client: TestClient, async_session, test_model, test_data
):
    """Test that auto_fields are injected during create."""
    # Create a new router with CreateConfig
    create_config = CreateConfig(
        auto_fields={
            "tier_id": lambda: 1,  # Auto-inject tier_id
        },
        exclude_from_schema=[],
    )

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=CreateSchemaWithoutTierId,
        update_schema=UpdateSchemaWithoutTierId,
        create_config=create_config,
        path="/test_auto",
        tags=["Test"],
    )

    client.app.include_router(router)  # type: ignore

    # Create item WITHOUT tier_id in payload (should be auto-injected)
    response = client.post("/test_auto", json={"name": "Auto Item"})

    assert response.status_code == 200, response.text

    # Verify the item was created with auto-injected tier_id
    stmt = select(test_model).where(test_model.name == "Auto Item")
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "Auto Item"
    assert fetched_record.tier_id == 1  # Auto-injected!


@pytest.mark.asyncio
async def test_update_with_auto_fields(
    client: TestClient, async_session, test_model, test_data, create_schema
):
    """Test that auto_fields are injected during update."""
    # First create an item
    test_item = test_model(name="Update Test", tier_id=1)
    async_session.add(test_item)
    await async_session.commit()
    await async_session.refresh(test_item)
    item_id = test_item.id

    # Create router with UpdateConfig
    update_config = UpdateConfig(
        auto_fields={
            "tier_id": lambda: 2,  # Auto-update tier_id
        },
        exclude_from_schema=[],
    )

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=UpdateSchemaWithoutTierId,
        update_config=update_config,
        path="/test_update_auto",
        tags=["Test"],
    )

    client.app.include_router(router)  # type: ignore

    # Update item WITHOUT tier_id in payload (should be auto-injected)
    response = client.patch(
        f"/test_update_auto/{item_id}", json={"name": "Updated Name"}
    )

    assert response.status_code == 200, response.text

    # Verify the item was updated with auto-injected tier_id
    stmt = select(test_model).where(test_model.id == item_id)
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "Updated Name"
    assert fetched_record.tier_id == 2  # Auto-injected!


@pytest.mark.asyncio
async def test_create_without_config_still_works(
    client: TestClient,
    async_session,
    test_model,
    test_data,
    create_schema,
    update_schema,
):
    """Test that create works normally without CreateConfig."""
    # Create router WITHOUT CreateConfig
    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        path="/test_normal",
        tags=["Test"],
    )

    client.app.include_router(router)  # type: ignore

    # Normal create with all fields
    response = client.post("/test_normal", json={"name": "Normal Item", "tier_id": 1})

    assert response.status_code == 200, response.text

    # Verify the item was created normally
    stmt = select(test_model).where(test_model.name == "Normal Item")
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "Normal Item"
    assert fetched_record.tier_id == 1


@pytest.mark.asyncio
async def test_create_config_multiple_auto_fields(
    client: TestClient, async_session, test_model
):
    """Test CreateConfig with multiple auto_fields."""
    create_config = CreateConfig(
        auto_fields={
            "tier_id": lambda: 3,
            "name": lambda: "Auto Generated Name",
        },
        exclude_from_schema=[],
    )

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=EmptyCreateSchema,
        update_schema=EmptyUpdateSchema,
        create_config=create_config,
        path="/test_multi_auto",
        tags=["Test"],
    )

    client.app.include_router(router)  # type: ignore

    # Create item with empty payload - all fields auto-injected
    response = client.post("/test_multi_auto", json={})

    assert response.status_code == 200, response.text

    # Verify all fields were auto-injected
    stmt = select(test_model).where(test_model.name == "Auto Generated Name")
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "Auto Generated Name"
    assert fetched_record.tier_id == 3


@pytest.mark.asyncio
async def test_create_with_exclude_from_schema(
    client: TestClient, async_session, test_model, create_schema
):
    """Test that exclude_from_schema removes fields from the request schema."""
    # Config that auto-injects tier_id and excludes it from the request schema
    create_config = CreateConfig(
        auto_fields={
            "tier_id": lambda: 5,  # Auto-inject tier_id
        },
        exclude_from_schema=["tier_id"],  # Exclude from request schema
    )

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,  # Original schema includes tier_id
        update_schema=EmptyUpdateSchema,
        create_config=create_config,
        path="/test_exclude",
        tags=["Test"],
    )

    client.app.include_router(router)  # type: ignore

    # Create item WITHOUT tier_id in payload (should work because it's excluded)
    response = client.post("/test_exclude", json={"name": "Excluded Test"})

    assert response.status_code == 200, response.text

    # Verify the item was created with auto-injected tier_id
    stmt = select(test_model).where(test_model.name == "Excluded Test")
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "Excluded Test"
    assert fetched_record.tier_id == 5  # Auto-injected!


@pytest.mark.asyncio
async def test_update_with_exclude_from_schema(
    client: TestClient, async_session, test_model, create_schema, update_schema
):
    """Test that exclude_from_schema works for updates."""
    # First create an item
    test_item = test_model(name="Update Exclude Test", tier_id=1)
    async_session.add(test_item)
    await async_session.commit()
    await async_session.refresh(test_item)
    item_id = test_item.id

    # Config that auto-updates tier_id and excludes it from the request schema
    update_config = UpdateConfig(
        auto_fields={
            "tier_id": lambda: 10,
        },
        exclude_from_schema=["tier_id"],
    )

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        update_config=update_config,
        path="/test_update_exclude",
        tags=["Test"],
    )

    client.app.include_router(router)  # type: ignore

    # Update item WITHOUT tier_id in payload
    response = client.patch(
        f"/test_update_exclude/{item_id}", json={"name": "Updated Excluded"}
    )

    assert response.status_code == 200, response.text

    # Verify the item was updated with auto-injected tier_id
    stmt = select(test_model).where(test_model.id == item_id)
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "Updated Excluded"
    assert fetched_record.tier_id == 10  # Auto-injected!


@pytest.mark.asyncio
async def test_delete_with_auto_fields(
    client: TestClient, async_session, test_model, create_schema, update_schema
):
    """Test that auto_fields are injected during soft delete."""
    # First create an item
    test_item = test_model(name="Delete Test", tier_id=1)
    async_session.add(test_item)
    await async_session.commit()
    await async_session.refresh(test_item)
    item_id = test_item.id

    # Create router with DeleteConfig
    delete_config = DeleteConfig(
        auto_fields={
            "tier_id": lambda: 99,  # Auto-inject tier_id during delete
        }
    )

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        delete_config=delete_config,
        path="/test_delete_auto",
        tags=["Test"],
        is_deleted_column="is_deleted",
        deleted_at_column="deleted_at",
    )

    client.app.include_router(router)  # type: ignore

    # Soft delete the item
    response = client.delete(f"/test_delete_auto/{item_id}")

    assert response.status_code == 200, response.text
    assert response.json()["message"] == "Item deleted successfully"

    # Verify the item was soft deleted with auto-injected tier_id
    stmt = select(test_model).where(test_model.id == item_id)
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.tier_id == 99  # Auto-injected during delete!
    assert fetched_record.is_deleted is True  # Soft deleted


@pytest.mark.asyncio
async def test_delete_without_config_still_works(
    client: TestClient, async_session, test_model, create_schema, update_schema
):
    """Test that delete works normally without DeleteConfig."""
    # First create an item
    test_item = test_model(name="Normal Delete", tier_id=1)
    async_session.add(test_item)
    await async_session.commit()
    await async_session.refresh(test_item)
    item_id = test_item.id

    # Create router WITHOUT DeleteConfig
    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        path="/test_delete_normal",
        tags=["Test"],
        is_deleted_column="is_deleted",
        deleted_at_column="deleted_at",
    )

    client.app.include_router(router)  # type: ignore

    # Normal soft delete
    response = client.delete(f"/test_delete_normal/{item_id}")

    assert response.status_code == 200, response.text

    # Verify the item was soft deleted normally
    stmt = select(test_model).where(test_model.id == item_id)
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.tier_id == 1  # Unchanged
    assert fetched_record.is_deleted is True  # Soft deleted


@pytest.mark.asyncio
async def test_delete_config_multiple_auto_fields(
    client: TestClient, async_session, test_model, create_schema, update_schema
):
    """Test DeleteConfig with multiple auto_fields for audit trail."""
    # First create an item
    test_item = test_model(name="Multi Delete Test", tier_id=1)
    async_session.add(test_item)
    await async_session.commit()
    await async_session.refresh(test_item)
    item_id = test_item.id

    # Mock timestamp for testing
    fixed_timestamp = datetime(2024, 3, 15, 14, 30, 0)

    # Create router with DeleteConfig that sets multiple fields
    delete_config = DeleteConfig(
        auto_fields={
            "tier_id": lambda: 999,  # Changed to track deletion
            "deleted_at": lambda: fixed_timestamp,  # Set deletion timestamp
        }
    )

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        delete_config=delete_config,
        path="/test_delete_multi",
        tags=["Test"],
        is_deleted_column="is_deleted",
        deleted_at_column="deleted_at",
    )

    client.app.include_router(router)  # type: ignore

    # Soft delete with multiple auto fields
    response = client.delete(f"/test_delete_multi/{item_id}")

    assert response.status_code == 200, response.text

    # Verify all auto fields were injected
    stmt = select(test_model).where(test_model.id == item_id)
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.tier_id == 999  # Auto-injected!
    assert fetched_record.deleted_at == fixed_timestamp  # Auto-injected!
    assert fetched_record.is_deleted is True  # Soft deleted
