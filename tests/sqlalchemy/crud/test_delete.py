import pytest
from typing import Optional
from sqlalchemy import select
from fastcrud.crud.fast_crud import FastCRUD
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from pydantic import BaseModel


@pytest.mark.asyncio
async def test_db_delete_hard_delete(async_session, test_data_tier, tier_model):
    for tier_item in test_data_tier:
        async_session.add(tier_model(**tier_item))
    await async_session.commit()

    crud = FastCRUD(tier_model)
    some_existing_id = test_data_tier[0]["id"]
    await crud.db_delete(db=async_session, id=some_existing_id)

    deleted_record = await async_session.execute(
        select(tier_model).where(tier_model.id == some_existing_id)
    )
    assert deleted_record.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_delete_soft_delete(async_session, test_data, test_model):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    some_existing_id = test_data[0]["id"]
    await crud.delete(db=async_session, id=some_existing_id)

    soft_deleted_record = await async_session.execute(
        select(test_model).where(test_model.id == some_existing_id)
    )
    soft_deleted = soft_deleted_record.scalar_one()
    assert soft_deleted.is_deleted is True
    assert soft_deleted.deleted_at is not None


@pytest.mark.asyncio
async def test_delete_hard_delete_as_fallback(
    async_session, test_data_tier, tier_model
):
    for tier_item in test_data_tier:
        async_session.add(tier_model(**tier_item))
    await async_session.commit()

    crud = FastCRUD(tier_model)
    some_existing_id = test_data_tier[0]["id"]
    await crud.delete(db=async_session, id=some_existing_id)

    hard_deleted_record = await async_session.execute(
        select(tier_model).where(tier_model.id == some_existing_id)
    )
    assert hard_deleted_record.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_delete_multiple_records(async_session, test_data, test_model):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    with pytest.raises(Exception):
        await crud.delete(db=async_session, allow_multiple=False, tier_id=1)


@pytest.mark.asyncio
async def test_get_with_advanced_filters(async_session, test_data, test_model):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    records = await crud.get_multi(db=async_session, id__gt=5)
    for record in records["data"]:
        assert record["id"] > 5, "All fetched records should have 'id' greater than 5"


@pytest.mark.asyncio
async def test_soft_delete_with_custom_columns(async_session, test_data, test_model):
    crud = FastCRUD(
        test_model, is_deleted_column="is_deleted", deleted_at_column="deleted_at"
    )
    some_existing_id = test_data[0]["id"]

    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    await crud.delete(db=async_session, id=some_existing_id, allow_multiple=False)

    deleted_record = await async_session.execute(
        select(test_model)
        .where(test_model.id == some_existing_id)
        .where(getattr(test_model, "is_deleted") == True)  # noqa
    )
    deleted_record = deleted_record.scalar_one_or_none()

    assert deleted_record is not None, "Record should exist after soft delete"
    assert (
        getattr(deleted_record, "is_deleted") == True  # noqa
    ), "Record should be marked as soft deleted"
    assert (
        getattr(deleted_record, "deleted_at") is not None
    ), "Record should have a deletion timestamp"


@pytest.mark.asyncio
async def test_soft_delete_custom_columns(async_session, test_model, test_data):
    crud = FastCRUD(
        test_model,
        is_deleted_column="custom_is_deleted",
        deleted_at_column="custom_deleted_at",
    )
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    existing_record = await crud.get(async_session, id=test_data[0]["id"])
    assert existing_record is not None, "Record should exist before deletion"

    await crud.delete(async_session, id=test_data[0]["id"], allow_multiple=False)

    deleted_record = await crud.get(async_session, id=test_data[0]["id"])
    assert (
        deleted_record is None
    ), "Custom columns not found, so record should be deleted."


@pytest.mark.asyncio
async def test_db_delete_disallow_multiple_matches(
    async_session, test_data, test_model
):
    tier_id_for_multiple_records = 1
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    with pytest.raises(MultipleResultsFound):
        await crud.db_delete(
            db=async_session, allow_multiple=False, tier_id=tier_id_for_multiple_records
        )

    remaining_records = await async_session.execute(
        select(test_model).where(test_model.tier_id == tier_id_for_multiple_records)
    )
    assert remaining_records.scalars().all(), "No records should have been deleted"


@pytest.mark.asyncio
async def test_soft_delete_db_row_provided(async_session, test_data, test_model):
    test_record = test_model(**test_data[0])
    async_session.add(test_record)
    await async_session.commit()

    crud = FastCRUD(
        test_model, is_deleted_column="is_deleted", deleted_at_column="deleted_at"
    )

    db_row = await async_session.get(test_model, test_record.id)

    await crud.delete(db=async_session, db_row=db_row)

    soft_deleted_record = await async_session.get(test_model, test_record.id)
    assert soft_deleted_record.is_deleted
    assert soft_deleted_record.deleted_at is not None


@pytest.mark.asyncio
async def test_hard_delete_db_row_provided(async_session, test_data_tier, tier_model):
    test_record = tier_model(**test_data_tier[0])
    async_session.add(test_record)
    await async_session.commit()

    crud = FastCRUD(tier_model)
    db_row = await async_session.get(tier_model, test_record.id)

    await crud.delete(db=async_session, db_row=db_row)

    deleted_record = await async_session.get(tier_model, test_record.id)
    assert deleted_record is None


@pytest.mark.asyncio
async def test_delete_no_records_match_filters_raises_no_result_found(
    async_session, test_data, test_model
):
    crud = FastCRUD(test_model)
    non_matching_filter_criteria = {"id": 99999}

    with pytest.raises(NoResultFound):
        await crud.delete(db=async_session, **non_matching_filter_criteria)


# Test classes for new typing functionality
class DeleteTestSchema(BaseModel):
    id: int | None = None
    name: str | None = None


class DeleteTierSchema(BaseModel):
    id: int | None = None
    name: str | None = None


# Tests for new filters parameter functionality
@pytest.mark.asyncio
async def test_delete_with_filters_schema(async_session, test_data, test_model):
    """Test delete method with filters parameter using Pydantic schema."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    some_existing_id = test_data[0]["id"]

    # Create delete filter schema
    delete_filters = DeleteTestSchema(id=some_existing_id)

    await crud.delete(db=async_session, filters=delete_filters)

    soft_deleted_record = await async_session.execute(
        select(test_model).where(test_model.id == some_existing_id)
    )
    soft_deleted = soft_deleted_record.scalar_one()
    assert soft_deleted.is_deleted is True
    assert soft_deleted.deleted_at is not None


@pytest.mark.asyncio
async def test_db_delete_with_filters_schema(async_session, test_data_tier, tier_model):
    """Test db_delete method with filters parameter using Pydantic schema."""
    for tier_item in test_data_tier:
        async_session.add(tier_model(**tier_item))
    await async_session.commit()

    crud = FastCRUD(tier_model)
    some_existing_id = test_data_tier[0]["id"]

    # Create delete filter schema
    delete_filters = DeleteTierSchema(id=some_existing_id)

    await crud.db_delete(db=async_session, filters=delete_filters)

    deleted_record = await async_session.execute(
        select(tier_model).where(tier_model.id == some_existing_id)
    )
    assert deleted_record.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_delete_with_combined_filters(async_session, test_data, test_model):
    """Test delete method with both filters schema and kwargs."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    some_existing_id = test_data[0]["id"]
    some_existing_name = test_data[0]["name"]

    # Create delete filter schema with partial filters
    delete_filters = DeleteTestSchema(id=some_existing_id)

    # Add additional filters via kwargs
    await crud.delete(db=async_session, filters=delete_filters, name=some_existing_name)

    soft_deleted_record = await async_session.execute(
        select(test_model).where(test_model.id == some_existing_id)
    )
    soft_deleted = soft_deleted_record.scalar_one()
    assert soft_deleted.is_deleted is True
    assert soft_deleted.deleted_at is not None


@pytest.mark.asyncio
async def test_db_delete_with_combined_filters(
    async_session, test_data_tier, tier_model
):
    """Test db_delete method with both filters schema and kwargs."""
    for tier_item in test_data_tier:
        async_session.add(tier_model(**tier_item))
    await async_session.commit()

    crud = FastCRUD(tier_model)
    some_existing_id = test_data_tier[0]["id"]
    some_existing_name = test_data_tier[0]["name"]

    # Create delete filter schema with partial filters
    delete_filters = DeleteTierSchema(id=some_existing_id)

    # Add additional filters via kwargs
    await crud.db_delete(
        db=async_session, filters=delete_filters, name=some_existing_name
    )

    deleted_record = await async_session.execute(
        select(tier_model).where(tier_model.id == some_existing_id)
    )
    assert deleted_record.scalar_one_or_none() is None


# Tests for safeguards against accidental deletion
@pytest.mark.asyncio
async def test_delete_no_filters_raises_value_error(
    async_session, test_data, test_model
):
    """Test that delete method raises ValueError when no filters are provided."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    with pytest.raises(ValueError, match="No filters provided"):
        await crud.delete(db=async_session)


@pytest.mark.asyncio
async def test_db_delete_no_filters_raises_value_error(
    async_session, test_data_tier, tier_model
):
    """Test that db_delete method raises ValueError when no filters are provided."""
    for tier_item in test_data_tier:
        async_session.add(tier_model(**tier_item))
    await async_session.commit()

    crud = FastCRUD(tier_model)

    with pytest.raises(ValueError, match="No filters provided"):
        await crud.db_delete(db=async_session)


@pytest.mark.asyncio
async def test_delete_empty_filters_schema_raises_value_error(
    async_session, test_data, test_model
):
    """Test that delete method raises ValueError when empty filters schema is provided."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Create empty delete filter schema
    delete_filters = DeleteTestSchema()

    with pytest.raises(ValueError, match="No filters provided"):
        await crud.delete(db=async_session, filters=delete_filters)


@pytest.mark.asyncio
async def test_db_delete_empty_filters_schema_raises_value_error(
    async_session, test_data_tier, tier_model
):
    """Test that db_delete method raises ValueError when empty filters schema is provided."""
    for tier_item in test_data_tier:
        async_session.add(tier_model(**tier_item))
    await async_session.commit()

    crud = FastCRUD(tier_model)

    # Create empty delete filter schema
    delete_filters = DeleteTierSchema()

    with pytest.raises(ValueError, match="No filters provided"):
        await crud.db_delete(db=async_session, filters=delete_filters)


# Tests for backward compatibility
@pytest.mark.asyncio
async def test_delete_backward_compatibility_with_kwargs(
    async_session, test_data, test_model
):
    """Test that delete method still works with only kwargs (backward compatibility)."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    some_existing_id = test_data[0]["id"]

    # Use only kwargs (old way)
    await crud.delete(db=async_session, id=some_existing_id)

    soft_deleted_record = await async_session.execute(
        select(test_model).where(test_model.id == some_existing_id)
    )
    soft_deleted = soft_deleted_record.scalar_one()
    assert soft_deleted.is_deleted is True
    assert soft_deleted.deleted_at is not None


@pytest.mark.asyncio
async def test_db_delete_backward_compatibility_with_kwargs(
    async_session, test_data_tier, tier_model
):
    """Test that db_delete method still works with only kwargs (backward compatibility)."""
    for tier_item in test_data_tier:
        async_session.add(tier_model(**tier_item))
    await async_session.commit()

    crud = FastCRUD(tier_model)
    some_existing_id = test_data_tier[0]["id"]

    # Use only kwargs (old way)
    await crud.db_delete(db=async_session, id=some_existing_id)

    deleted_record = await async_session.execute(
        select(tier_model).where(tier_model.id == some_existing_id)
    )
    assert deleted_record.scalar_one_or_none() is None
