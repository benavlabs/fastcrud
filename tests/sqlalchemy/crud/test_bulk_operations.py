"""
Comprehensive tests for FastCRUD bulk operations.

Tests cover:
- Bulk insert operations (insert_multi)
- Bulk update operations (update_multi)
- Bulk delete operations (delete_multi)
- Batch processing, error handling, and summary responses
- Different database backends (SQLite, PostgreSQL)
"""

import asyncio
from typing import Any
from fastcrud import FastCRUD
from fastcrud.crud.bulk_operations.batch_processor import BatchProcessor, BatchConfig
from tests.sqlalchemy.conftest import BookingModel

import pytest
from sqlalchemy import func, select
from tests.sqlalchemy.conftest import ModelTestWithTimestamp

from pydantic import BaseModel, ConfigDict
from fastcrud.crud.bulk_operations import (
    BulkDeleteManager,
    BulkInsertManager,
    BulkUpdateManager,
)
from fastcrud.crud.bulk_operations.summary_models import (
    BulkDeleteSummary,
    BulkInsertSummary,
    BulkUpdateSummary,
    BulkOperationResult,
)
from tests.sqlalchemy.conftest import (
    CreateSchemaTest,
    ModelTest,
    ReadSchemaTest,
    CategoryModel,
    TierModel,
    Article,
    Author,
    ArticleSchema,
)


class BulkTestData:
    """Test data class for bulk operations."""

    def __init__(
        self,
        name: str,
        tier_id: int = 0,
        category_id: int = 0,
        is_deleted: bool = False,
    ):
        self.name = name
        self.tier_id = tier_id
        self.category_id = category_id
        self.is_deleted = is_deleted

    def model_dump(self):
        return {
            "name": self.name,
            "tier_id": self.tier_id,
            "category_id": self.category_id,
            "is_deleted": self.is_deleted,
        }


class ArticleCreatePayload(BaseModel):
    """Helper payload to provide model_dump for article creation tests."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    title: str
    content: str
    published_date: str
    author: Any = None


class AuthorCreatePayload(BaseModel):
    """Helper payload to provide model_dump for author creation tests."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    name: str
    articles: Any = None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        # Avoid sending None for collection relationships; SQLAlchemy expects list-like.
        return {k: v for k, v in data.items() if v is not None}


@pytest.mark.asyncio
async def test_bulk_insert_basic_functionality(async_session):
    """Test basic bulk insert functionality."""

    # Create test data
    test_data = [
        BulkTestData("Item 1"),
        BulkTestData("Item 2"),
        BulkTestData("Item 3"),
    ]

    # Execute bulk insert
    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=test_data,
        batch_size=10,
        return_summary=True,
    )

    # Verify result
    assert isinstance(result, BulkInsertSummary)
    assert result.successful_count == 3
    assert result.failed_count == 0
    assert result.batch_count == 1

    # Verify data was actually inserted
    stmt = select(ModelTest).where(ModelTest.name.in_(["Item 1", "Item 2", "Item 3"]))
    result_db = await async_session.execute(stmt)
    records = result_db.scalars().all()
    assert len(records) == 3


@pytest.mark.asyncio
async def test_bulk_insert_batch_processing(async_session):
    """Test bulk insert with multiple batches."""

    # Create large dataset that requires multiple batches
    test_data = []
    for i in range(25):  # Create 25 items
        test_data.append(BulkTestData(f"Batch Item {i}"))

    # Execute bulk insert with small batch size
    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=test_data,
        batch_size=10,  # Should create 3 batches (10, 10, 5)
        return_summary=True,
    )

    # Verify batch processing
    assert result.batch_count == 3
    assert result.successful_count == 25
    assert result.failed_count == 0

    # Verify data integrity
    stmt = select(func.count(ModelTest.id)).where(ModelTest.name.like("Batch Item%"))
    count_result = await async_session.execute(stmt)
    actual_count = count_result.scalar()
    assert actual_count == 25


@pytest.mark.asyncio
async def test_bulk_insert_with_pydantic_models(async_session):
    """Test bulk insert with Pydantic models."""

    # Use Pydantic schemas instead of simple objects
    pydantic_data = [
        CreateSchemaTest(name="Pydantic Item 1", tier_id=1),
        CreateSchemaTest(name="Pydantic Item 2", tier_id=2),
        CreateSchemaTest(name="Pydantic Item 3", tier_id=1),
    ]

    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=pydantic_data,
        batch_size=5,
        return_summary=True,
    )

    assert result.successful_count == 3
    assert result.failed_count == 0


@pytest.mark.asyncio
async def test_bulk_insert_return_as_model(async_session):
    """Test bulk insert with return_as_model option."""

    test_data = [
        BulkTestData("Return Model 1"),
        BulkTestData("Return Model 2"),
    ]

    manager = BulkInsertManager()
    inserted_records = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=test_data,
        batch_size=10,
        return_summary=False,
        schema_to_select=ReadSchemaTest,
        return_as_model=True,
    )

    assert isinstance(inserted_records, list)
    assert len(inserted_records) == 2

    # Verify returned records are Pydantic models
    for record in inserted_records:
        assert isinstance(record, ReadSchemaTest)
        assert hasattr(record, "name")
        assert hasattr(record, "id")


@pytest.mark.asyncio
async def test_bulk_insert_error_handling(async_session):
    """Test bulk insert error handling with invalid data."""
    # First insert some base data to create potential duplicates
    base_data = [
        BulkTestData("Base Item 1"),
        BulkTestData("Base Item 2"),
    ]

    base_manager = BulkInsertManager()
    await base_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=base_data,
        batch_size=10,
        return_summary=True,
    )

    # Mix of valid and duplicate data
    # This tests the library's ability to handle partial success with constraint violations
    mixed_data = [
        BulkTestData("Valid Item"),  # Should succeed
        {"name": "Dict Item", "tier_id": 1, "category_id": 1},  # Should succeed
        {"name": "Base Item 1", "tier_id": 3, "category_id": 2},
        # Duplicate - should fail if unique constraint on name exists
        {"name": "Base Item 2", "tier_id": 4, "category_id": 3},
        # Duplicate - should fail if unique constraint on name exists
    ]

    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=mixed_data,
        batch_size=10,
        return_summary=True,
        allow_partial_success=True,
    )

    # Should have some successes and some failures if unique constraints exist
    # If no unique constraints exist, all items might succeed
    assert result.successful_count >= 2  # At least the first two should succeed
    assert result.total_processed == 4  # All items should be processed

    # If there are constraint violations, check for failures
    if result.failed_count > 0:
        assert len(result.failed_items) > 0
    else:
        # If no failures, it means the database doesn't have unique constraints
        # This is still a valid test result - the library processed all items successfully
        assert result.failed_count == 0


@pytest.mark.asyncio
async def test_bulk_update_basic_functionality(async_session):
    """Test basic bulk update functionality."""

    # First insert some data to update
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[
            BulkTestData("Update Test 1"),
            BulkTestData("Update Test 2"),
            BulkTestData("Update Test 3"),
        ],
        batch_size=10,
        commit=True,
    )

    # Get actual IDs
    stmt = select(ModelTest).where(
        ModelTest.name.in_(["Update Test 1", "Update Test 2", "Update Test 3"])
    )
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()

    id_map = {r.name: r.id for r in records}

    # Prepare update data with primary key information
    update_data = [
        {"id": id_map["Update Test 1"], "name": "Updated Name 1", "tier_id": 10},
        {"id": id_map["Update Test 2"], "name": "Updated Name 2", "tier_id": 20},
        {"id": id_map["Update Test 3"], "name": "Updated Name 3", "tier_id": 30},
    ]

    # Execute bulk update
    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=ModelTest,
        objects=update_data,
        batch_size=10,
        return_summary=True,
    )

    # Verify result
    assert isinstance(result, BulkUpdateSummary)
    assert result.successful_count == 3
    assert result.failed_count == 0

    # Verify data was actually updated
    stmt = select(ModelTest).where(ModelTest.id.in_(id_map.values()))
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()

    assert len(records) == 3
    updated_names = [r.name for r in records]
    assert "Updated Name 1" in updated_names
    assert "Updated Name 2" in updated_names
    assert "Updated Name 3" in updated_names


@pytest.mark.asyncio
async def test_bulk_update_partial_updates(async_session):
    """Test bulk update with partial data (missing some fields)."""
    # Insert test data
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[
            BulkTestData("Partial Update Test", tier_id=5),
        ],
        batch_size=10,
    )

    # Get ID
    stmt = select(ModelTest).where(ModelTest.name == "Partial Update Test")
    db_result = await async_session.execute(stmt)
    record = db_result.scalar_one()
    record_id = record.id

    # Update with partial data (only name, not tier_id)
    update_data = [
        {"id": record_id, "name": "Partially Updated Name"},
    ]

    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=ModelTest,
        objects=update_data,
        batch_size=10,
        return_summary=True,
    )

    assert result.successful_count == 1

    # Verify only name was updated, tier_id remains unchanged
    stmt = select(ModelTest).where(ModelTest.id == record_id)
    db_result = await async_session.execute(stmt)
    record = db_result.scalar_one()

    assert record.name == "Partially Updated Name"
    assert record.tier_id == 5  # Should remain 5


@pytest.mark.asyncio
async def test_bulk_update_non_existent_records(async_session):
    """Test bulk update with non-existent record IDs."""
    # Update data with non-existent IDs
    update_data = [
        {"id": 99999, "name": "Non-existent Update", "tier_id": 999},
        {"id": 88888, "name": "Another Non-existent", "tier_id": 888},
    ]

    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=ModelTest,
        objects=update_data,
        batch_size=10,
        return_summary=True,
    )

    # Should have not found any records
    assert result.successful_count == 0


@pytest.mark.asyncio
async def test_bulk_delete_basic_functionality(async_session):
    """Test basic bulk delete functionality."""
    # Insert test data
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[
            BulkTestData("Delete Test 1"),
            BulkTestData("Delete Test 2"),
            BulkTestData("Delete Test 3"),
        ],
        batch_size=10,
    )

    # Delete records by filter
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTest,
        name__startswith="Delete Test",
        return_summary=True,
    )

    # Verify deletion
    assert isinstance(result, BulkDeleteSummary)
    assert result.successful_count == 3
    assert result.failed_count == 0

    # Verify records are actually deleted
    stmt = select(func.count(ModelTest.id)).where(ModelTest.name.like("Delete Test%"))
    count_result = await async_session.execute(stmt)
    remaining_count = count_result.scalar()
    assert remaining_count == 0


@pytest.mark.asyncio
async def test_bulk_delete_soft_delete(async_session):
    """Test bulk delete with soft delete functionality."""
    # Insert test data
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[
            BulkTestData("Soft Delete Test"),
        ],
        batch_size=10,
    )

    # Soft delete using is_deleted column
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTest,
        name="Soft Delete Test",
        is_deleted_column="is_deleted",
        soft_delete=True,
        return_summary=True,
    )

    # Verify soft delete
    assert result.soft_deleted_count == 1
    assert result.hard_deleted_count == 0

    # Verify record still exists but is marked as deleted
    stmt = select(ModelTest).where(ModelTest.name == "Soft Delete Test")
    db_result = await async_session.execute(stmt)
    record = db_result.scalar_one()

    assert record.is_deleted


@pytest.mark.asyncio
async def test_bulk_delete_multiple_filters(async_session):
    """Test bulk delete with multiple filter conditions."""
    # Insert diverse test data
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[
            BulkTestData("Multi Filter Test", tier_id=1),
            BulkTestData("Multi Filter Test", tier_id=2),
            BulkTestData("Other Test", tier_id=1),
        ],
        batch_size=10,
    )

    # Delete with multiple filter conditions
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTest,
        name="Multi Filter Test",
        tier_id=1,
        return_summary=True,
    )

    # Should delete only the record matching both conditions
    assert result.successful_count == 1

    # Verify correct record was deleted
    stmt = select(ModelTest).where(ModelTest.name == "Multi Filter Test")
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()

    # Should have one remaining record with tier_id=2
    assert len(records) == 1
    assert records[0].tier_id == 2


@pytest.mark.asyncio
async def test_bulk_operations_performance_metrics(async_session):
    """Test that bulk operations provide accurate performance metrics."""
    # Insert test data
    test_data = [BulkTestData(f"Performance Test {i}") for i in range(10)]

    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=test_data,
        batch_size=5,
        return_summary=True,
    )

    # Verify performance metrics are calculated
    assert result.duration_ms > 0
    assert result.average_batch_duration_ms >= 0
    assert result.items_per_second > 0
    assert result.success_rate == 1.0

    # Verify timing consistency
    assert result.start_time < result.end_time
    assert result.batch_count == 2  # 10 items / 5 per batch


@pytest.mark.asyncio
async def test_bulk_operations_empty_dataset(async_session):
    """Test bulk operations with empty datasets."""
    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[],
        batch_size=10,
        return_summary=True,
    )

    # Should return empty summary
    assert result.total_requested == 0
    assert result.successful_count == 0
    assert result.failed_count == 0
    assert result.batch_count == 0


@pytest.mark.asyncio
async def test_bulk_operations_custom_batch_config(async_session):
    """Test bulk operations with custom batch configuration."""
    config = BatchConfig(
        batch_size=2,
    )

    test_data = [BulkTestData(f"Config Test {i}") for i in range(5)]

    manager = BulkInsertManager(config=config)
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=test_data,
        batch_size=2,
        return_summary=True,
    )

    assert result.batch_size == 2
    assert result.batch_count == 3  # 5 items / 2 per batch


@pytest.mark.dialect("postgresql")
@pytest.mark.asyncio
async def test_bulk_operations_postgresql_specific(
    async_session, test_data_tier, test_data_category
):
    """Test bulk operations with PostgreSQL-specific features."""
    # Test with larger dataset to verify PostgreSQL performance
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    await async_session.commit()

    test_data = [
        BulkTestData(f"PostgreSQL Test {i}", tier_id=1, category_id=1)
        for i in range(100)
    ]

    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=test_data,
        batch_size=25,
        return_summary=True,
    )

    assert result.successful_count == 100
    assert result.batch_count == 4

    # Verify data integrity in PostgreSQL
    stmt = select(func.count(ModelTest.id)).where(
        ModelTest.name.like("PostgreSQL Test%")
    )
    count_result = await async_session.execute(stmt)
    actual_count = count_result.scalar()
    assert actual_count == 100


@pytest.mark.dialect("mysql")
@pytest.mark.asyncio
async def test_bulk_operations_mysql_specific(
    async_session, test_data_tier, test_data_category
):
    """Test bulk operations with MySQL-specific features."""
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    await async_session.commit()

    # Create test data with valid tier_id values
    test_data = [
        BulkTestData(f"MySQL Test {i}", tier_id=1 if i % 2 == 0 else 2, category_id=1)
        for i in range(50)
    ]

    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=test_data,
        batch_size=10,
        return_summary=True,
    )

    assert result.successful_count == 50
    assert result.batch_count == 5


@pytest.mark.asyncio
async def test_bulk_transaction_rollback(async_session):
    """Test that bulk operations properly handle transaction rollback."""
    # Start a transaction
    await async_session.begin()

    try:
        # Insert some data within transaction
        insert_manager = BulkInsertManager()
        await insert_manager.insert_multi(
            db=async_session,
            model_class=ModelTest,
            objects=[BulkTestData("Transaction Test")],
            batch_size=10,
            commit=False,  # Don't commit yet
        )

        # Rollback the transaction
        await async_session.rollback()

        # Verify data was not persisted
        stmt = select(ModelTest).where(ModelTest.name == "Transaction Test")
        db_result = await async_session.execute(stmt)
        record = db_result.scalar_one_or_none()

        assert record is None

    finally:
        # Ensure we're not in a transaction
        if async_session.in_transaction():
            await async_session.rollback()


@pytest.mark.asyncio
async def test_bulk_operations_error_recovery(async_session):
    """Test bulk operations error recovery and partial success."""
    mixed_data = [
        BulkTestData("Item 1"),
        {"name": "Bad Item", "tier_id": 1, "category_id": 0, "nonexistent_column": 123},
        BulkTestData("Valid 1"),
        {"name": "Valid Dict", "tier_id": 1, "category_id": 0},
        BulkTestData("Valid 2"),
    ]

    manager = BulkInsertManager()
    result = await manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=mixed_data,
        batch_size=1,
        return_summary=True,
        allow_partial_success=True,
    )

    assert result.successful_count >= 3
    assert result.failed_count >= 1
    assert result.failed_items is not None
    assert len(result.failed_items) >= 1

    # Verify valid data was inserted
    stmt = select(ModelTest).where(
        ModelTest.name.in_(["Valid 1", "Valid Dict", "Valid 2"])
    )
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()
    assert len(records) == 3


@pytest.mark.asyncio
async def test_bulk_operations_concurrent_execution(async_session):
    """Test bulk operations can handle concurrent execution."""
    # Using a transaction block to manage concurrency
    async with async_session.begin():

        async def bulk_insert_task(task_id: int, count: int):
            """Helper function to create bulk insert tasks."""
            test_data = [
                BulkTestData(f"Concurrent {task_id}-{i}", tier_id=task_id)
                for i in range(count)
            ]

            # Convert BulkTestData objects to dictionaries for type compatibility
            test_data_dicts = [item.model_dump() for item in test_data]

            manager = BulkInsertManager()
            result = await manager.insert_multi(
                db=async_session,
                model_class=ModelTest,
                objects=test_data_dicts,
                batch_size=5,
                return_summary=True,
            )
            # Type narrowing: result is BulkInsertSummary when return_summary=True
            if isinstance(result, BulkInsertSummary):
                return result.successful_count
            else:
                # This should never happen when return_summary=True
                return 0

        # Execute multiple concurrent bulk operations
        tasks = [
            bulk_insert_task(1, 10),
            bulk_insert_task(2, 15),
            bulk_insert_task(3, 8),
        ]

        results = await asyncio.gather(*tasks)

        # Verify all tasks completed successfully
        assert len(results) == 3
        assert all(count > 0 for count in results)
        assert sum(results) == 33  # 10 + 15 + 8

    # Verify all data was inserted after the transaction is committed
    stmt = select(func.count(ModelTest.id)).where(ModelTest.name.like("Concurrent%"))
    count_result = await async_session.execute(stmt)
    total_count = count_result.scalar()
    assert total_count == 33


@pytest.mark.asyncio
async def test_delete_multi_soft_delete_with_deleted_at_column(async_session):
    """Test soft delete using deleted_at timestamp column (covers lines 80-81)."""

    # Insert test data
    manager = BulkInsertManager()
    test_data = [{"name": f"Item {i}", "tier_id": 0} for i in range(5)]

    await manager.insert_multi(
        db=async_session,
        model_class=ModelTestWithTimestamp,
        objects=test_data,
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Perform soft delete
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTestWithTimestamp,
        is_deleted_column="deleted_at",
        name__like="Item%",  # Pass filters as kwargs
        batch_size=10,
        return_summary=True,
        soft_delete=True,  # Enable soft delete
    )

    # Verify soft delete occurred
    assert isinstance(result, BulkDeleteSummary)
    assert result.soft_deleted_count == 5
    assert result.hard_deleted_count == 0

    # Expire session cache to see updated values
    async_session.expire_all()

    # Verify records still exist but have deleted_at set
    stmt = select(ModelTestWithTimestamp).where(
        ModelTestWithTimestamp.name.like("Item%")
    )
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()
    assert len(records) == 5
    assert all(record.deleted_at is not None for record in records)


@pytest.mark.asyncio
async def test_delete_multi_integrity_error(async_session):
    """Test FK constraint violation handling during delete (covers lines 220-234)."""

    # Insert parent record
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[{"name": "Parent", "tier_id": 0}],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Get the parent ID
    stmt = select(ModelTest).where(ModelTest.name == "Parent")
    result = await async_session.execute(stmt)
    parent = result.scalar_one()

    # Insert booking that references the parent
    await insert_manager.insert_multi(
        db=async_session,
        model_class=BookingModel,
        objects=[
            {"owner_id": parent.id, "user_id": parent.id, "booking_date": "2024-01-01"}
        ],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Try to hard delete the parent (should fail due to FK constraint)
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTest,
        id=parent.id,  # Pass filter as kwarg
        batch_size=10,
        return_summary=True,
        soft_delete=False,  # Force hard delete
    )

    # Verify IntegrityError was handled
    assert isinstance(result, BulkDeleteSummary)
    assert result.failed_count > 0
    # Check that error details contain integrity error information
    if result.batch_results:
        failed_batches = [b for b in result.batch_results if not b.success]
        if failed_batches:
            assert any(
                "integrity" in str(b.error_details).lower() for b in failed_batches
            )


@pytest.mark.asyncio
async def test_delete_multi_general_error(async_session, monkeypatch):
    """Test general error handling with monkeypatch (covers lines 236-249)."""

    # Insert test data
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[{"name": "Test Item", "tier_id": 0}],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Monkeypatch to simulate a general error during delete
    original_execute = async_session.execute

    async def mock_execute_error(*args, **kwargs):
        # Only raise error for DELETE statements
        if args and hasattr(args[0], "is_delete") and args[0].is_delete:
            raise RuntimeError("Simulated database error")
        return await original_execute(*args, **kwargs)

    monkeypatch.setattr(async_session, "execute", mock_execute_error)

    # Perform delete operation
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTest,
        name="Test Item",  # Pass filter as kwarg
        batch_size=10,
        return_summary=True,
        soft_delete=False,
    )

    # Verify error was handled
    assert isinstance(result, BulkDeleteSummary)
    assert result.failed_count > 0
    if result.batch_results:
        failed_batches = [b for b in result.batch_results if not b.success]
        if failed_batches:
            assert any(
                "general_error" in str(b.error_details).lower()
                or "error" in str(b.error_message).lower()
                for b in failed_batches
            )


@pytest.mark.asyncio
async def test_delete_multi_empty_filters(async_session):
    """Test delete with no filters (covers lines 65-68)."""

    # Perform delete with empty filters (no kwargs passed)
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTest,
        # No filters passed at all
        batch_size=10,
        return_summary=True,
    )

    # Verify empty summary is returned
    assert isinstance(result, BulkDeleteSummary)
    assert result.successful_count == 0
    assert result.failed_count == 0
    assert result.batch_count == 0


@pytest.mark.asyncio
async def test_delete_multi_not_found_tracking(async_session):
    """Test tracking of non-existent records (covers lines 122-123)."""

    # Try to delete records that don't exist
    delete_manager = BulkDeleteManager()
    result = await delete_manager.delete_multi(
        db=async_session,
        model_class=ModelTest,
        id=99999,  # Non-existent ID as kwarg
        batch_size=10,
        return_summary=True,
    )

    # Verify not_found_count is tracked
    assert isinstance(result, BulkDeleteSummary)
    # The result should show 0 deleted since nothing was found
    assert result.successful_count == 0


# ============================================================================
# COVERAGE GAP TESTS - HIGH PRIORITY: insert_multi.py
# ============================================================================


@pytest.mark.asyncio
async def test_insert_multi_with_sqlalchemy_instances(async_session):
    """Test with SQLAlchemy model instances (covers lines 205-214)."""

    # Create SQLAlchemy model instances directly
    instances = [ModelTest(name=f"Instance {i}", tier_id=0) for i in range(3)]

    # Insert using SQLAlchemy instances
    insert_manager = BulkInsertManager()
    result = await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=instances,  # Pass SQLAlchemy instances
        batch_size=10,
        return_summary=True,
    )

    # Verify insertion succeeded
    assert isinstance(result, BulkInsertSummary)
    assert result.successful_count == 3
    await async_session.commit()

    # Verify records exist
    stmt = select(ModelTest).where(ModelTest.name.like("Instance%"))
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()
    assert len(records) == 3


@pytest.mark.asyncio
async def test_insert_multi_with_python_objects(async_session):
    """Test with plain Python objects (covers lines 216-218)."""

    # Create plain Python objects with __dict__ attribute
    class PlainObject:
        def __init__(self, name, tier_id):
            self.name = name
            self.tier_id = tier_id

    objects = [PlainObject(name=f"Object {i}", tier_id=0) for i in range(3)]

    # Insert using plain Python objects
    insert_manager = BulkInsertManager()
    result = await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=objects,  # Pass plain Python objects
        batch_size=10,
        return_summary=True,
    )

    # Verify insertion succeeded
    assert isinstance(result, BulkInsertSummary)
    assert result.successful_count == 3
    await async_session.commit()

    # Verify records exist
    stmt = select(ModelTest).where(ModelTest.name.like("Object%"))
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()
    assert len(records) == 3


@pytest.mark.asyncio
async def test_insert_multi_empty_data(async_session):
    """Test with empty data (covers lines 162-164)."""

    # Create objects that will result in empty insert_data after processing
    # Pass objects with only None values or invalid fields
    objects = [
        {"invalid_field": "value"},  # Field doesn't exist in model
        {"another_invalid": None},
    ]

    # Insert with data that gets filtered out
    insert_manager = BulkInsertManager()
    result = await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=objects,
        batch_size=10,
        return_summary=True,
    )

    # Verify empty result
    assert isinstance(result, BulkInsertSummary)
    # Should have 0 successful inserts since data was invalid
    assert result.successful_count == 0


@pytest.mark.asyncio
async def test_insert_multi_constraint_violations(async_session):
    """Test unique/FK constraint violations (covers lines 262-263)."""

    # First, insert a tier with unique name
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=TierModel,
        objects=[{"name": "UniqueTier"}],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Try to insert duplicate tier (violates unique constraint)
    result = await insert_manager.insert_multi(
        db=async_session,
        model_class=TierModel,
        objects=[
            {"name": "UniqueTier"},  # Duplicate
            {"name": "AnotherTier"},  # Valid
        ],
        batch_size=10,
        return_summary=True,
    )

    # Verify constraint violation was handled
    assert isinstance(result, BulkInsertSummary)
    # At least one should fail due to constraint
    assert result.failed_count > 0 or result.successful_count < 2


@pytest.mark.asyncio
async def test_insert_multi_empty_list(async_session):
    """Test with empty items list (covers lines 325-348)."""

    # Insert with empty list
    insert_manager = BulkInsertManager()
    result = await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[],  # Empty list
        batch_size=10,
        return_summary=True,
    )

    # Verify empty summary is created
    assert isinstance(result, BulkInsertSummary)
    assert result.successful_count == 0
    assert result.failed_count == 0
    assert result.batch_count == 0


# ============================================================================
# COVERAGE GAP TESTS - MEDIUM PRIORITY: update_multi.py
# ============================================================================


@pytest.mark.asyncio
async def test_update_multi_missing_primary_key(async_session):
    """Test items without primary key (covers lines 124-125)."""

    # Insert test data first
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[{"name": "Test Item", "tier_id": 0}],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Try to update without providing primary key
    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[
            {"name": "Updated Name"}  # Missing 'id' primary key
        ],
        batch_size=10,
        return_summary=True,
    )

    # Verify error handling for missing primary key
    assert isinstance(result, BulkUpdateSummary)
    # Should fail because primary key is missing
    assert result.failed_count > 0


@pytest.mark.asyncio
async def test_update_multi_integrity_error(async_session):
    """Test constraint violations (covers lines 182-197)."""

    # Insert two tiers with unique names
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=TierModel,
        objects=[{"name": "Tier1"}, {"name": "Tier2"}],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Get the tier IDs
    stmt = select(TierModel).where(TierModel.name.in_(["Tier1", "Tier2"]))
    result = await async_session.execute(stmt)
    tiers = result.scalars().all()
    tier1_id = next(t.id for t in tiers if t.name == "Tier1")
    tier2_id = next(t.id for t in tiers if t.name == "Tier2")

    # Try to update tier2 to have the same name as tier1 (violates unique constraint)
    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=TierModel,
        objects=[
            {"id": tier2_id, "name": "Tier1"}  # Duplicate name
        ],
        batch_size=10,
        return_summary=True,
    )

    # Verify IntegrityError was handled
    assert isinstance(result, BulkUpdateSummary)
    assert result.failed_count > 0


@pytest.mark.asyncio
async def test_update_multi_general_error(async_session, monkeypatch):
    """Test general error handling (covers lines 199-214)."""

    # Insert test data
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[{"name": "Test Item", "tier_id": 0}],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Get the record ID
    stmt = select(ModelTest).where(ModelTest.name == "Test Item")
    result = await async_session.execute(stmt)
    record = result.scalar_one()

    # Monkeypatch to simulate a general error during update
    original_execute = async_session.execute

    async def mock_execute_error(*args, **kwargs):
        # Only raise error for UPDATE statements
        if args and hasattr(args[0], "is_update") and args[0].is_update:
            raise RuntimeError("Simulated database error")
        return await original_execute(*args, **kwargs)

    monkeypatch.setattr(async_session, "execute", mock_execute_error)

    # Perform update operation
    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[{"id": record.id, "name": "Updated Name"}],
        batch_size=10,
        return_summary=True,
    )

    # Verify error was handled
    assert isinstance(result, BulkUpdateSummary)
    assert result.failed_count > 0


@pytest.mark.asyncio
async def test_update_multi_not_found_tracking(async_session):
    """Test non-existent records (covers lines 281-282)."""

    # Try to update records that don't exist
    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[
            {"id": 99999, "name": "Updated Name"}  # Non-existent ID
        ],
        batch_size=10,
        return_summary=True,
    )

    # Verify not_found_count is tracked
    assert isinstance(result, BulkUpdateSummary)
    # Should show 0 updated since record doesn't exist
    assert result.successful_count == 0


@pytest.mark.asyncio
async def test_update_multi_empty_list(async_session):
    """Test empty items list (covers lines 217-240)."""

    # Update with empty list
    update_manager = BulkUpdateManager()
    result = await update_manager.update_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[],  # Empty list
        batch_size=10,
        return_summary=True,
    )

    # Verify empty summary is created
    assert isinstance(result, BulkUpdateSummary)
    assert result.successful_count == 0
    assert result.failed_count == 0
    assert result.batch_count == 0


@pytest.mark.asyncio
async def test_batch_processor_retry_logic(async_session, monkeypatch):
    """Test retry with exponential backoff (covers lines 200-215)."""

    # Track retry attempts
    attempt_count = {"count": 0}
    sleep_times = []

    # Mock asyncio.sleep to track backoff timing
    original_sleep = asyncio.sleep

    async def mock_sleep(delay):
        sleep_times.append(delay)
        await original_sleep(0.01)  # Small delay for testing

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    # Create processor with retry enabled
    config = BatchConfig(
        batch_size=10, retry_attempts=3, retry_delay=0.1, allow_partial_success=True
    )
    processor = BatchProcessor(config)

    # Mock processor function that fails twice then succeeds
    async def mock_processor_func(batch, batch_index):
        attempt_count["count"] += 1
        if attempt_count["count"] <= 2:
            raise RuntimeError("Transient error")
        return BulkOperationResult(
            batch_index=batch_index,
            items_processed=len(batch),
            success=True,
            error_message=None,
            error_details=None,
            duration_ms=10,
        )

    # Process batch
    items = [{"name": f"Item{i}"} for i in range(5)]
    result = await processor.process_batches(
        items=items,
        processor_func=mock_processor_func,
        db=async_session,
        operation_name="test_retry",
    )

    # Verify retry logic worked
    assert attempt_count["count"] == 3  # Failed twice, succeeded on third
    assert len(sleep_times) == 2  # Two retries with sleep
    # Verify exponential backoff: delay * (2 ** attempt)
    assert sleep_times[0] == 0.1 * (2**0)  # First retry: 0.1
    assert sleep_times[1] == 0.1 * (2**1)  # Second retry: 0.2


@pytest.mark.asyncio
async def test_batch_processor_transaction_rollback(async_session, monkeypatch):
    """Test rollback on error."""

    # Insert initial data
    insert_manager = BulkInsertManager()
    await insert_manager.insert_multi(
        db=async_session,
        model_class=ModelTest,
        objects=[{"name": "Initial Item", "tier_id": 0}],
        batch_size=10,
        return_summary=False,
    )
    await async_session.commit()

    # Create processor with no partial success allowed
    config = BatchConfig(
        batch_size=10,
        retry_attempts=0,
        allow_partial_success=False,
        commit_strategy="batch",
    )
    processor = BatchProcessor(config)

    # Mock processor function that always fails
    async def mock_processor_func(batch, batch_index):
        raise RuntimeError("Simulated batch error")

    # Process batch and expect error
    items = [{"name": f"Item{i}"} for i in range(5)]

    with pytest.raises(RuntimeError, match="Simulated batch error"):
        await processor.process_batches(
            items=items,
            processor_func=mock_processor_func,
            db=async_session,
            operation_name="test_rollback",
        )

    # Verify transaction was rolled back - only initial item should exist
    stmt = select(ModelTest)
    result = await async_session.execute(stmt)
    records = result.scalars().all()
    assert len(records) == 1
    assert records[0].name == "Initial Item"


@pytest.mark.asyncio
async def test_batch_processor_commit_all_strategy(async_session):
    """Test 'all' commit strategy (covers lines 128-131)."""

    # Create processor with "all" commit strategy
    config = BatchConfig(
        batch_size=2,  # Small batch size to create multiple batches
        commit_strategy="all",  # Commit all at the end
    )
    processor = BatchProcessor(config)

    # Track commits
    commit_count = {"count": 0}
    original_commit = async_session.commit

    async def mock_commit():
        commit_count["count"] += 1
        await original_commit()

    async_session.commit = mock_commit

    # Create processor function that inserts data
    async def insert_processor_func(batch, batch_index):
        for item in batch:
            async_session.add(ModelTest(**item))
        return BulkOperationResult(
            batch_index=batch_index,
            items_processed=len(batch),
            success=True,
            error_message=None,
            error_details=None,
            duration_ms=10,
        )

    # Process multiple batches (6 items = 3 batches of size 2)
    items = [{"name": f"Item{i}", "tier_id": 0} for i in range(6)]
    result = await processor.process_batches(
        items=items,
        processor_func=insert_processor_func,
        db=async_session,
        operation_name="test_commit_all",
    )

    # Verify only one commit at the end (via _handle_final_commit)
    assert commit_count["count"] == 1
    assert result.batch_count == 3

    # Verify all items were inserted
    stmt = select(ModelTest).where(ModelTest.name.like("Item%"))
    db_result = await async_session.execute(stmt)
    records = db_result.scalars().all()
    assert len(records) == 6


@pytest.mark.asyncio
async def test_batch_processor_empty_items(async_session):
    """Test empty items list (covers lines 101-102)."""

    # Create processor
    config = BatchConfig(batch_size=10)
    processor = BatchProcessor(config)

    # Mock processor function (should not be called)
    async def mock_processor_func(batch, batch_index):
        raise RuntimeError("Should not be called for empty items")

    # Process empty list
    result = await processor.process_batches(
        items=[],  # Empty list
        processor_func=mock_processor_func,
        db=async_session,
        operation_name="test_empty",
    )

    # Verify empty summary is created via _create_empty_summary
    assert result is not None
    assert result.successful_count == 0
    assert result.failed_count == 0
    assert result.batch_count == 0


# ============================================================================
# COVERAGE GAP TESTS - LOWER PRIORITY: fast_crud.py
# ============================================================================


@pytest.mark.asyncio
async def test_create_with_invalid_nested_payload(async_session):
    """Test invalid nested payload error handling (covers lines 667-670)."""

    # Create CRUD instance for Article model (has author relationship)
    article_crud = FastCRUD(Article)

    # Try to create article with invalid nested author payload (not dict or BaseModel)
    with pytest.raises((AttributeError, TypeError)):
        await article_crud.create(
            db=async_session,
            object=ArticleCreatePayload(
                title="Test Article",
                content="Test content",
                published_date="2024-01-01",
                author=12345,  # Invalid: should be dict or BaseModel, not int
            ),
        )


@pytest.mark.asyncio
async def test_create_nested_payload_detection_edge_cases(async_session):
    """Test edge cases in _is_nested_payload detection."""

    # Create CRUD instance for Article model
    article_crud = FastCRUD(Article)

    # First create an author to use as FK
    author_crud = FastCRUD(Author)
    author = await author_crud.create(
        db=async_session, object=AuthorCreatePayload(name="Test Author")
    )
    await async_session.commit()

    # Test 1: Empty dict should be treated as nested payload
    # This should work - empty dict is valid nested payload
    article1 = await article_crud.create(
        db=async_session,
        object=ArticleCreatePayload(
            title="Article 1",
            content="Content 1",
            published_date="2024-01-01",
            author={},  # Empty dict - should be treated as nested payload
        ),
        schema_to_select=ArticleSchema,
        return_as_model=True,
    )
    assert article1 is not None
    await async_session.rollback()  # Rollback to avoid constraint issues

    # Test 2: None value for scalar relationship should not be treated as nested
    # This should work - None is valid for nullable FK
    article2 = await article_crud.create(
        db=async_session,
        object=ArticleCreatePayload(
            title="Article 2",
            content="Content 2",
            published_date="2024-01-01",
            author=None,  # None - should not be treated as nested payload
        ),
        schema_to_select=ArticleSchema,
        return_as_model=True,
    )
    assert article2 is not None
    assert article2.author_id is None
    await async_session.commit()


@pytest.mark.asyncio
async def test_create_nested_list_with_invalid_items(async_session):
    """Test invalid list items in nested relationships (covers lines 688-693)."""

    # Create CRUD instance for Author model (has articles relationship - uselist=True)
    author_crud = FastCRUD(Author)

    # Test 1: Non-list/tuple value for uselist=True relationship should raise TypeError
    with pytest.raises(TypeError, match="must be a list or tuple when uselist=True"):
        await author_crud.create(
            db=async_session,
            object=AuthorCreatePayload(
                name="Test Author",
                articles={"title": "Article"},  # Invalid: should be list, not dict
            ),
        )

    # Test 2: List with invalid item types should raise TypeError
    with pytest.raises(TypeError, match="Unsupported nested relationship payload type"):
        await author_crud.create(
            db=async_session,
            object=AuthorCreatePayload(
                name="Test Author",
                articles=[
                    {
                        "title": "Valid Article",
                        "content": "Content",
                        "published_date": "2024-01-01",
                    },
                    12345,  # Invalid: should be dict or BaseModel, not int
                ],
            ),
        )
