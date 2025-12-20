"""
Bulk insert operations for FastCRUD.

This module provides efficient bulk insert capabilities with batching,
error handling, and comprehensive reporting.
"""

import time
from datetime import datetime
from functools import partial
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .batch_processor import BatchConfig, BatchProcessor
from .summary_models import BulkInsertSummary, BulkOperationResult


class BulkInsertManager:
    """
    Manager for bulk insert operations.
    This class handles bulk inserting data with proper batching,
    error handling, transaction management, and detailed reporting.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.batch_processor = BatchProcessor(self.config)
        self._current_db_session: Optional[AsyncSession] = None

    async def insert_multi(
            self,
            db: AsyncSession,
            model_class: Any,
            objects: List[Union[dict, Any]],
            *,
            batch_size: int = 1000,
            commit: bool = True,
            allow_partial_success: bool = True,
            return_summary: bool = False,
            schema_to_select: Optional[Type[BaseModel]] = None,
            return_as_model: bool = False,
    ) -> Union[BulkInsertSummary, List[Union[dict, Any]]]:
        """
        Insert multiple objects efficiently with batch processing.

        Args:
            db: Database session
            model_class: SQLAlchemy model class
            objects: List of objects to insert (dicts or Pydantic models)
            batch_size: Number of objects to insert per batch
            commit: Whether to commit the transaction
            allow_partial_success: Whether to continue on errors
            return_summary: Whether to return detailed summary
            schema_to_select: Pydantic schema for return format
            return_as_model: Whether to return as model instances

        Returns:
            Either BulkInsertSummary (if return_summary=True) or a list of inserted records
        """
        if not objects:
            return self._handle_empty_input(batch_size, return_summary)

        # Create and setup configuration
        config = self._create_operation_config(batch_size, commit, allow_partial_success)
        processor = BatchProcessor(config)

        # Set context for the processor
        self._current_db_session = db
        try:
            process_func = partial(
                self._process_insert_batch,
                model_class=model_class,
                schema_to_select=schema_to_select,
                return_as_model=return_as_model
            )

            result = await processor.process_batches(
                objects,
                process_func,
                db=db,
                operation_name="insert"
            )
        finally:
            self._current_db_session = None

        if return_summary:
            return self._create_summary_result(result)

        return self._extract_inserted_records(result)

    def _handle_empty_input(self, batch_size: int, return_summary: bool) -> Union[BulkInsertSummary, List[Any]]:
        if return_summary:
            return self._create_empty_summary(batch_size=batch_size)
        return []

    def _create_operation_config(self, batch_size: int, commit: bool, allow_partial_success: bool) -> BatchConfig:
        base_config = self.config or BatchConfig()
        return BatchConfig(
            batch_size=batch_size,
            enable_transactions=True,
            commit_strategy="batch" if commit else "never",
            allow_partial_success=allow_partial_success,
            max_workers=getattr(base_config, "max_workers", 4),
            timeout_seconds=getattr(base_config, "timeout_seconds", None),
            retry_attempts=getattr(base_config, "retry_attempts", 0)
        )

    def _create_summary_result(self, result: Any) -> BulkInsertSummary:
        summary = BulkInsertSummary(**result.model_dump())
        dup_count, constraint_count = self._calculate_error_metrics(result)
        summary.duplicate_count = dup_count
        summary.constraint_violations = constraint_count
        return summary

    @staticmethod
    def _calculate_error_metrics(result: Any) -> tuple[int, int]:
        duplicate_count = 0
        constraint_violations = 0
        for batch_result in result.batch_results:
            if not batch_result.success and batch_result.error_details:
                error_msg = (batch_result.error_message or "").lower()
                details = batch_result.error_details
                count = details.get("count", 0)

                if "duplicate_key" in error_msg or "unique constraint" in error_msg:
                    duplicate_count += details.get("duplicate_count", count)
                elif "constraint" in error_msg or "foreign key" in error_msg:
                    constraint_violations += details.get("constraint_violations", count)
        return duplicate_count, constraint_violations

    @staticmethod
    def _extract_inserted_records(result: Any) -> List[Any]:
        inserted_records = []
        for batch_result in result.batch_results:
            if batch_result.success and batch_result.error_details:
                records = batch_result.error_details.get("inserted_records", [])
                if records:
                    inserted_records.extend(records)
        return inserted_records

    async def _process_insert_batch(
            self,
            batch_items: List[Any],
            batch_index: int,
            model_class: Any,
            schema_to_select: Optional[Type[BaseModel]] = None,
            return_as_model: bool = False,
    ) -> BulkOperationResult:
        """Process a single batch of insert operations."""
        db = self._current_db_session
        if not db:
            raise RuntimeError("Database session not available in batch context")

        start_time = time.perf_counter()
        try:
            insert_data = self._prepare_insert_data(batch_items, model_class)

            if not insert_data:
                return self._create_result(batch_index, 0, True, start_time,
                                           details={"inserted_records": [], "inserted_count": 0})

            inserted_records = await self._execute_insert(
                db, model_class, insert_data, schema_to_select, return_as_model
            )

            return self._create_result(
                batch_index,
                len(batch_items),
                True,
                start_time,
                details={
                    "inserted_records": inserted_records,
                    "inserted_count": len(batch_items)
                }
            )

        except IntegrityError as e:
            return self._handle_integrity_error(e, batch_index, batch_items, start_time)
        except Exception as e:
            return self._handle_general_error(e, batch_index, batch_items, start_time)

    def _prepare_insert_data(self, batch_items: List[Any], model_class: Any) -> List[dict]:
        """Convert a list of items into a list of dictionaries for insertion."""
        insert_data = []
        for item in batch_items:
            data = self._extract_item_data(item, model_class)
            insert_data.append(data)
        return insert_data

    @staticmethod
    def _extract_item_data(item: Any, model_class: Any) -> dict:
        """Extract dictionary data from a single item."""
        if isinstance(item, BaseModel):
            # Pydantic model - use exclude_unset to allow DB defaults
            return item.model_dump(exclude_unset=True)

        if isinstance(item, dict):
            # Dictionary - use as is
            return item.copy()

        if hasattr(item, '_sa_instance_state'):
            # SQLAlchemy model instance
            data = {}
            for column in model_class.__table__.columns:
                if hasattr(item, column.name):
                    value = getattr(item, column.name)
                    # Only include if the value is set (not None), allowing DB defaults for None
                    if value is not None:
                        data[column.name] = value
            return data

        if hasattr(item, "__dict__"):
            # Regular Python object - exclude private/internal attributes
            return {k: v for k, v in item.__dict__.items() if not k.startswith("_")}

        return {}

    @staticmethod
    async def _execute_insert(
            db: AsyncSession,
            model_class: Any,
            insert_data: List[dict],
            schema_to_select: Optional[Type[BaseModel]],
            return_as_model: bool
    ) -> List[Any]:
        """Execute the SQL insert statement."""
        stmt = insert(model_class).values(insert_data)
        needs_return = schema_to_select is not None or return_as_model

        if needs_return:
            # Use RETURNING clause which is safer and cleaner
            stmt = stmt.returning(model_class)
            result = await db.execute(stmt)
            fetched_records = result.scalars().all()

            if schema_to_select:
                return [schema_to_select.model_validate(record.__dict__) for record in fetched_records]
            return list(fetched_records)
        else:
            await db.execute(stmt)
            return []

    def _handle_integrity_error(
            self,
            e: IntegrityError,
            batch_index: int,
            batch_items: List[Any],
            start_time: float
    ) -> BulkOperationResult:
        error_message = str(e)
        error_lower = error_message.lower()

        duplicate_count = 0
        constraint_violations = 0

        if "duplicate key" in error_lower or "unique constraint" in error_lower:
            duplicate_count = len(batch_items)
        elif "foreign key" in error_lower or "constraint" in error_lower:
            constraint_violations = len(batch_items)

        return self._create_result(
            batch_index,
            len(batch_items),
            False,
            start_time,
            error_message=error_message,
            details={
                "error_type": "integrity_error",
                "duplicate_count": duplicate_count,
                "constraint_violations": constraint_violations,
                "count": len(batch_items),
                "failed_items": [
                    {"error": error_message, "item": str(item)}
                    for item in batch_items
                ]
            }
        )

    def _handle_general_error(
            self,
            e: Exception,
            batch_index: int,
            batch_items: List[Any],
            start_time: float
    ) -> BulkOperationResult:
        return self._create_result(
            batch_index,
            len(batch_items),
            False,
            start_time,
            error_message=str(e),
            details={
                "error_type": "general_error",
                "count": len(batch_items),
                "failed_items": [
                    {"error": str(e), "item": str(item)}
                    for item in batch_items
                ]
            }
        )

    @staticmethod
    def _create_result(
            batch_index: int,
            processed: int,
            success: bool,
            start_time: float,
            error_message: str = "",
            details: Optional[dict] = None
    ) -> BulkOperationResult:
        return BulkOperationResult(
            batch_index=batch_index,
            items_processed=processed,
            success=success,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            error_message=error_message,
            error_details=details or {}
        )

    @staticmethod
    def _create_empty_summary(batch_size: int) -> BulkInsertSummary:
        """Create an empty summary for operations with no items."""
        now = datetime.now()
        return BulkInsertSummary(
            operation_type="insert",
            total_requested=0,
            total_processed=0,
            successful_count=0,
            failed_count=0,
            batch_count=0,
            start_time=now,
            end_time=now,
            duration_ms=0.0,
            success_rate=1.0,
            average_batch_duration_ms=0.0,
            items_per_second=0.0,
            failed_items=[],
            batch_results=[],
            batch_size=batch_size,
            commit_strategy="batch",
            allow_partial_success=True,
            duplicate_count=0,
            constraint_violations=0,
        )
