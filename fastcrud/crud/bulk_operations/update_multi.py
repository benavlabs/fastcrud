"""
Bulk update operations for FastCRUD.

This module provides efficient bulk update capabilities with batching,
error handling, and comprehensive reporting.
"""
import time
from datetime import datetime
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel
from sqlalchemy import and_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .batch_processor import BatchConfig, BatchProcessor
from .summary_models import BulkOperationResult, BulkOperationSummary, BulkUpdateSummary


class BulkUpdateManager:
    """
    Manager for bulk update operations.

    This class handles bulk updating data with proper batching,
    error handling, transaction management, and detailed reporting.
    """

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()
        self.batch_processor = BatchProcessor(self.config)

    async def update_multi(
            self,
            db: AsyncSession,
            model_class: Any,
            objects: list[dict | Any],
            *,
            batch_size: int = 1000,
            commit: bool = True,
            allow_partial_success: bool = True,
            return_summary: bool = False,
            schema_to_select: type | None = None,
            return_as_model: bool = False,
    ) -> BulkUpdateSummary | list[dict | Any]:
        """
        Update multiple objects efficiently with batch processing.

        Args:
            db: Database session
            model_class: SQLAlchemy model class
            objects: List of objects to update (dicts or Pydantic models)
            batch_size: Number of objects to update per batch
            commit: Whether to commit the transaction
            allow_partial_success: Whether to continue on errors
            return_summary: Whether to return detailed summary
            schema_to_select: Pydantic schema for return format
            return_as_model: Whether to return as model instances

        Returns:
            Either BulkUpdateSummary (if return_summary=True) or a list of updated records
        """
        self._validate_update_request(objects, model_class)

        # Create a processor with the specific config
        config = self._create_execution_config(batch_size, commit, allow_partial_success)
        processor = BatchProcessor(config)
        primary_key_columns = self._get_model_primary_keys(model_class)

        async def _batch_handler(batch_items: list[Any], batch_idx: int):
            return await self._process_update_batch(
                db=db,
                batch_items=batch_items,
                batch_index=batch_idx,
                model_class=model_class,
                primary_key_columns=primary_key_columns,
                schema_to_select=schema_to_select,
                return_as_model=return_as_model
            )

        summary = await processor.process_batches(
            items=objects,
            processor_func=_batch_handler,
            db=db,
            operation_name="bulk_update"
        )

        return self._handle_update_results(summary, return_summary)

    @staticmethod
    async def _process_update_batch(
            db: AsyncSession,
            batch_items: list[Any],
            batch_index: int,
            model_class: Any,
            primary_key_columns: list[str],
            schema_to_select: type[BaseModel] | None = None,
            return_as_model: bool = False,
    ) -> BulkOperationResult:
        """
        Process a single batch of update operations.

        Args:
            db: Database session
            batch_items: Items to update in this batch
            batch_index: Index of this batch
            model_class: SQLAlchemy model class
            primary_key_columns: List of primary key column names
            schema_to_select: Schema for return format
            return_as_model: Whether to return as model instances

        Returns:
            BulkOperationResult for this batch
        """
        start_time = time.perf_counter()

        try:
            updated_records: list[Any] = []
            rows_updated = 0

            for item in batch_items:
                item_data = item.model_dump() if hasattr(item, "model_dump") else item

                # Ensure PK values exist
                if not all(pk in item_data for pk in primary_key_columns):
                    continue

                pk_values = {pk: item_data.get(pk) for pk in primary_key_columns}
                update_data = {
                    k: v for k, v in item_data.items() if k not in primary_key_columns
                }

                stmt = (
                    update(model_class)
                    .where(
                        and_(
                            *(
                                getattr(model_class, pk) == pk_values[pk]
                                for pk in primary_key_columns
                            )
                        )
                    )
                    .values(**update_data)
                )

                wants_rows = bool(schema_to_select or return_as_model)
                if wants_rows:
                    stmt = stmt.returning(model_class)

                result = await db.execute(stmt)

                if wants_rows:
                    returned = result.scalars().all()
                    rows_updated += len(returned)
                    updated_records.extend(returned)
                else:
                    rc = getattr(result, "rowcount", 0) or 0
                    if rc > 0:
                        rows_updated += rc

            if schema_to_select:
                updated_records = [
                    schema_to_select.model_validate(rec, from_attributes=True)
                    for rec in updated_records
                ]

            duration_ms = (time.perf_counter() - start_time) * 1000
            success = rows_updated > 0

            return BulkOperationResult(
                batch_index=batch_index,
                items_processed=len(batch_items),
                success=success,
                duration_ms=duration_ms,
                error_message=None if success else "No records found to update",
                error_details={
                    "updated_records": updated_records,
                    "updated_count": rows_updated,
                    "not_found_count": len(batch_items) - rows_updated,
                },
            )

        except IntegrityError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return BulkOperationResult(
                batch_index=batch_index,
                items_processed=len(batch_items),
                success=False,
                error_message=str(e),
                error_details={
                    "error_type": "integrity_error",
                    "failed_items": [
                        {"error": str(e), "item": batch_items[i]}
                        for i in range(len(batch_items))
                    ],
                },
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return BulkOperationResult(
                batch_index=batch_index,
                items_processed=len(batch_items),
                success=False,
                error_message=str(e),
                error_details={
                    "error_type": "general_error",
                    "failed_items": [
                        {"error": str(e), "item": batch_items[i]}
                        for i in range(len(batch_items))
                    ],
                },
                duration_ms=duration_ms,
            )

    @staticmethod
    def _create_empty_summary(batch_size: int) -> BulkUpdateSummary:
        """Create an empty summary for operations with no items."""
        now = datetime.now()

        return BulkUpdateSummary(
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
            not_found_count=0,
            unchanged_count=0,
        )

    def _validate_update_request(self, objects: list[Any], model_class: Any) -> None:
        """Validates the input objects for the update operation."""
        if not objects:
            raise ValueError("No objects provided for update operation")

        primary_key_columns = self._get_model_primary_keys(model_class)
        if not primary_key_columns:
            raise ValueError("Model class must have a primary key for update operations")

    @staticmethod
    def _get_model_primary_keys(model_class: Any) -> list[str]:
        """Extracts primary key column names from the SQLAlchemy model."""
        return [key.name for key in model_class.__table__.primary_key.columns]

    @staticmethod
    def _create_execution_config(
            batch_size: int,
            commit: bool,
            allow_partial_success: bool
    ) -> BatchConfig:
        """Creates a BatchConfig instance based on update parameters."""
        return BatchConfig(
            batch_size=batch_size,
            enable_transactions=commit,
            commit_strategy="batch" if commit else "never",
            allow_partial_success=allow_partial_success
        )

    @staticmethod
    def _handle_update_results(summary: BulkOperationSummary, return_summary: bool) -> BulkUpdateSummary | list[dict | Any]:
        # Convert to update-specific summary
        update_summary = BulkUpdateSummary(**summary.model_dump())

        # Update-specific metrics
        not_found_count = 0
        unchanged_count = 0

        for batch_result in summary.batch_results:
            if not batch_result.success and batch_result.error_details:
                if batch_result.error_message and "not_found" in batch_result.error_message.lower():
                    not_found_count += batch_result.error_details.get("count", 0)
            elif batch_result.success and batch_result.error_details:
                if "unchanged" in batch_result.error_details:
                    unchanged_count += batch_result.error_details.get("unchanged", 0)

        update_summary.not_found_count = not_found_count
        update_summary.unchanged_count = unchanged_count

        if return_summary:
            return update_summary

        # Return a list of updated records
        updated_records: list[Any] = []
        for batch_result in summary.batch_results:
            if batch_result.success and batch_result.error_details and "updated_records" in batch_result.error_details:
                updated_records.extend(batch_result.error_details["updated_records"])

        return updated_records
