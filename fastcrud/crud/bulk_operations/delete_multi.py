"""
Bulk delete operations for FastCRUD.

This module provides efficient bulk delete capabilities with batching,
error handling, soft delete support, and comprehensive reporting.
"""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import and_, delete, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from fastcrud.core.filtering.processor import FilterProcessor
from .batch_processor import BatchConfig, BatchProcessor
from .summary_models import BulkDeleteSummary, BulkOperationResult


class BulkDeleteManager:
    """
    Manager for bulk delete operations.
    
    This class handles bulk deleting data with proper batching,
    error handling, transaction management, soft delete support,
    and detailed reporting.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.batch_processor = BatchProcessor(self.config)

    async def delete_multi(
            self,
            db: AsyncSession,
            model_class: Any,
            *,
            allow_multiple: bool = True,
            batch_size: int = 1000,
            commit: bool = True,
            allow_partial_success: bool = True,
            return_summary: bool = False,
            is_deleted_column: Optional[str] = None,
            soft_delete: Optional[bool] = None,
            **filters: Any,
    ) -> Union[BulkDeleteSummary, int]:
        """
        Delete multiple objects efficiently with batch processing.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
            allow_multiple: Whether to allow deletion of multiple records
            batch_size: Number of records to process per batch
            commit: Whether to commit the transaction
            allow_partial_success: Whether to continue on errors
            return_summary: Whether to return detailed summary
            is_deleted_column: Name of soft delete column (if any)
            soft_delete: Whether to use soft delete (auto-detected from a model)
            **filters: Filter conditions for deletion
            
        Returns:
            Either BulkDeleteSummary (if return_summary=True) or count of deleted records
        """
        if not filters:
            if return_summary:
                return self._create_empty_summary(batch_size=batch_size)
            return 0

        # Auto-detect soft delete if not specified
        if soft_delete is None:
            # Only auto-enable soft delete if is_deleted_column is explicitly provided
            # or if the method is called with soft_delete=True parameter
            soft_delete = is_deleted_column is not None

        if soft_delete and not is_deleted_column:
            # Check for common soft delete column names
            if hasattr(model_class, 'is_deleted'):
                is_deleted_column = 'is_deleted'
            elif hasattr(model_class, 'deleted_at'):
                is_deleted_column = 'deleted_at'

        # Create config with the provided batch_size
        config = BatchConfig(
            batch_size=batch_size,
            enable_transactions=True,
            commit_strategy="batch" if commit else "never",
            allow_partial_success=allow_partial_success,
        )

        # Create a processor with the specific config
        processor = BatchProcessor(config)

        async def process_delete_batch(batch_filters: List[Dict[str, Any]], batch_index: int) -> BulkOperationResult:
            return await self._process_delete_batch(
                db, batch_filters, batch_index, model_class,
                is_deleted_column, soft_delete, allow_multiple
            )
            
        result = await processor.process_batches(
            [filters],  # Single batch with all filters
            process_delete_batch,
            db=db,
            operation_name="delete"
        )

        # Convert to delete-specific summary
        delete_summary = BulkDeleteSummary(**result.model_dump())

        # Delete-specific metrics
        not_found_count = 0
        soft_deleted_count = 0
        hard_deleted_count = 0

        for batch_result in result.batch_results:
            if batch_result.success and batch_result.error_details:
                soft_deleted_count += batch_result.error_details.get("soft_deleted_count", 0)
                hard_deleted_count += batch_result.error_details.get("hard_deleted_count", 0)
            elif not batch_result.success:
                error_msg = batch_result.error_message or ""
                error_details = batch_result.error_details or {}
                if "not_found" in error_msg.lower():
                    not_found_count += error_details.get("count", 0)

        delete_summary.not_found_count = not_found_count
        delete_summary.soft_deleted_count = soft_deleted_count
        delete_summary.hard_deleted_count = hard_deleted_count
        delete_summary.successful_count = soft_deleted_count + hard_deleted_count

        if return_summary:
            return delete_summary

        # Return count of deleted records
        return delete_summary.successful_count

    @staticmethod
    async def _process_delete_batch(
            db: AsyncSession,
            batch_filters: List[Dict[str, Any]],
            batch_index: int,
            model_class: Any,
            is_deleted_column: Optional[str] = None,
            soft_delete: bool = True,
            allow_multiple: bool = True,
    ) -> BulkOperationResult:
        """
        Process a single batch of delete operations.
        
        Args:
            db: Database session
            batch_filters: Filter conditions for this batch
            batch_index: Index of this batch
            model_class: SQLAlchemy model class
            is_deleted_column: Name of soft delete column
            soft_delete: Whether to use soft delete
            allow_multiple: Whether to allow deletion of multiple records
            
        Returns:
            BulkOperationResult for this batch
        """
        start_time = time.perf_counter()

        try:
            if not batch_filters:
                raise ValueError("No filter conditions provided")

            # Use the first (and typically only) filter set
            filters = batch_filters[0]

            filter_processor = FilterProcessor(model_class)
            filter_conditions = filter_processor.parse_filters(**filters)

            # Build the WHERE clause
            where_clause = and_(*filter_conditions)

            # Perform the deletion
            if soft_delete:
                if not is_deleted_column:
                    raise ValueError("is_deleted_column must be specified for soft delete")
                
                update_values: Dict[str, Any] = {is_deleted_column: True}
                if is_deleted_column.endswith('_at'):
                    update_values[is_deleted_column] = datetime.now()

                stmt = update(model_class).where(where_clause).values(**update_values)
                result = await db.execute(stmt)
                deleted_count = result.rowcount

                duration_ms = (time.perf_counter() - start_time) * 1000
                return BulkOperationResult(
                    batch_index=batch_index,
                    items_processed=deleted_count,
                    success=True,
                    duration_ms=duration_ms,
                    error_details={
                        "soft_deleted_count": deleted_count,
                        "hard_deleted_count": 0,
                    },
                    error_message=None
                )
            else:
                # Hard delete
                delete_stmt = delete(model_class).where(where_clause)
                result = await db.execute(delete_stmt)
                deleted_count = result.rowcount

                duration_ms = (time.perf_counter() - start_time) * 1000
                return BulkOperationResult(
                    batch_index=batch_index,
                    items_processed=deleted_count,
                    success=True,
                    duration_ms=duration_ms,
                    error_details={
                        "soft_deleted_count": 0,
                        "hard_deleted_count": deleted_count,
                    },
                    error_message=None
                )

        except IntegrityError as e:
            # Handle integrity errors (foreign key constraints, etc.)
            duration_ms = (time.perf_counter() - start_time) * 1000

            return BulkOperationResult(
                batch_index=batch_index,
                items_processed=0,
                success=False,
                error_message=str(e),
                error_details={
                    "error_type": "integrity_error",
                    "reason": "Referential integrity constraint violated"
                },
                duration_ms=duration_ms,
            )

        except Exception as e:
            # Handle other errors
            duration_ms = (time.perf_counter() - start_time) * 1000

            return BulkOperationResult(
                batch_index=batch_index,
                items_processed=0,
                success=False,
                error_message=str(e),
                error_details={
                    "error_type": "general_error"
                },
                duration_ms=duration_ms,
            )

    def _create_empty_summary(self, batch_size: int) -> BulkDeleteSummary:
        """Create an empty summary for operations with no items."""
        now = datetime.now()

        return BulkDeleteSummary(
            operation_type="delete",
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
            soft_deleted_count=0,
            hard_deleted_count=0,
        )