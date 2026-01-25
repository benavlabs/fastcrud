"""
Batch processor for bulk operations.

This module provides the core infrastructure for processing large datasets
in batches, with comprehensive error handling, transaction management,
and performance monitoring.

Classes:
    BatchProcessor: Core batch processing engine
    BatchConfig: Configuration for batch processing
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from .summary_models import BulkOperationResult, BulkOperationSummary

T = TypeVar("T")
U = TypeVar("U")


class BatchConfig:
    """
    Configuration for batch processing operations.
    
    Attributes:
        batch_size: Number of items to process per batch
        max_workers: Maximum number of concurrent workers (for async operations)
        enable_transactions: Whether to use transactions per batch
        commit_strategy: When to commit ("batch", "all", "never")
        allow_partial_success: Whether to continue on errors
        timeout_seconds: Maximum time per batch in seconds
        retry_attempts: Number of retry attempts for failed batches
        retry_delay: Delay between retries in seconds
    """

    def __init__(
            self,
            batch_size: int = 1000,
            max_workers: int = 4,
            enable_transactions: bool = True,
            commit_strategy: str = "batch",
            allow_partial_success: bool = True,
            timeout_seconds: float | None = None,
            retry_attempts: int = 0,
            retry_delay: float = 0.1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if commit_strategy not in ["batch", "all", "never"]:
            raise ValueError("commit_strategy must be 'batch', 'all', or 'never'")
        if retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")

        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_transactions = enable_transactions
        self.commit_strategy = commit_strategy
        self.allow_partial_success = allow_partial_success
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay


class BatchProcessor:
    """
    Core batch processing engine for bulk operations.
    This class provides the infrastructure for processing large datasets
    in batches with proper error handling, transaction management,
    and performance monitoring.
    """

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()
        self._operation_start_time: datetime | None = None

    async def process_batches(
            self,
            items: list[T],
            processor_func: Callable[[list[T], int], Coroutine[Any, Any, BulkOperationResult]],
            db: AsyncSession | None = None,
            operation_name: str = "batch_operation",
    ) -> BulkOperationSummary:
        """
        Process items in batches using the provided processor function.
        Args:
            items: List of items to process
            processor_func: Async function that processes a batch and returns a result
            db: Optional database session for transaction management
            operation_name: Name of the operation for reporting
        Returns:
            BulkOperationSummary with detailed results and metrics
        """
        if not items:
            return self._create_empty_summary(operation_name)

        self._operation_start_time = datetime.now()
        batches = self.create_chunks(items, self.config.batch_size)
        use_single_commit = self.config.commit_strategy == "all" and db is not None
        started_transaction = False

        try:
            if use_single_commit and db is not None and not db.in_transaction():
                await db.begin()
                started_transaction = True

            batch_results = await self._process_all_batches(batches, processor_func, db)

            if use_single_commit and db is not None and db.in_transaction():
                await db.commit()
            else:
                await self._handle_final_commit(db)
        except Exception:
            if use_single_commit and db is not None and db.in_transaction():
                await db.rollback()
            raise
        finally:
            if started_transaction and db is not None and db.in_transaction():
                await db.rollback()

        return self._create_operation_summary(
            operation_name,
            len(items),
            batch_results
        )

    async def _process_all_batches(
            self,
            batches: list[list[T]],
            processor_func: Callable[[list[T], int], Coroutine[Any, Any, BulkOperationResult]],
            db: AsyncSession | None,
    ) -> list[BulkOperationResult]:
        """Process all batches sequentially and return results."""
        results = []
        for i, batch in enumerate(batches):
            result = await self._process_single_batch(batch, i, processor_func, db)
            results.append(result)
        return results

    async def _handle_final_commit(self, db: AsyncSession | None) -> None:
        """Handle final commit if using 'all' strategy."""
        if self.config.commit_strategy == "all" and db and db.in_transaction():
            await db.commit()

    def _create_operation_summary(
            self,
            operation_name: str,
            total_items: int,
            batch_results: list[BulkOperationResult]
    ) -> BulkOperationSummary:
        """Create the final operation summary from batch results."""
        successful_count, failed_count, failed_items = self._aggregate_batch_results(batch_results)
        end_time = datetime.now()
        assert self._operation_start_time is not None, "Operation start time not set"
        duration_ms = (end_time - self._operation_start_time).total_seconds() * 1000
        batch_count = len(batch_results)

        # Calculate metrics
        success_rate = (
            (successful_count / max(total_items, successful_count + failed_count))
            if (successful_count + failed_count) > 0 else 1.0
        )
        avg_batch_duration = duration_ms / batch_count if batch_count > 0 else 0
        items_per_sec = (successful_count / (duration_ms / 1000)) if duration_ms > 0 else 0

        return BulkOperationSummary(
            operation_type=operation_name,
            total_requested=total_items,
            total_processed=successful_count + failed_count,
            successful_count=successful_count,
            failed_count=failed_count,
            batch_count=batch_count,
            start_time=self._operation_start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success_rate=success_rate,
            average_batch_duration_ms=avg_batch_duration,
            items_per_second=items_per_sec,
            failed_items=failed_items,
            batch_results=batch_results,
            batch_size=self.config.batch_size,
            commit_strategy=self.config.commit_strategy,
            allow_partial_success=self.config.allow_partial_success,
        )

    @staticmethod
    def _aggregate_batch_results(batch_results: list[BulkOperationResult]) -> tuple[
        int, int, list[dict[str, Any]]]:
        """Aggregate results from all batches."""
        successful_count = 0
        failed_count = 0
        failed_items: list[dict[str, Any]] = []
        for result in batch_results:
            if result.success:
                successful_count += result.items_processed
            else:
                failed_count += result.items_processed
                if result.error_details and "failed_items" in result.error_details:
                    failed_items.extend(result.error_details["failed_items"])
        return successful_count, failed_count, failed_items

    async def _process_single_batch(
            self,
            batch_items: list[T],
            batch_index: int,
            processor_func: Callable[[list[T], int], Coroutine[Any, Any, BulkOperationResult]],
            db: AsyncSession | None,
    ) -> BulkOperationResult:
        """
        Process a single batch with error handling and retry logic.
        """
        for attempt in range(self.config.retry_attempts + 1):
            try:
                return await self._execute_batch_step(batch_items, batch_index, processor_func, db)
            except Exception as e:
                if attempt == self.config.retry_attempts:
                    if not self.config.allow_partial_success:
                        raise e
                    return BulkOperationResult(
                        batch_index=batch_index,
                        items_processed=len(batch_items),
                        success=False,
                        error_message=str(e),
                        error_details={"failed_items": batch_items},
                        duration_ms=0,
                    )
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        # This part should be unreachable
        raise RuntimeError("Batch processing failed after all retries.")

    async def _execute_batch_step(
            self,
            batch_items: list[T],
            batch_index: int,
            processor_func: Callable[[list[T], int], Coroutine[Any, Any, BulkOperationResult]],
            db: AsyncSession | None,
    ) -> BulkOperationResult:
        """
        Execute the batch processing step, managing transactions if configured.
        """
        should_use_transaction = (
                self.config.enable_transactions
                and db is not None
                and self.config.commit_strategy in ("batch", "all")
        )

        if should_use_transaction and db is not None:
            # For batch or all strategies, use an existing transaction or start a new one
            if db.in_transaction():
                return await processor_func(batch_items, batch_index)

            async with db.begin():
                return await processor_func(batch_items, batch_index)

        # For "never" strategy or no transactions
        return await processor_func(batch_items, batch_index)

    def _create_empty_summary(self, operation_name: str) -> BulkOperationSummary:
        """Create an empty summary for operations with no items."""
        now = datetime.now()
        return BulkOperationSummary(
            operation_type=operation_name,
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
            batch_size=self.config.batch_size,
            commit_strategy=self.config.commit_strategy,
            allow_partial_success=self.config.allow_partial_success,
        )

    @staticmethod
    def create_chunks(items: list[T], chunk_size: int) -> list[list[T]]:
        """
        Create chunks from a list of items.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
