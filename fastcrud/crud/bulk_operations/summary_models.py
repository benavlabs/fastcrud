"""
Summary models for bulk operations.

This module defines Pydantic models for representing the results and summaries
of bulk database operations, providing detailed feedback on operation success,
failures, and performance metrics.

Classes:
    BulkOperationResult: Base class for individual operation results
    BulkOperationSummary: Comprehensive summary of bulk operation results
    BulkInsertSummary: Specific summary for bulk insert operations
    BulkUpdateSummary: Specific summary for bulk update operations  
    BulkDeleteSummary: Specific summary for bulk delete operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer


class BulkOperationResult(BaseModel):
    """
    Represents the result of a single bulk operation batch.
    
    This model captures the outcome of processing a single batch within
    a larger bulk operation, including success/failure status and any
    error details.
    
    Attributes:
        batch_index: Zero-based index of the batch (0 for first batch)
        items_processed: Number of items processed in this batch
        success: Whether the batch completed successfully
        error_message: Error message if the batch failed
        error_details: Additional error details and context
        duration_ms: Time taken to process this batch in milliseconds
    """

    batch_index: int = Field(..., description="Zero-based index of the batch")
    items_processed: int = Field(..., ge=0, description="Number of items processed in this batch")
    success: bool = Field(..., description="Whether the batch completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if the batch failed")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details and context")
    duration_ms: float = Field(..., ge=0, description="Time taken to process this batch in milliseconds")


class BulkOperationSummary(BaseModel):
    """
    Comprehensive summary of a bulk operation result.
    
    This model provides detailed feedback on bulk operations, including
    success rates, error information, and performance metrics. It's designed
    to give users complete visibility into bulk operation outcomes.
    
    Attributes:
        operation_type: Type of operation performed ("insert", "update", "delete")
        total_requested: Total number of items requested for processing
        total_processed: Total number of items actually processed
        successful_count: Number of items successfully processed
        failed_count: Number of items that failed processing
        batch_count: Number of batches processed
        success_rate: Percentage of successful operations (0.0 to 1.0)
        start_time: When the bulk operation started
        end_time: When the bulk operation completed
        duration_ms: Total duration in milliseconds
        average_batch_duration_ms: Average time per batch in milliseconds
        items_per_second: Processing rate in items per second
        failed_items: List of failed items with error details
        batch_results: Detailed results for each batch
    """

    operation_type: str = Field(..., description="Type of operation performed (insert, update, delete)")
    total_requested: int = Field(..., ge=0, description="Total number of items requested for processing")
    total_processed: int = Field(..., ge=0, description="Total number of items actually processed")
    successful_count: int = Field(..., ge=0, description="Number of items successfully processed")
    failed_count: int = Field(..., ge=0, description="Number of items that failed processing")
    batch_count: int = Field(..., ge=0, description="Number of batches processed")

    # Timing information
    start_time: datetime = Field(..., description="When the bulk operation started")
    end_time: datetime = Field(..., description="When the bulk operation completed")
    duration_ms: float = Field(..., ge=0, description="Total duration in milliseconds")

    # Computed properties
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of successful operations")
    average_batch_duration_ms: float = Field(..., ge=0.0, description="Average time per batch in milliseconds")
    items_per_second: float = Field(..., ge=0.0, description="Processing rate in items per second")

    # Error and detailed information
    failed_items: List[Dict[str, Any]] = Field(default_factory=list,
                                               description="List of failed items with error details")
    batch_results: List[BulkOperationResult] = Field(default_factory=list,
                                                     description="Detailed results for each batch")

    # Configuration used
    batch_size: Optional[int] = Field(None, description="Batch size used for processing")
    commit_strategy: Optional[str] = Field(None, description="Commit strategy used (e.g., 'batch', 'all')")
    allow_partial_success: bool = Field(False, description="Whether partial success was allowed")

    @property
    def is_complete_success(self) -> bool:
        """Check if all operations completed successfully."""
        return self.failed_count == 0

    @property
    def is_complete_failure(self) -> bool:
        """Check if all operations failed."""
        return self.successful_count == 0

    @property
    def has_failures(self) -> bool:
        """Check if there were any failures."""
        return self.failed_count > 0

    @field_serializer('start_time', 'end_time')
    def serialize_dt(self, dt: datetime, _info) -> str:
        return dt.isoformat()


class BulkInsertSummary(BulkOperationSummary):
    """
    Summary specific to bulk insert operations.
    
    Extends the base summary with insert-specific information and computed
    properties relevant to insert operations.
    """

    # Override operation_type with default
    operation_type: str = Field(default="insert", description="Type of operation performed")

    # Insert-specific metrics
    duplicate_count: int = Field(default=0, ge=0, description="Number of duplicate records detected")
    constraint_violations: int = Field(default=0, ge=0, description="Number of constraint violations")


class BulkUpdateSummary(BulkOperationSummary):
    """
    Summary specific to bulk update operations.
    
    Extends the base summary with update-specific information and computed
    properties relevant to update operations.
    """

    # Override operation_type with default
    operation_type: str = Field(default="update", description="Type of operation performed")

    # Update-specific metrics
    not_found_count: int = Field(default=0, ge=0, description="Number of records not found for updating")
    unchanged_count: int = Field(default=0, ge=0, description="Number of records that were already up to date")


class BulkDeleteSummary(BulkOperationSummary):
    """
    Summary specific to bulk delete operations.
    
    Extends the base summary with delete-specific information and computed
    properties relevant to delete operations.
    """

    # Override operation_type with default
    operation_type: str = Field(default="delete", description="Type of operation performed")

    # Delete-specific metrics
    not_found_count: int = Field(default=0, ge=0, description="Number of records not found for deletion")
    soft_deleted_count: int = Field(default=0, ge=0, description="Number of records soft deleted")
    hard_deleted_count: int = Field(default=0, ge=0, description="Number of records hard deleted")
