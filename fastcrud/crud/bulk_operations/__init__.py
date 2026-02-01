"""
Bulk operations module for FastCRUD.

This module provides efficient batch processing capabilities for database operations,
including bulk insert, update, and delete operations with comprehensive error handling
and performance optimization.

Features:
- Configurable batch sizes for memory management
- Database-specific optimizations
- Comprehensive error reporting and partial success handling
- Transaction management and rollback capabilities
- Performance monitoring and metrics collection
"""

from .batch_processor import BatchConfig, BatchProcessor
from .delete_multi import BulkDeleteManager
from .insert_multi import BulkInsertManager
from .summary_models import (
    BulkDeleteSummary,
    BulkInsertSummary,
    BulkOperationResult,
    BulkOperationSummary,
    BulkUpdateSummary,
)
from .update_multi import BulkUpdateManager

__all__ = [
    "BulkOperationSummary",
    "BulkInsertSummary",
    "BulkUpdateSummary",
    "BulkDeleteSummary",
    "BulkOperationResult",
    "BatchConfig",
    "BatchProcessor",
    "BulkInsertManager",
    "BulkUpdateManager",
    "BulkDeleteManager",
]
