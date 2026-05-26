"""
Filter processing engine for FastCRUD.

This module provides comprehensive filter processing capabilities for converting
Python filter arguments into SQLAlchemy WHERE clauses. It supports a wide range
of operators and complex filtering scenarios including OR, NOT, and joined model filters.
"""

from .filter_model import Filter
from .processor import FilterProcessor
from .operators import (
    COLLECTION_OPERATORS,
    SUPPORTED_FILTERS,
    FilterCallable,
    get_operator_wrap_type,
    get_sqlalchemy_filter,
)
from .validators import validate_joined_filter_format, validate_filter_operator

__all__ = [
    "COLLECTION_OPERATORS",
    "Filter",
    "FilterProcessor",
    "FilterCallable",
    "SUPPORTED_FILTERS",
    "get_operator_wrap_type",
    "get_sqlalchemy_filter",
    "validate_joined_filter_format",
    "validate_filter_operator",
]
