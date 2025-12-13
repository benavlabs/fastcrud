"""
Filter processing engine for FastCRUD.

This module provides comprehensive filter processing capabilities for converting
Python filter arguments into SQLAlchemy WHERE clauses. It supports a wide range
of operators and complex filtering scenarios including OR, NOT, and joined model filters.
"""

from .processor import FilterProcessor
from .operators import SUPPORTED_FILTERS, get_sqlalchemy_filter, FilterCallable
from .validators import validate_joined_filter_format, validate_filter_operator

__all__ = [
    "FilterProcessor",
    "FilterCallable",
    "SUPPORTED_FILTERS",
    "get_sqlalchemy_filter",
    "validate_joined_filter_format",
    "validate_filter_operator",
]
