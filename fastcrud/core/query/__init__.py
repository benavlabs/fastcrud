"""
SQL Query Building Module - Centralized query construction utilities.

This module provides utilities for building and modifying SQLAlchemy SELECT
statements with support for filtering, sorting, pagination, and joins.
"""

from .builder import SQLQueryBuilder, build_joined_query, execute_joined_query
from .sorting import SortProcessor
from .joins import JoinBuilder
from .one_to_many import (
    split_join_configs,
    fetch_and_merge_one_to_many,
    fetch_one_to_many_with_limit,
)

__all__ = [
    "SQLQueryBuilder",
    "SortProcessor",
    "JoinBuilder",
    "build_joined_query",
    "execute_joined_query",
    "split_join_configs",
    "fetch_and_merge_one_to_many",
    "fetch_one_to_many_with_limit",
]
