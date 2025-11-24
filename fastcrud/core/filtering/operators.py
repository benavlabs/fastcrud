"""
SQLAlchemy filter operator mappings and utilities.

This module defines the mapping between filter operator strings and their
corresponding SQLAlchemy column methods, along with validation logic.
"""

from typing import Callable, Any, Optional
from sqlalchemy import Column
from sqlalchemy.sql.elements import ColumnElement

from ...types import FilterValueType

FilterCallable = Callable[[Column[Any]], Callable[..., ColumnElement[bool]]]

COLLECTION_OPERATORS = {"in", "not_in", "between"}

SUPPORTED_FILTERS: dict[str, FilterCallable] = {
    "eq": lambda column: column.__eq__,
    "gt": lambda column: column.__gt__,
    "lt": lambda column: column.__lt__,
    "gte": lambda column: column.__ge__,
    "lte": lambda column: column.__le__,
    "ne": lambda column: column.__ne__,
    "is": lambda column: column.is_,
    "is_not": lambda column: column.is_not,
    "like": lambda column: column.like,
    "notlike": lambda column: column.notlike,
    "ilike": lambda column: column.ilike,
    "notilike": lambda column: column.notilike,
    "startswith": lambda column: column.startswith,
    "endswith": lambda column: column.endswith,
    "contains": lambda column: column.contains,
    "match": lambda column: column.match,
    "between": lambda column: column.between,
    "in": lambda column: column.in_,
    "not_in": lambda column: column.not_in,
    "or": lambda column: column.or_,
    "not": lambda column: column.not_,
}


def get_sqlalchemy_filter(
    operator: str, value: FilterValueType
) -> Optional[FilterCallable]:
    """
    Get SQLAlchemy filter function for operator with validation.

    This function validates certain operators that require specific value types
    and returns the appropriate SQLAlchemy filter function.

    Args:
        operator: Filter operator string (e.g., 'eq', 'gt', 'in', 'between')
        value: The value to be filtered (used for validation)

    Returns:
        FilterCallable function if operator is supported, None otherwise

    Raises:
        ValueError: If operator requires specific value type that doesn't match

    Example:
        >>> filter_func = get_sqlalchemy_filter('eq', 'test')
        >>> condition = filter_func(User.name)('test')  # User.name == 'test'

        >>> # This will raise ValueError
        >>> get_sqlalchemy_filter('in', 'invalid')  # Should be list/tuple/set
    """
    if operator in COLLECTION_OPERATORS:
        if not isinstance(value, (tuple, list, set)):
            raise ValueError(f"<{operator}> filter must be tuple, list or set")

    if (
        operator == "between"
        and isinstance(value, (tuple, list, set))
        and len(value) != 2
    ):
        raise ValueError("Between operator requires exactly 2 values")

    return SUPPORTED_FILTERS.get(operator)


def get_operator_wrap_type(operator: str) -> Optional[type]:
    if operator in COLLECTION_OPERATORS:
        return list

    return None
