"""
SQLAlchemy filter operator mappings and utilities.

This module defines the mapping between filter operator strings and their
corresponding SQLAlchemy column methods, along with validation logic.
"""

from typing import Callable, Any
from sqlalchemy import Column
from sqlalchemy.sql.elements import ColumnElement

from ...types import FilterValueType

FilterCallable = Callable[[Column[Any]], Callable[..., ColumnElement[bool]]]

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
    operator: str,
    value: FilterValueType,
    custom_filters: dict[str, FilterCallable] | None = None,
) -> FilterCallable | None:
    """
    Get SQLAlchemy filter function for operator with validation.

    This function validates certain operators that require specific value types
    and returns the appropriate SQLAlchemy filter function. Custom filters
    take precedence over built-in filters.

    Args:
        operator: Filter operator string (e.g., 'eq', 'gt', 'in', 'between')
        value: The value to be filtered (used for validation)
        custom_filters: Optional dictionary of custom filter operators

    Returns:
        FilterCallable function if operator is supported, None otherwise

    Raises:
        ValueError: If operator requires specific value type that doesn't match

    Example:
        >>> filter_func = get_sqlalchemy_filter('eq', 'test')
        >>> condition = filter_func(User.name)('test')  # User.name == 'test'

        >>> # This will raise ValueError
        >>> get_sqlalchemy_filter('in', 'invalid')  # Should be list/tuple/set

        >>> # With custom filter
        >>> custom = {"year": lambda col: lambda val: func.extract('year', col) == val}
        >>> filter_func = get_sqlalchemy_filter('year', 2024, custom_filters=custom)
    """
    if operator in {"in", "not_in", "between"}:
        if not isinstance(value, (tuple, list, set)):
            raise ValueError(f"<{operator}> filter must be tuple, list or set")

    if (
        operator == "between"
        and isinstance(value, (tuple, list, set))
        and len(value) != 2
    ):
        raise ValueError("Between operator requires exactly 2 values")

    if custom_filters and operator in custom_filters:
        return custom_filters[operator]

    return SUPPORTED_FILTERS.get(operator)
