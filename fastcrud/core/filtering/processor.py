"""
Main filter processing engine.

This module contains the FilterProcessor class which converts filter arguments
into SQLAlchemy WHERE clauses. It supports complex filtering scenarios including
OR conditions, NOT conditions, and joined model filters.
"""

from typing import Any, Optional, Union
from sqlalchemy import Column, or_, not_, and_
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql.elements import ColumnElement

from ..introspection import get_model_column
from ...types import ModelType, FilterValueType
from .operators import get_sqlalchemy_filter, FilterCallable
from .validators import validate_joined_filter_format


class FilterProcessor:
    """
    Processes filter arguments into SQLAlchemy filter conditions.

    This class provides the core filtering functionality for FastCRUD, converting
    Python filter arguments into SQLAlchemy WHERE clauses. It supports a wide
    range of filtering scenarios including simple equality, comparison operators,
    OR/NOT conditions, and joined model filters.

    The processor is designed to be stateless (except for the model reference
    and optional custom filters) and can be reused across multiple filtering operations.

    Attributes:
        model: The base SQLAlchemy model for filtering operations
        custom_filters: Optional dictionary of custom filter operators

    Example:
        >>> processor = FilterProcessor(User)
        >>> filters = processor.parse_filters(name='John', age__gt=25)
        >>> # Returns [User.name == 'John', User.age > 25]

        >>> # With custom filters
        >>> from sqlalchemy import func
        >>> custom = {"year": lambda col: lambda val: func.extract('year', col) == val}
        >>> processor = FilterProcessor(User, custom_filters=custom)
        >>> filters = processor.parse_filters(created_at__year=2024)
    """

    def __init__(
        self,
        model: type[ModelType],
        custom_filters: Optional[dict[str, FilterCallable]] = None,
    ):
        """
        Initialize the filter processor.

        Args:
            model: The SQLAlchemy model to use as the base for filtering
            custom_filters: Optional dictionary of custom filter operators.
                Keys are operator names (e.g., 'year'), values are callables
                that take a column and return a filter function.
        """
        self.model = model
        self.custom_filters = custom_filters

    def parse_filters(
        self, model: Optional[Union[type[ModelType], AliasedClass]] = None, **kwargs
    ) -> list[ColumnElement]:
        """
        Parse and convert filter arguments into SQLAlchemy filter conditions.

        This is the main entry point for filter processing. It handles all types
        of filters including simple equality, operator-based filters, OR/NOT conditions,
        and multi-field OR filters.

        Supported filter formats:
        - Simple filters: field=value → field == value
        - Operator filters: field__gt=value → field > value
        - OR filters: field__or={'gt': 5, 'lt': 10} → field > 5 OR field < 10
        - NOT filters: field__not={'gt': 5} → NOT (field > 5)
        - Joined filters: related.field__gt=value → related.field > value
        - Multi-field OR: _or={'field1': value1, 'field2': value2} → field1 == value1 OR field2 == value2

        Args:
            model: The model to apply filters to. Defaults to self.model
            **kwargs: Filter arguments in various supported formats

        Returns:
            List of SQLAlchemy ColumnElement objects representing WHERE conditions

        Example:
            >>> filters = processor.parse_filters(
            ...     name='John',           # Simple equality
            ...     age__gt=25,           # Greater than
            ...     status__or=['active', 'pending'],  # OR condition
            ...     _or={'city': 'NYC', 'state': 'CA'}  # Multi-field OR
            ... )
        """
        model = model or self.model
        filters = []

        if "_or" in kwargs:
            filters.extend(self._handle_multi_field_or_filter(model, kwargs.pop("_or")))

        for key, value in kwargs.items():
            if "." in key:
                filters.extend(self._handle_joined_filter(key, value))
            elif "__" not in key:
                filters.extend(self._handle_simple_filter(model, key, value))
            else:
                field_name, operator = key.rsplit("__", 1)

                if "." in field_name:
                    filters.extend(self._handle_joined_filter(key, value))
                else:
                    model_column = get_model_column(model, field_name)

                    if operator == "or":
                        filters.extend(self._handle_or_filter(model_column, value))
                    elif operator == "not":
                        filters.extend(self._handle_not_filter(model_column, value))
                    else:
                        filters.extend(
                            self._handle_standard_filter(model_column, operator, value)
                        )

        return filters

    def _handle_simple_filter(
        self,
        model: Union[type[ModelType], AliasedClass],
        key: str,
        value: FilterValueType,
    ) -> list[ColumnElement]:
        """
        Handle simple equality filters: field=value

        Args:
            model: Model to get the column from
            key: Field name
            value: Value to filter for

        Returns:
            List containing single equality condition
        """
        model_column = get_model_column(model, key)
        return [model_column == value]

    def _handle_or_filter(self, col: Column, value: dict) -> list[ColumnElement]:
        """
        Handle OR conditions: field__or={'gt': 18, 'lt': 65}

        This creates OR conditions for multiple operators on the same field.
        The value should be a dictionary mapping operators to their values.

        Args:
            col: SQLAlchemy column to apply conditions to
            value: Dictionary of operator -> value mappings

        Returns:
            List containing single OR condition

        Raises:
            ValueError: If value is not a dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("OR filter value must be a dictionary")

        or_conditions = []
        for or_op, or_value in value.items():
            if isinstance(or_value, list):
                for single_value in or_value:
                    filter_func = get_sqlalchemy_filter(
                        or_op, single_value, self.custom_filters
                    )
                    if filter_func:
                        condition = (
                            filter_func(col)(*single_value)
                            if or_op == "between"
                            else filter_func(col)(single_value)
                        )
                        or_conditions.append(condition)
            else:
                filter_func = get_sqlalchemy_filter(
                    or_op, or_value, self.custom_filters
                )
                if filter_func:
                    condition = (
                        filter_func(col)(*or_value)
                        if or_op == "between"
                        else filter_func(col)(or_value)
                    )
                    or_conditions.append(condition)

        return [or_(*or_conditions)] if or_conditions else []

    def _handle_not_filter(self, col: Column, value: dict) -> list[ColumnElement]:
        """
        Handle NOT conditions: field__not={'gt': 5}

        This creates individual NOT conditions for each operator on the same field.
        The result is [NOT condition1, NOT condition2, ...] (all must be true)

        Args:
            col: SQLAlchemy column to apply conditions to
            value: Dictionary of operator -> value mappings

        Returns:
            List containing individual NOT conditions

        Raises:
            ValueError: If value is not a dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("NOT filter value must be a dictionary")

        not_conditions = []
        for not_op, not_value in value.items():
            if isinstance(not_value, list):
                for single_value in not_value:
                    filter_func = get_sqlalchemy_filter(
                        not_op, single_value, self.custom_filters
                    )
                    if filter_func:
                        condition = (
                            filter_func(col)(*single_value)
                            if not_op == "between"
                            else filter_func(col)(single_value)
                        )
                        not_conditions.append(not_(condition))
            else:
                filter_func = get_sqlalchemy_filter(
                    not_op, not_value, self.custom_filters
                )
                if filter_func:
                    condition = (
                        filter_func(col)(*not_value)
                        if not_op == "between"
                        else filter_func(col)(not_value)
                    )
                    not_conditions.append(not_(condition))

        return [and_(*not_conditions)] if not_conditions else []

    def _handle_standard_filter(
        self, col: Column, operator: str, value: FilterValueType
    ) -> list[ColumnElement]:
        """
        Handle standard operator filters: field__gt=5

        Args:
            col: SQLAlchemy column to apply condition to
            operator: Filter operator string
            value: Value to filter for

        Returns:
            List containing single condition

        Raises:
            ValueError: If operator is not supported
        """
        filter_func = get_sqlalchemy_filter(operator, value, self.custom_filters)
        if not filter_func:
            raise ValueError(f"Unsupported filter operator: {operator}")

        if operator == "between":
            if isinstance(value, (tuple, list, set)):
                return [filter_func(col)(*value)]
            else:
                raise ValueError("Between operator requires a sequence value")
        else:
            return [filter_func(col)(value)]

    def _handle_multi_field_or_filter(
        self, model: Union[type[ModelType], AliasedClass], or_dict: dict
    ) -> list[ColumnElement]:
        """
        Handle multi-field OR: _or={'field1': value1, 'field2': value2}

        This creates OR conditions across different fields:
        field1 == value1 OR field2 == value2

        Args:
            model: Model to get columns from
            or_dict: Dictionary mapping field names to values

        Returns:
            List containing single OR condition

        Raises:
            ValueError: If or_dict is not a dictionary or contains invalid field names
        """
        if not isinstance(or_dict, dict):
            raise ValueError("Multi-field OR filter must be a dictionary")

        or_conditions = []
        for field, value in or_dict.items():
            if "." in field:
                or_conditions.extend(self._handle_joined_filter(field, value))
            elif "__" in field:
                field_name, operator = field.rsplit("__", 1)
                model_column = get_model_column(model, field_name)
                or_conditions.extend(
                    self._handle_standard_filter(model_column, operator, value)
                )
            else:
                model_column = get_model_column(model, field)
                or_conditions.append(model_column == value)

        return [or_(*or_conditions)] if or_conditions else []

    def _handle_joined_filter(
        self, filter_key: str, value: FilterValueType
    ) -> list[ColumnElement]:
        """
        Handle joined model filters: user.company.name__eq='Acme'

        This method traverses relationship paths to find the target column
        and applies the appropriate filter condition.

        Args:
            filter_key: Joined filter key (e.g., 'user.company.name__eq')
            value: Value to filter for

        Returns:
            List of filter conditions

        Raises:
            ValueError: If relationship path or field is invalid

        Example:
            >>> # For filter: 'tier.name__eq': 'Premium'
            >>> # Traverses: model -> tier relationship -> name field
            >>> # Returns: [TierModel.name == 'Premium']
        """
        validate_joined_filter_format(filter_key)

        if "__" in filter_key:
            field_path, operator = filter_key.rsplit("__", 1)
        else:
            field_path, operator = filter_key, None

        path_parts = field_path.split(".")
        if len(path_parts) < 2:
            raise ValueError(f"Invalid joined filter format: {filter_key}")

        relationship_path = path_parts[:-1]
        final_field = path_parts[-1]

        current_model = self.model
        for relationship_name in relationship_path:
            relationship = getattr(current_model, relationship_name, None)
            if relationship is None:
                raise ValueError(
                    f"Relationship '{relationship_name}' not found in model '{current_model.__name__}'"
                )

            if hasattr(relationship.property, "mapper"):
                current_model = relationship.property.mapper.class_
            else:
                raise ValueError(
                    f"Invalid relationship '{relationship_name}' in model '{current_model.__name__}'"
                )

        target_column = getattr(current_model, final_field, None)
        if target_column is None:
            raise ValueError(
                f"Column '{final_field}' not found in model '{current_model.__name__}'"
            )

        if operator is None:
            return [target_column == value]
        else:
            return self._handle_standard_filter(target_column, operator, value)

    def separate_joined_filters(
        self, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Detect and separate joined model filters from regular filters.

        This method analyzes filter arguments and separates those that target
        joined models (indicated by dot notation) from regular model filters.
        Uses existing core validation utilities for robust parsing.

        Args:
            **kwargs: Filter arguments to process

        Returns:
            tuple: (regular_filters, joined_filters_info)
            - regular_filters: Filters that apply to the base model
            - joined_filters_info: Dict mapping relationship names to their filters

        Example:
            >>> processor = FilterProcessor(User)
            >>> regular, joined = processor.separate_joined_filters(
            ...     name="John",
            ...     profile__bio__ilike="%developer%",
            ...     posts__title="My Post"
            ... )
            >>> # regular = {"name": "John"}
            >>> # joined = {
            >>> #     "profile": {"bio__ilike": "%developer%"},
            >>> #     "posts": {"title": "My Post"}
            >>> # }
        """
        regular_filters = {}
        joined_filters_info: dict[str, dict[str, Any]] = {}

        for key, value in kwargs.items():
            if "." in key:
                try:
                    validate_joined_filter_format(key)

                    if "__" in key:
                        field_path, operator = key.rsplit("__", 1)
                    else:
                        field_path, operator = key, None

                    path_parts = field_path.split(".")
                    relationship_name = path_parts[0]
                    remaining_path = ".".join(path_parts[1:])

                    if relationship_name not in joined_filters_info:
                        joined_filters_info[relationship_name] = {}

                    filter_key = remaining_path + (f"__{operator}" if operator else "")
                    joined_filters_info[relationship_name][filter_key] = value

                except ValueError:
                    regular_filters[key] = value
            else:
                regular_filters[key] = value

        return regular_filters, joined_filters_info
