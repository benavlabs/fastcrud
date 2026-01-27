"""
Core model introspection utilities with caching for performance optimization.

This module provides cached SQLAlchemy model introspection capabilities to avoid
repeated expensive operations. It includes both class-based and functional approaches
for different use cases.
"""

from typing import Sequence, Any, cast
from uuid import UUID

from sqlalchemy import Column, inspect as sa_inspect
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.types import TypeEngine
from sqlalchemy.sql.elements import KeyedColumnElement
from sqlalchemy.orm.util import AliasedClass

from ..types import ModelType


class ModelInspector:
    """
    Cached SQLAlchemy model introspection with memory management.

    This class provides an efficient way to inspect SQLAlchemy models by caching
    expensive introspection operations. It's particularly useful when the same
    model needs to be inspected multiple times during complex join operations.

    Attributes:
        model: The SQLAlchemy model to inspect.

    Example:
        >>> inspector = ModelInspector(User)
        >>> pk_names = inspector.primary_key_names  # Cached
        >>> column_types = inspector.column_types   # Uses same cached inspector
        >>> pk = inspector.primary_key             # Fast access to first PK
    """

    def __init__(self, model: ModelType):
        self.model = model
        self._inspector = None
        self._pk_names_cache: list[str] | None = None
        self._column_types_cache: dict[str, type | None] | None = None

    @property
    def inspector(self):
        """
        Cached SQLAlchemy inspector - expensive operation cached per instance.

        Returns:
            SQLAlchemy inspector object for the model.

        Raises:
            ValueError: If model inspection fails.
        """
        if self._inspector is None:
            self._inspector = sa_inspect(self.model)
            if self._inspector is None:  # pragma: no cover
                raise ValueError(f"Model inspection failed for {self.model}")
        return self._inspector

    @property
    def primary_key_names(self) -> list[str]:
        """
        Get primary key names using cached inspector.

        Returns:
            List of primary key column names.

        Example:
            >>> inspector = ModelInspector(User)
            >>> inspector.primary_key_names
            ['id']

            >>> inspector = ModelInspector(CompositeModel)
            >>> inspector.primary_key_names
            ['user_id', 'version']
        """
        if self._pk_names_cache is None:
            pk_names = [pk.name for pk in self.inspector.mapper.primary_key]
            self._pk_names_cache = pk_names
        return self._pk_names_cache

    @property
    def primary_key_columns(self) -> Sequence[Column]:
        """
        Get primary key columns using cached inspector.

        Returns:
            Sequence of primary key Column objects.
        """
        return cast(Sequence[Column], self.inspector.mapper.primary_key)

    @property
    def column_types(self) -> dict[str, type | None]:
        """
        Get column types using cached inspector with UUID handling.

        Returns:
            Dictionary mapping column names to their Python types.

        Example:
            >>> inspector = ModelInspector(User)
            >>> inspector.column_types
            {'id': <class 'int'>, 'name': <class 'str'>, 'uuid_field': <class 'uuid.UUID'>}
        """
        if self._column_types_cache is None:
            column_types: dict[str, type | None] = {}
            for column in self.inspector.mapper.columns:
                column_type = get_python_type(column)
                if (
                    hasattr(column.type, "__visit_name__")
                    and column.type.__visit_name__ == "uuid"
                ):
                    column_type = UUID
                column_types[column.name] = column_type
            self._column_types_cache = column_types
        return self._column_types_cache

    @property
    def first_primary_key(self) -> str:
        """
        Get first primary key name.

        Returns:
            The name of the first primary key column.

        Example:
            >>> inspector = ModelInspector(User)
            >>> inspector.first_primary_key
            'id'
        """
        return self.primary_key_names[0]

    @property
    def unique_columns(self) -> Sequence[KeyedColumnElement]:
        """
        Get columns marked as unique using cached inspector.

        Returns:
            Sequence of unique column elements.

        Raises:
            AttributeError: If model doesn't have __table__ attribute.
        """
        validate_model_has_table(self.model)
        return [column for column in self.model.__table__.columns if column.unique]


def get_model_inspector(model: ModelType) -> ModelInspector:
    """
    Factory function to create ModelInspector instances.

    Args:
        model: The SQLAlchemy model to inspect.

    Returns:
        A ModelInspector instance for the model.

    Example:
        >>> inspector = get_model_inspector(User)
        >>> inspector.primary_key_names
        ['id']
    """
    return ModelInspector(model)


def get_primary_key_names(model: ModelType) -> tuple[str, ...]:
    """
    Get primary key names for a model (uses ModelInspector for caching).

    Args:
        model: The SQLAlchemy model to inspect.

    Returns:
        Tuple of primary key column names (immutable for caching).

    Example:
        >>> get_primary_key_names(User)
        ('id',)
        >>> get_primary_key_names(CompositeModel)
        ('user_id', 'version')
    """
    inspector = get_model_inspector(model)
    return tuple(inspector.primary_key_names)


def get_column_types(model: ModelType) -> tuple:
    """
    Get column types as tuple for compatibility.

    Args:
        model: The SQLAlchemy model to inspect.

    Returns:
        Tuple of (column_name, type) pairs for caching compatibility.

    Example:
        >>> get_column_types(User)
        (('id', <class 'int'>), ('name', <class 'str'>), ...)
    """
    inspector = get_model_inspector(model)
    return tuple(inspector.column_types.items())


def get_first_primary_key(model: ModelType) -> str:
    """
    Get the first primary key name for a model.

    Args:
        model: The SQLAlchemy model to inspect.

    Returns:
        The name of the first primary key column.
    """
    inspector = get_model_inspector(model)
    return inspector.first_primary_key


def get_primary_key_columns(model: ModelType) -> Sequence[Column]:
    """
    Get primary key columns for a SQLAlchemy model.

    Args:
        model: The SQLAlchemy model to inspect.

    Returns:
        Sequence of primary key Column objects.

    Example:
        >>> columns = get_primary_key_columns(User)
        >>> columns[0].name
        'id'
    """
    inspector = get_model_inspector(model)
    return inspector.primary_key_columns


def get_unique_columns(model: ModelType) -> Sequence[KeyedColumnElement]:
    """
    Get columns marked as unique for a SQLAlchemy model.

    Args:
        model: The SQLAlchemy model to inspect.

    Returns:
        Sequence of unique column elements.

    Example:
        >>> unique_cols = get_unique_columns(User)
        >>> [col.name for col in unique_cols if col.unique]
        ['email', 'username']
    """
    inspector = get_model_inspector(model)
    return inspector.unique_columns


def validate_model_has_table(model) -> None:
    """
    Validates that a model has a __table__ attribute.

    Args:
        model: The model to validate.

    Raises:
        AttributeError: If the model doesn't have a __table__ attribute.
    """
    if not hasattr(model, "__table__"):
        raise AttributeError(f"{model.__name__} does not have a '__table__' attribute.")


def is_uuid_type(column_type: TypeEngine) -> bool:
    """
    Check if a SQLAlchemy column type represents a UUID.
    Handles various SQL dialects and common UUID implementations.

    Args:
        column_type: The SQLAlchemy column type to check.

    Returns:
        True if the column type represents a UUID, False otherwise.
    """
    if isinstance(column_type, PostgresUUID):
        return True

    type_name = getattr(column_type, "__visit_name__", "").lower()
    if "uuid" in type_name:
        return True

    if hasattr(column_type, "impl"):
        return is_uuid_type(column_type.impl)

    return False


def get_python_type(column: Column) -> type | None:
    """
    Get the Python type for a SQLAlchemy column, with special handling for UUIDs.

    Args:
        column: The SQLAlchemy column to get the type for.

    Returns:
        The corresponding Python type, or None if it cannot be determined.

    Raises:
        NotImplementedError: If the column uses a custom type without proper type mapping.
    """
    try:
        if is_uuid_type(column.type):
            return UUID

        direct_type: type | None = column.type.python_type
        return direct_type
    except NotImplementedError:
        if hasattr(column.type, "impl") and hasattr(column.type.impl, "python_type"):
            if is_uuid_type(column.type.impl):
                return UUID
            indirect_type: type | None = column.type.impl.python_type
            return indirect_type
        else:
            raise NotImplementedError(
                f"The primary key column {column.name} uses a custom type without a defined `python_type` or suitable `impl` fallback."
            )


def create_composite_key(item: dict, pk_names: list[str]) -> tuple:
    """
    Create a composite key tuple from an item using primary key names.

    Args:
        item: Dictionary containing the data.
        pk_names: List of primary key field names.

    Returns:
        Tuple representing the composite primary key.

    Example:
        >>> create_composite_key({"id": 1, "version": 2, "name": "test"}, ["id", "version"])
        (1, 2)
    """
    return tuple(item.get(pk_name) for pk_name in pk_names)


def get_model_column(model: ModelType | AliasedClass, field_name: str) -> Column[Any]:
    """
    Get column from model, raising ValueError if not found.

    This utility function retrieves a column attribute from a SQLAlchemy model
    or aliased model, providing consistent error handling across the codebase.
    It's designed to replace the repetitive getattr/error checking pattern
    used throughout FastCRUD.

    Args:
        model: SQLAlchemy model or alias from which to get the column
        field_name: Name of the field/column to retrieve

    Returns:
        SQLAlchemy Column object for the specified field

    Raises:
        ValueError: If field doesn't exist on the model

    Example:
        >>> user_name_col = get_model_column(User, "name")
        >>> user_id_col = get_model_column(aliased_user, "id")

        >>> # This will raise ValueError
        >>> invalid_col = get_model_column(User, "nonexistent_field")
    """
    model_column = getattr(model, field_name, None)
    if model_column is None:
        model_name = getattr(model, "__name__", str(model))
        raise ValueError(f"Invalid column '{field_name}' for model {model_name}")
    return cast(Column[Any], model_column)
