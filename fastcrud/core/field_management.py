"""
Schema and field injection utilities with caching for performance optimization.

This module provides functions for managing Pydantic schemas, field injection,
and column extraction with caching where beneficial for performance.
"""

from functools import lru_cache
from typing import Any, Optional, Union, cast

from pydantic import BaseModel, create_model
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.sql import ColumnElement
from sqlalchemy.orm.util import AliasedClass

from .introspection import validate_model_has_table
from .data import build_column_label
from .config import JoinConfig
from ..types import ModelType, SelectSchemaType

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


@lru_cache(maxsize=128)
def create_modified_schema(
    original_schema: type[BaseModel],
    exclude_fields: tuple[str, ...],
    schema_name: str = "ModifiedSchema",
) -> type[BaseModel]:
    """
    Creates a new Pydantic schema with specified fields excluded - expensive operation, cache it.

    This function dynamically creates a new Pydantic schema class that excludes certain fields
    from the original schema. This is particularly useful for auto field injection where
    certain fields should not appear in API documentation or request schemas.

    Args:
        original_schema: The original Pydantic schema class.
        exclude_fields: Tuple of field names to exclude (tuple for cache hashability).
        schema_name: Name for the new schema class.

    Returns:
        A new Pydantic schema class without the excluded fields.

    Example:
        >>> class UserSchema(BaseModel):
        ...     id: int
        ...     name: str
        ...     email: str
        ...     password: str
        ...
        >>> # Create schema without sensitive fields
        >>> public_schema = create_modified_schema(
        ...     UserSchema,
        ...     ("password",),
        ...     "PublicUserSchema"
        ... )
        >>> # New schema only has id, name, email fields
    """
    if not exclude_fields:
        return original_schema

    field_definitions: dict[str, Any] = {}
    for field_name, field_info in original_schema.model_fields.items():
        if field_name not in exclude_fields:
            field_definitions[field_name] = (field_info.annotation, field_info)

    new_schema: type[BaseModel] = create_model(
        schema_name,
        **field_definitions,  # type: ignore[arg-type]
    )

    return new_schema


def extract_schema_columns(
    model_or_alias: Union[ModelType, AliasedClass],
    schema: type[SelectSchemaType],
    mapper,
    prefix: Optional[str],
    use_temporary_prefix: bool,
    temp_prefix: str,
) -> list[Any]:
    """
    Extracts specific columns from a SQLAlchemy model based on Pydantic schema field names.

    This function matches the field names defined in the provided Pydantic schema with the columns
    available in the SQLAlchemy model, excluding relationship fields. Each matched column can be
    optionally labeled with prefixes for use in joined queries.

    Args:
        model_or_alias: The SQLAlchemy model or its alias from which to extract columns.
        schema: The Pydantic schema containing field names to match against model columns.
        mapper: The SQLAlchemy mapper for the model, used to identify relationships.
        prefix: Optional prefix to be added to column labels. If None, no prefix is added.
        use_temporary_prefix: Whether to use the temporary prefix for column labeling.
        temp_prefix: The temporary prefix string to be used if use_temporary_prefix is True.

    Returns:
        A list of SQLAlchemy column objects, potentially with custom labels applied.

    Example:
        >>> schema = AuthorSchema  # Has fields: id, name, email
        >>> columns = extract_schema_columns(Author, schema, mapper, "author_", True, "joined__")
        >>> # Returns [Author.id.label("joined__author_id"), Author.name.label("joined__author_name"), ...]
    """
    columns = []
    for field in schema.model_fields.keys():
        if hasattr(model_or_alias, field) and field not in mapper.relationships:
            column = getattr(model_or_alias, field)
            if prefix is not None or use_temporary_prefix:
                column_label = build_column_label(temp_prefix, prefix, field)
                column = column.label(column_label)
            columns.append(column)
    return columns


def extract_all_columns(
    model_or_alias: Union[ModelType, AliasedClass],
    mapper,
    prefix: Optional[str],
    use_temporary_prefix: bool,
    temp_prefix: str,
) -> list[Any]:
    """
    Extracts all available columns from a SQLAlchemy model.

    This function retrieves all column attributes from the provided SQLAlchemy model,
    excluding relationship fields. Each column can be optionally labeled with prefixes
    for use in joined queries.

    Args:
        model_or_alias: The SQLAlchemy model or its alias from which to extract all columns.
        mapper: The SQLAlchemy mapper for the model, used to access column attributes.
        prefix: Optional prefix to be added to column labels. If None, no prefix is added.
        use_temporary_prefix: Whether to use the temporary prefix for column labeling.
        temp_prefix: The temporary prefix string to be used if use_temporary_prefix is True.

    Returns:
        A list of all SQLAlchemy column objects from the model, potentially with custom labels applied.

    Example:
        >>> columns = extract_all_columns(User, mapper, "user_", True, "joined__")
        >>> # Returns [User.id.label("joined__user_id"), User.name.label("joined__user_name"), ...]
    """
    columns = []
    for prop in mapper.column_attrs:
        column = getattr(model_or_alias, prop.key)
        if prefix is not None or use_temporary_prefix:
            column_label = build_column_label(temp_prefix, prefix, prop.key)
            column = column.label(column_label)
        columns.append(column)
    return columns


def extract_matching_columns_from_schema(
    model: Union[ModelType, AliasedClass],
    schema: Optional[type[SelectSchemaType]],
    prefix: Optional[str] = None,
    alias: Optional[AliasedClass] = None,
    use_temporary_prefix: Optional[bool] = False,
    temp_prefix: Optional[str] = "joined__",
) -> list[Any]:
    """
    Retrieves a list of ORM column objects from a SQLAlchemy model that match the field names in a given Pydantic schema,
    or all columns from the model if no schema is provided. When an alias is provided, columns are referenced through
    this alias, and a prefix can be applied to column names if specified.

    Args:
        model: The SQLAlchemy ORM model containing columns to be matched with the schema fields.
        schema: Optional; a Pydantic schema containing field names to be matched with the model's columns. If `None`, all columns from the model are used.
        prefix: Optional; a prefix to be added to all column names. If `None`, no prefix is added.
        alias: Optional; an alias for the model, used for referencing the columns through this alias in the query. If `None`, the original model is used.
        use_temporary_prefix: Whether to use or not an aditional prefix for joins. Default `False`.
        temp_prefix: The temporary prefix to be used. Default `"joined__"`.

    Returns:
        A list of ORM column objects (potentially labeled with a prefix) that correspond to the field names defined
        in the schema or all columns from the model if no schema is specified. These columns are correctly referenced
        through the provided alias if one is given.
    """
    validate_model_has_table(model)

    model_or_alias = alias if alias else model
    temp_prefix = (
        temp_prefix if use_temporary_prefix and temp_prefix is not None else ""
    )
    mapper = sa_inspect(model).mapper

    use_temp_prefix = (
        use_temporary_prefix if use_temporary_prefix is not None else False
    )
    if schema:
        return extract_schema_columns(
            model_or_alias, schema, mapper, prefix, use_temp_prefix, temp_prefix
        )
    else:
        return extract_all_columns(
            model_or_alias, mapper, prefix, use_temp_prefix, temp_prefix
        )


def auto_detect_join_condition(
    base_model: ModelType,
    join_model: ModelType,
):
    """
    Automatically detects the join condition for SQLAlchemy models based on foreign key relationships.

    Args:
        base_model: The base SQLAlchemy model from which to join.
        join_model: The SQLAlchemy model to join with the base model.

    Returns:
        A SQLAlchemy `ColumnElement` representing the join condition, if successfully detected.

    Raises:
        ValueError: If the join condition cannot be automatically determined.
        AttributeError: If either base_model or join_model does not have a `__table__` attribute.
    """
    validate_model_has_table(base_model)
    validate_model_has_table(join_model)

    inspector = sa_inspect(base_model)
    if inspector is not None:
        fk_columns = [col for col in inspector.c if col.foreign_keys]
        join_on = next(
            (
                cast(
                    ColumnElement,
                    base_model.__table__.c[col.name]
                    == join_model.__table__.c[list(col.foreign_keys)[0].column.name],
                )
                for col in fk_columns
                if list(col.foreign_keys)[0].column.table == join_model.__table__
            ),
            None,
        )

        if join_on is None:
            raise ValueError(
                "Could not automatically determine join condition. Please provide join_on."
            )
    else:
        raise ValueError("Could not automatically get model columns.")

    return join_on


def discover_model_relationships(
    model: ModelType,
) -> list[tuple[str, Any]]:
    """
    Discover all SQLAlchemy relationships defined on a model.

    This function inspects a SQLAlchemy model and returns all defined relationships,
    which can then be used to automatically build join configurations.

    Args:
        model: The SQLAlchemy model to inspect.

    Returns:
        List of tuples: (relationship_name, relationship_property)

    Example:
        ```python
        relationships = discover_model_relationships(User)
        for name, rel_prop in relationships:
            print(f"Found relationship: {name}")
        ```
    """
    inspector = sa_inspect(model)
    return list(inspector.relationships.items())


def build_relationship_joins_config(
    model: ModelType,
) -> list[Any]:
    """
    Auto-detect all relationships and build JoinConfig list.

    For each relationship discovered on the model:
    - Auto-detects join condition via foreign keys using auto_detect_join_condition()
    - Determines relationship type (one-to-one vs one-to-many) from SQLAlchemy metadata
    - Creates JoinConfig with sensible defaults
    - Skips relationships where join condition cannot be auto-detected

    Args:
        model: The SQLAlchemy model to build join configurations for.

    Returns:
        List of JoinConfig objects for all detected relationships that can be auto-joined.

    Example:
        ```python
        joins_config = build_relationship_joins_config(User)
        # Returns JoinConfig for each relationship (e.g., profile, posts, etc.)
        ```
    """
    relationships = discover_model_relationships(model)
    joins_config = []

    for rel_name, rel_prop in relationships:
        related_model = rel_prop.mapper.class_

        try:
            # Try to auto-detect join condition using existing function
            join_on = auto_detect_join_condition(model, related_model)
        except (ValueError, AttributeError):
            # Skip relationships where join condition cannot be auto-detected
            # (e.g., complex relationships, multiple primary keys, etc.)
            continue

        # Determine relationship type from SQLAlchemy property
        relationship_type = "one-to-many" if rel_prop.uselist else "one-to-one"

        joins_config.append(
            JoinConfig(
                model=related_model,
                join_on=join_on,
                join_prefix=f"{rel_name}_",
                schema_to_select=None,  # Include all columns
                join_type="left",
                relationship_type=relationship_type,
            )
        )

    return joins_config
