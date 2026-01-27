"""
Schema and field injection utilities with caching for performance optimization.

This module provides functions for managing Pydantic schemas, field injection,
and column extraction with caching where beneficial for performance.
"""

import logging
from functools import lru_cache
from typing import Any, Sequence, cast

from pydantic import BaseModel, create_model
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.sql import ColumnElement
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.orm import aliased

from .introspection import validate_model_has_table
from .data import build_column_label
from .config import JoinConfig
from ..types import ModelType, SelectSchemaType

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


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
    model_or_alias: ModelType | AliasedClass,
    schema: type[SelectSchemaType],
    mapper,
    prefix: str | None,
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
    model_or_alias: ModelType | AliasedClass,
    mapper,
    prefix: str | None,
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
    model: ModelType | AliasedClass,
    schema: type[SelectSchemaType] | None,
    prefix: str | None = None,
    alias: AliasedClass | None = None,
    use_temporary_prefix: bool | None = False,
    temp_prefix: str | None = "joined__",
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


def _find_join_condition(
    fk_model: ModelType,
    target_model: ModelType,
    fk_alias: AliasedClass | None = None,
) -> ColumnElement | None:
    """
    Helper function to find a join condition from fk_model to target_model.

    Args:
        fk_model: The model that may have a foreign key
        target_model: The model that the FK may reference
        fk_alias: Optional alias for the fk_model (the model being joined)

    Returns:
        Join condition if found, None otherwise
    """
    inspector = sa_inspect(fk_model)
    if inspector is None:
        return None

    fk_columns = [col for col in inspector.c if col.foreign_keys]
    fk_ref = fk_alias if fk_alias is not None else fk_model.__table__

    join_on = next(
        (
            cast(
                ColumnElement,
                fk_ref.c[col.name]
                == target_model.__table__.c[list(col.foreign_keys)[0].column.name],
            )
            for col in fk_columns
            if list(col.foreign_keys)[0].column.table == target_model.__table__
        ),
        None,
    )

    return join_on


def auto_detect_join_condition(
    base_model: ModelType,
    join_model: ModelType,
    join_alias: AliasedClass | None = None,
):
    """
    Automatically detects the join condition for SQLAlchemy models based on foreign key relationships.

    Supports bidirectional FK detection:
    - Checks for FKs on base_model pointing to join_model
    - If not found, checks for FKs on join_model pointing to base_model

    Args:
        base_model: The base SQLAlchemy model from which to join.
        join_model: The SQLAlchemy model to join with the base model.
        join_alias: Optional alias for the join_model (used when joining to the same table multiple times).

    Returns:
        A SQLAlchemy `ColumnElement` representing the join condition, if successfully detected.

    Raises:
        ValueError: If the join condition cannot be automatically determined.
        AttributeError: If either base_model or join_model does not have a `__table__` attribute.
    """
    validate_model_has_table(base_model)
    validate_model_has_table(join_model)

    join_on = _find_join_condition(base_model, join_model, fk_alias=None)
    if join_on is None:
        join_on = _find_join_condition(join_model, base_model, fk_alias=join_alias)

    if join_on is None:
        raise ValueError(
            f"Could not automatically determine join condition between "
            f"{base_model.__name__} and {join_model.__name__}. "
            f"Please provide join_on explicitly."
        )

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
    relationship_names: list[str] | None = None,
    default_nested_limit: int | None = None,
    include_one_to_many: bool = False,
) -> list[JoinConfig]:
    """
    Auto-detect relationships and build JoinConfig list.

    For each relationship discovered on the model:
    - Auto-detects join condition via foreign keys using auto_detect_join_condition()
    - Determines relationship type (one-to-one vs one-to-many) from SQLAlchemy metadata
    - Creates JoinConfig with sensible defaults
    - Skips relationships where join condition cannot be auto-detected
    - Creates aliases for multiple relationships to the same model

    By default, only one-to-one relationships are included for safety. One-to-many
    relationships can return unbounded data via JOINs, which may cause performance
    issues. To include one-to-many relationships, either:
    - Explicitly list them in `relationship_names`
    - Set `include_one_to_many=True`

    Args:
        model: The SQLAlchemy model to build join configurations for.
        relationship_names: Optional list of relationship names to include.
            If None, relationships are filtered by `include_one_to_many`.
            If provided, only relationships with matching names will be included
            (both one-to-one and one-to-many, regardless of `include_one_to_many`).
        default_nested_limit: Optional default limit for nested items in
            one-to-many relationships. When set, each one-to-many relationship
            will return at most this many nested items. Note: This only filters
            results at the application level after the database query returns.
            Use None for no limit.
        include_one_to_many: Whether to include one-to-many relationships when
            auto-detecting. Defaults to False for safety. When False, only
            one-to-one relationships are auto-detected. Set to True to include
            all relationship types. This parameter is ignored when specific
            `relationship_names` are provided.

    Returns:
        List of JoinConfig objects for detected relationships that can be auto-joined.

    Raises:
        ValueError: If relationship_names contains names that don't exist on the model.

    Note:
        Relationships that cannot be auto-detected are logged at DEBUG level.
        Enable DEBUG logging to see which relationships were skipped and why.

    Example:
        ```python
        # Include only one-to-one relationships (safe default)
        joins_config = build_relationship_joins_config(User)

        # Include all relationships including one-to-many
        joins_config = build_relationship_joins_config(User, include_one_to_many=True)

        # Include only specific relationships (bypasses include_one_to_many filter)
        joins_config = build_relationship_joins_config(User, ["tier", "posts"])

        # Limit nested items to 10 per one-to-many relationship
        joins_config = build_relationship_joins_config(
            User, include_one_to_many=True, default_nested_limit=10
        )
        ```
    """
    relationships = discover_model_relationships(model)

    if relationship_names is not None:
        available_names = {name for name, _ in relationships}
        invalid_names = set(relationship_names) - available_names
        if invalid_names:
            raise ValueError(
                f"Invalid relationship names: {invalid_names}. "
                f"Available relationships on {model.__name__}: {available_names}"
            )
    joins_config: list[JoinConfig] = []
    model_counts: dict[Any, int] = {}
    skipped_relationships: list[tuple[str, str]] = []

    for rel_name, rel_prop in relationships:
        if relationship_names is not None and rel_name not in relationship_names:
            continue

        is_one_to_many = rel_prop.uselist
        if is_one_to_many and relationship_names is None and not include_one_to_many:
            skipped_relationships.append(
                (
                    rel_name,
                    "one-to-many excluded by default (use include_one_to_many=True)",
                )
            )
            logger.debug(
                "Skipping one-to-many relationship '%s' on %s. "
                "Set include_one_to_many=True or explicitly list it to include.",
                rel_name,
                model.__name__,
            )
            continue

        related_model = rel_prop.mapper.class_
        alias = None
        if related_model in model_counts:
            alias = aliased(related_model, name=rel_name)
        model_counts[related_model] = model_counts.get(related_model, 0) + 1

        try:
            join_on = auto_detect_join_condition(model, related_model, join_alias=alias)
        except ValueError as e:
            skipped_relationships.append((rel_name, str(e)))
            logger.debug(
                "Skipping relationship '%s' on %s: %s",
                rel_name,
                model.__name__,
                str(e),
            )
            continue
        except AttributeError as e:
            skipped_relationships.append((rel_name, f"AttributeError: {e}"))
            logger.debug(
                "Skipping relationship '%s' on %s due to missing attribute: %s",
                rel_name,
                model.__name__,
                str(e),
            )
            continue
        relationship_type = "one-to-many" if is_one_to_many else "one-to-one"
        nested_limit = (
            default_nested_limit if relationship_type == "one-to-many" else None
        )

        joins_config.append(
            JoinConfig(
                model=related_model,
                join_on=join_on,
                join_prefix=f"{rel_name}_",
                schema_to_select=None,
                join_type="left",
                relationship_type=relationship_type,
                alias=alias,
                nested_limit=nested_limit,
            )
        )

    if skipped_relationships:
        logger.debug(
            "Auto-detection for %s: %d relationships detected, %d skipped. "
            "Skipped: %s",
            model.__name__,
            len(joins_config),
            len(skipped_relationships),
            [name for name, _ in skipped_relationships],
        )

    return joins_config


def resolve_relationship_config(
    model: ModelType,
    include_relationships: bool | Sequence[str],
    default_nested_limit: int | None = None,
    include_one_to_many: bool = False,
) -> list[JoinConfig] | None:
    """
    Resolve include_relationships parameter to a list of JoinConfig objects.

    This helper function handles the different forms of the include_relationships
    parameter and returns the appropriate JoinConfig list for use in joined queries.

    Args:
        model: The SQLAlchemy model to resolve relationships for.
        include_relationships: Controls which relationships to include:
            - False: Don't include any relationships (returns None)
            - True: Include auto-detectable relationships (filtered by include_one_to_many)
            - List of strings: Include only the named relationships (all types included)
        default_nested_limit: Optional default limit for nested items in
            one-to-many relationships. When set, each one-to-many relationship
            will return at most this many nested items. Note: This only filters
            results at the application level. Use None for no limit.
        include_one_to_many: Whether to include one-to-many relationships when
            auto-detecting with `include_relationships=True`. Defaults to False
            for safety since one-to-many JOINs can return unbounded data. This
            parameter is ignored when specific relationship names are provided.

    Returns:
        List of JoinConfig objects, or None if include_relationships is False.

    Raises:
        ValueError: If include_relationships contains invalid relationship names.

    Example:
        ```python
        # No relationships
        config = resolve_relationship_config(User, False)  # Returns None

        # Only one-to-one relationships (safe default)
        config = resolve_relationship_config(User, True)

        # All relationships including one-to-many
        config = resolve_relationship_config(User, True, include_one_to_many=True)

        # Specific relationships only (all types included)
        config = resolve_relationship_config(User, ["tier", "posts"])

        # With nested limit for one-to-many
        config = resolve_relationship_config(
            User, True, include_one_to_many=True, default_nested_limit=10
        )
        ```
    """
    if include_relationships is False:
        return None

    relationship_names: list[str] | None = None
    if include_relationships is not True:
        relationship_names = list(include_relationships)

    return build_relationship_joins_config(
        model, relationship_names, default_nested_limit, include_one_to_many
    )
