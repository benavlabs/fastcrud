"""
CRUD-specific data preparation utilities.

This module contains utilities for preparing and validating data
before CRUD operations.
"""

from typing import Any
from datetime import datetime, timezone

from sqlalchemy import inspect as sa_inspect


def prepare_update_data(
    object: dict[str, Any] | Any,
    model_col_names: list[str],
    updated_at_column: str,
    model_instance: Any,
) -> dict[str, Any]:
    """
    Prepare and validate update data.

    Args:
        object: Update data as dict or Pydantic model
        model_col_names: List of valid column names for the model
        updated_at_column: Name of the updated_at column
        model_instance: Model instance to check for updated_at column existence

    Returns:
        Validated update data dictionary

    Raises:
        ValueError: If extra fields are provided that don't exist in the model
    """
    if isinstance(object, dict):
        update_data = object.copy()
    else:
        update_data = object.model_dump(exclude_unset=True)

    updated_at_col = getattr(model_instance, updated_at_column, None)
    if updated_at_col:
        update_data[updated_at_column] = datetime.now(timezone.utc)

    update_data_keys = set(update_data.keys())
    extra_fields = update_data_keys - set(model_col_names)
    if extra_fields:
        raise ValueError(f"Extra fields provided: {extra_fields}")

    return update_data


def instantiate_related_objects(
    data: dict[str, Any], model: type[Any]
) -> dict[str, Any]:
    """Turn nested dicts in relationship fields into model instances.

    ``model_dump()`` flattens a nested Pydantic model into a plain dict, which
    SQLAlchemy can't assign to a relationship - it raises ``AttributeError:
    'dict' object has no attribute '_sa_instance_state'``. Every relationship
    named in ``data`` is rebuilt as its own model, to any depth, so a create can
    carry the objects it relates to.

    A to-many relationship takes a list and a to-one takes a single object; the
    wrong shape raises ``ValueError`` here, naming the field, rather than
    surfacing as a SQLAlchemy error further along. ``None`` means the schema
    carried no related objects, so the field is dropped and the row is created
    without touching that relationship.

    Args:
        data: The create data, already dumped from the schema.
        model: The SQLAlchemy model the data is for.

    Returns:
        A copy of ``data`` with relationship values as model instances.

    Raises:
        ValueError: A to-many relationship was given something other than a
            list, or a to-one relationship was given a list.
    """
    prepared = dict(data)
    for relationship in sa_inspect(model).relationships:
        field = relationship.key
        if field not in prepared:
            continue

        value = prepared[field]
        if value is None:
            del prepared[field]
            continue

        related_model = relationship.mapper.class_

        if relationship.uselist:
            if not isinstance(value, list):
                raise ValueError(
                    f"'{field}' is a to-many relationship on {model.__name__} and takes a list, "
                    f"got {type(value).__name__}."
                )
            prepared[field] = [_as_related(item, related_model) for item in value]
        elif isinstance(value, list):
            raise ValueError(
                f"'{field}' is a to-one relationship on {model.__name__} and takes a single "
                f"object, got a list."
            )
        else:
            prepared[field] = _as_related(value, related_model)

    return prepared


def _as_related(value: Any, related_model: type[Any]) -> Any:
    """One relationship value as a model instance, recursing into its own relationships."""
    if not isinstance(value, dict):
        return value
    return related_model(**instantiate_related_objects(value, related_model))
