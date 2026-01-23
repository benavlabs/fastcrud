"""
CRUD-specific data preparation utilities.

This module contains utilities for preparing and validating data
before CRUD operations.
"""

from typing import Any
from datetime import datetime, timezone


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
