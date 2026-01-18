"""
Data nesting functions that depend on introspection utilities.

This module contains functions for nesting joined data structures
that require model introspection but don't create circular dependencies.
"""

import logging
from typing import Any, Optional, Callable, TYPE_CHECKING

from .transforms import handle_one_to_one, handle_one_to_many, sort_nested_list

logger = logging.getLogger(__name__)

_LARGE_NESTED_THRESHOLD = 50

if TYPE_CHECKING:  # pragma: no cover
    from ..config import JoinConfig


def get_nested_key_for_join(join_config: "JoinConfig") -> str:
    """
    Determines the nested key name for a join configuration in the result data structure.

    This function extracts the appropriate key name that will be used to nest joined data
    in the final result. It prioritizes the custom join_prefix if provided, otherwise
    falls back to the model's table name.

    Args:
        join_config: The join configuration instance containing join configuration details.

    Returns:
        The string key name to be used for nesting the joined data.

    Examples:
        >>> class JoinConfig:
        ...     def __init__(self, model, join_prefix=None):
        ...         self.model = model
        ...         self.join_prefix = join_prefix
        ...
        >>> class Article:
        ...     __tablename__ = "articles"
        ...
        >>> join_config = JoinConfig(Article, join_prefix="articles_")
        >>> get_nested_key_for_join(join_config)
        "articles"

        >>> join_config = JoinConfig(Article)  # No prefix specified
        >>> get_nested_key_for_join(join_config)
        "articles"  # Uses model.__tablename__
    """
    return (
        join_config.join_prefix.rstrip("_")
        if join_config.join_prefix
        else join_config.model.__tablename__
    )


def process_joined_field(
    nested_data: dict[str, Any],
    join_config: "JoinConfig",
    nested_field: str,
    value: Any,
) -> dict[str, Any]:
    """
    Processes a single joined field and updates the nested data structure accordingly.

    This function handles the nesting of a single field from joined table data based on the
    relationship type defined in the join configuration. It delegates to the appropriate
    handler function for one-to-one or one-to-many relationships.

    Args:
        nested_data: The current nested data dictionary being built.
        join_config: The join configuration instance defining the join relationship type and configuration.
        nested_field: The name of the field being processed from the joined table.
        value: The value of the field being processed.

    Returns:
        The updated nested data dictionary with the processed field added.

    Examples:
        >>> nested_data = {"id": 1, "title": "Test"}
        >>> class MockJoinConfig:
        ...     relationship_type = "one-to-many"
        ...     join_prefix = "articles_"
        ...     model = type("Article", (), {"__tablename__": "articles"})
        >>> join_config = MockJoinConfig()
        >>> result = process_joined_field(nested_data, join_config, "title", "Article 1")
        >>> # Returns updated nested_data with article data nested appropriately
    """
    nested_key = get_nested_key_for_join(join_config)

    if join_config.relationship_type == "one-to-many":
        return handle_one_to_many(nested_data, nested_key, nested_field, value)
    else:
        return handle_one_to_one(nested_data, nested_key, nested_field, value)


def process_data_fields(
    data: dict,
    join_definitions: list["JoinConfig"],
    temp_prefix: str,
    nested_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Processes all fields in the flat data dictionary and nests joined data according to join definitions.

    This function iterates through all key-value pairs in the input data, identifying which fields
    belong to joined tables based on their prefixes, and nests them under their appropriate parent
    keys. Fields that don't match any join prefix are added directly to the nested data.

    Args:
        data: The flat dictionary containing data with potentially prefixed keys from joined tables.
        join_definitions: List of join configuration instances defining how to identify and nest joined data.
        temp_prefix: The temporary prefix used to identify joined fields (e.g., "joined__").
        nested_data: The target dictionary where nested data will be organized.

    Returns:
        The updated nested data dictionary with all fields properly organized.

    Example:
        Input data: {
            "id": 1,
            "name": "Author 1",
            "joined__articles_id": 10,
            "joined__articles_title": "Article Title"
        }

        Output: {
            "id": 1,
            "name": "Author 1",
            "articles": [{"id": 10, "title": "Article Title"}]
        }
    """
    for key, value in data.items():
        nested = False
        for join in join_definitions:
            join_prefix = join.join_prefix or ""
            full_prefix = f"{temp_prefix}{join_prefix}"

            if isinstance(key, str) and key.startswith(full_prefix):
                nested_field = key[len(full_prefix) :]
                nested_data = process_joined_field(
                    nested_data, join, nested_field, value
                )
                nested = True
                break

        if not nested:
            stripped_key = (
                key[len(temp_prefix) :]
                if isinstance(key, str) and key.startswith(temp_prefix)
                else key
            )
            nested_data[stripped_key] = value

    return nested_data


def cleanup_null_joins(
    nested_data: dict[str, Any],
    join_definitions: list["JoinConfig"],
    get_primary_key_func: Callable,
) -> dict[str, Any]:
    """
    Cleans up nested join data by handling null primary keys and applying sorting configurations.

    This function performs post-processing on nested join data to:
    1. Remove or replace entries with null primary keys (indicating no actual joined data)
    2. Apply sorting to one-to-many relationships when sort configurations are specified
    3. Convert one-to-one relationships with null primary keys to None

    Args:
        nested_data: The nested data dictionary containing organized joined data.
        join_definitions: List of join configuration instances with sorting and relationship configurations.
        get_primary_key_func: Function to get the primary key for a model.

    Returns:
        The cleaned nested data dictionary with null entries handled and sorting applied.

    Example:
        Before cleanup:
        {
            "id": 1,
            "articles": [{"id": None, "title": None}, {"id": 2, "title": "Real Article"}],
            "profile": {"id": None, "bio": None}
        }

        After cleanup:
        {
            "id": 1,
            "articles": [{"id": 2, "title": "Real Article"}],  # Null entry removed
            "profile": None  # Null one-to-one converted to None
        }
    """
    for join in join_definitions:
        join_primary_key = get_primary_key_func(join.model)
        nested_key = get_nested_key_for_join(join)

        if join.relationship_type == "one-to-many" and nested_key in nested_data:
            if isinstance(nested_data.get(nested_key, []), list):
                if any(
                    item[join_primary_key] is None for item in nested_data[nested_key]
                ):
                    nested_data[nested_key] = []
                else:
                    nested_count = len(nested_data[nested_key])
                    if (
                        nested_count >= _LARGE_NESTED_THRESHOLD
                        and join.nested_limit is None
                    ):
                        model_name = getattr(join.model, "__name__", str(join.model))
                        logger.warning(
                            "One-to-many relationship '%s' returned %d nested items. "
                            "This may cause performance issues. Consider setting "
                            "'nested_limit' in JoinConfig to enable SQL-level limiting "
                            "with separate queries, or use 'default_nested_limit' "
                            "in crud_router/EndpointCreator.",
                            model_name,
                            nested_count,
                        )

                    if join.sort_columns and nested_data[nested_key]:
                        nested_data[nested_key] = sort_nested_list(
                            nested_data[nested_key], join.sort_columns, join.sort_orders
                        )
                    if join.nested_limit is not None and nested_data[nested_key]:
                        nested_data[nested_key] = nested_data[nested_key][
                            : join.nested_limit
                        ]

        if nested_key in nested_data and isinstance(nested_data[nested_key], dict):
            if (
                join_primary_key in nested_data[nested_key]
                and nested_data[nested_key][join_primary_key] is None
            ):
                nested_data[nested_key] = None

    return nested_data


def nest_join_data(
    data: dict,
    join_definitions: list["JoinConfig"],
    get_primary_key_func: Callable,
    temp_prefix: str = "joined__",
    nested_data: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Nests joined data based on join definitions provided. This function processes the input `data` dictionary,
    identifying keys that correspond to joined tables using the provided `join_definitions` and nest them
    under their respective table keys.

    Args:
        data: The flat dictionary containing data with potentially prefixed keys from joined tables.
        join_definitions: A list of join configuration instances defining the join configurations, including prefixes.
        get_primary_key_func: Function to get the primary key for a model.
        temp_prefix: The temporary prefix applied to joined columns to differentiate them. Defaults to `"joined__"`.
        nested_data: The nested dictionary to which the data will be added. If None, a new dictionary is created. Defaults to `None`.

    Returns:
        dict[str, Any]: A dictionary with nested structures for joined table data.

    Examples:

        Input:

        ```python
        data = {
            'id': 1,
            'title': 'Test Author',
            'joined__articles_id': 1,
            'joined__articles_title': 'Article 1',
            'joined__articles_author_id': 1
        }

        join_definitions = [
            JoinConfig(
                model=Article,
                join_prefix='articles_',
                relationship_type='one-to-many',
            ),
        ]
        ```

        Output:

        ```json
        {
            'id': 1,
            'title': 'Test Author',
            'articles': [
                {
                    'id': 1,
                    'title': 'Article 1',
                    'author_id': 1
                }
            ]
        }
        ```

        Input:

        ```python
        data = {
            'id': 1,
            'title': 'Test Article',
            'joined__author_id': 1,
            'joined__author_name': 'Author 1'
        }

        join_definitions = [
            JoinConfig(
                model=Author,
                join_prefix='author_',
                relationship_type='one-to-one',
            ),
        ]
        ```

        Output:

        ```json
        {
            'id': 1,
            'title': 'Test Article',
            'author': {
                'id': 1,
                'name': 'Author 1'
            }
        }
        ```
    """
    if nested_data is None:
        nested_data = {}

    nested_data = process_data_fields(data, join_definitions, temp_prefix, nested_data)
    nested_data = cleanup_null_joins(
        nested_data, join_definitions, get_primary_key_func
    )

    assert nested_data is not None, "Couldn't nest the data."
    return nested_data
