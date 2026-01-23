"""
Pure data transformation functions with no external dependencies.

This module contains stateless functions for basic data manipulation,
formatting, and transformation that don't depend on FastCRUD internals.
All functions are pure (no side effects) and have minimal dependencies.
"""

from typing import Any

from ...types import SelectSchemaType


def handle_one_to_one(
    nested_data: dict[str, Any], nested_key: str, nested_field: str, value: Any
) -> dict[str, Any]:
    """
    Handles the nesting of one-to-one relationships in the data.

    Args:
        nested_data: The current state of the nested data.
        nested_key: The key under which the nested data should be stored.
        nested_field: The field name of the nested data to be added.
        value: The value of the nested data to be added.

    Returns:
        dict[str, Any]: The updated nested data dictionary.

    Examples:

        Input:

        ```python
        nested_data = {
            'id': 1,
            'name': 'Test Author',
        }
        nested_key = 'profile'
        nested_field = 'bio'
        value = 'This is a bio.'
        ```

        Output:

        ```json
        {
            'id': 1,
            'name': 'Test Author',
            'profile': {
                'bio': 'This is a bio.'
            }
        }
        ```
    """
    if nested_key not in nested_data or not isinstance(nested_data[nested_key], dict):
        nested_data[nested_key] = {}
    nested_data[nested_key][nested_field] = value
    return nested_data


def handle_one_to_many(
    nested_data: dict[str, Any], nested_key: str, nested_field: str, value: Any
) -> dict[str, Any]:
    """
    Handles the nesting of one-to-many relationships in the data.

    Args:
        nested_data: The current state of the nested data.
        nested_key: The key under which the nested data should be stored.
        nested_field: The field name of the nested data to be added.
        value: The value of the nested data to be added.

    Returns:
        dict[str, Any]: The updated nested data dictionary.

    Examples:

        Input:

        ```python
        nested_data = {
            'id': 1,
            'name': 'Test Author',
            'articles': [
                {
                    'title': 'First Article',
                    'content': 'Content of the first article!',
                }
            ],
        }
        nested_key = 'articles'
        nested_field = 'title'
        value = 'Second Article'
        ```

        Output:

        ```json
        {
            'id': 1,
            'name': 'Test Author',
            'articles': [
                {
                    'title': 'First Article',
                    'content': 'Content of the first article!'
                },
                {
                    'title': 'Second Article'
                }
            ]
        }
        ```

        Input:

        ```python
        nested_data = {
            'id': 1,
            'name': 'Test Author',
            'articles': [],
        }
        nested_key = 'articles'
        nested_field = 'title'
        value = 'First Article'
        ```

        Output:

        ```json
        {
            'id': 1,
            'name': 'Test Author',
            'articles': [
                {
                    'title': 'First Article'
                }
            ]
        }
        ```
    """
    if nested_key not in nested_data or not isinstance(nested_data[nested_key], list):
        nested_data[nested_key] = []

    if not nested_data[nested_key] or nested_field in nested_data[nested_key][-1]:
        nested_data[nested_key].append({nested_field: value})
    else:
        nested_data[nested_key][-1][nested_field] = value

    return nested_data


def sort_nested_list(
    nested_list: list[dict],
    sort_columns: str | list[str],
    sort_orders: str | list[str] | None = None,
) -> list[dict]:
    """
    Sorts a list of dictionaries based on specified sort columns and orders.

    Args:
        nested_list: The list of dictionaries to sort.
        sort_columns: A single column name or a list of column names on which to apply sorting.
        sort_orders: A single sort order ("asc" or "desc") or a list of sort orders corresponding
            to the columns in `sort_columns`. If not provided, defaults to "asc" for each column.

    Returns:
        The sorted list of dictionaries.

    Examples:
        Sorting a list of dictionaries by a single column in ascending order:
        >>> sort_nested_list([{"id": 2, "name": "B"}, {"id": 1, "name": "A"}], "name")
        [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]

        Sorting by multiple columns with different orders:
        >>> sort_nested_list([{"id": 1, "name": "A"}, {"id": 2, "name": "A"}], ["name", "id"], ["asc", "desc"])
        [{"id": 2, "name": "A"}, {"id": 1, "name": "A"}]
    """
    if not nested_list or not sort_columns:
        return nested_list

    if not isinstance(sort_columns, list):
        sort_columns = [sort_columns]

    if sort_orders:
        if not isinstance(sort_orders, list):
            sort_orders = [sort_orders] * len(sort_columns)
        if len(sort_columns) != len(sort_orders):
            raise ValueError("The length of sort_columns and sort_orders must match.")

        for order in sort_orders:
            if order not in ["asc", "desc"]:
                raise ValueError(
                    f"Invalid sort order: {order}. Only 'asc' or 'desc' are allowed."
                )
    else:  # pragma: no cover
        sort_orders = ["asc"] * len(sort_columns)

    sort_specs = [
        (col, 1 if order == "asc" else -1)
        for col, order in zip(sort_columns, sort_orders)
    ]

    sorted_list = nested_list.copy()
    for col, direction in reversed(sort_specs):
        sorted_list.sort(
            key=lambda x: (x.get(col) is None, x.get(col)), reverse=direction == -1
        )

    return sorted_list


def build_column_label(temp_prefix: str, prefix: str | None, field_name: str) -> str:
    """
    Builds a column label with appropriate prefixes for SQLAlchemy column selection.

    Args:
        temp_prefix: The temporary prefix to be prepended to the column label.
        prefix: Optional prefix to be added between temp_prefix and field_name. If None, only temp_prefix is used.
        field_name: The base field name for the column.

    Returns:
        A formatted column label string combining the prefixes and field name.

    Examples:
        >>> build_column_label("joined__", "articles_", "title")
        "joined__articles_title"

        >>> build_column_label("joined__", None, "id")
        "joined__id"
    """
    if prefix:
        return f"{temp_prefix}{prefix}{field_name}"
    else:
        return f"{temp_prefix}{field_name}"


def format_single_response(
    data: Any, schema_to_select: type | None = None, return_as_model: bool = False
) -> dict | Any:
    """
    Format single record response with optional model conversion.

    Port of FastCRUD._as_single_response logic.

    Args:
        data: Raw data from database query result
        schema_to_select: Pydantic schema for model conversion
        return_as_model: Whether to convert to Pydantic model

    Returns:
        Formatted single record (dict or Pydantic model)

    Raises:
        ValueError: If schema_to_select required but not provided
    """
    if not return_as_model:
        return data

    if not schema_to_select:
        raise ValueError(
            "schema_to_select must be provided when return_as_model is True."
        )

    return schema_to_select(**data)


def format_multi_response(
    data: list[Any],
    schema_to_select: type | None = None,
    return_as_model: bool = False,
) -> list[Any]:
    """
    Format multiple records response with optional model conversion.

    Port of FastCRUD._as_multi_response logic.

    Args:
        data: List of raw data from database query results
        schema_to_select: Pydantic schema for model conversion
        return_as_model: Whether to convert to Pydantic models

    Returns:
        List of formatted records (dicts or Pydantic models)

    Raises:
        ValueError: If schema_to_select required but not provided
        ValidationError: If data validation fails during model conversion
    """
    if not return_as_model:
        return data

    if not schema_to_select:
        raise ValueError(
            "schema_to_select must be provided when return_as_model is True"
        )

    try:
        converted_data = []
        for row in data:
            if isinstance(row, dict):
                converted_data.append(schema_to_select(**row))
            else:
                converted_data.append(row)
        return converted_data
    except Exception as e:
        raise ValueError(
            f"Data validation error for schema {schema_to_select.__name__}: {e}"
        )


def create_paginated_response_data(
    items: list,
    total_count: int,
    offset: int = 0,
    limit: int | None = None,
    data_key: str = "data",
) -> dict[str, Any]:
    """
    Create paginated response data structure.

    Combines items with pagination metadata in a standardized format.

    Args:
        items: List of data items to include in response
        total_count: Total number of items available (for pagination)
        offset: Number of items skipped (default: 0)
        limit: Maximum number of items per page (default: None - no limit)
        data_key: Key name for the data items (default: "data")

    Returns:
        Dictionary containing items and pagination metadata

    Example:
        >>> create_paginated_response_data([item1, item2], 50, 20, 10)
        {
            "data": [item1, item2],
            "total_count": 50,
            "has_more": True,
            "offset": 20,
            "limit": 10
        }

    """
    response = {  # pragma: no cover
        data_key: items,
        "total_count": total_count,
    }

    if limit is not None:  # pragma: no cover
        response["has_more"] = (offset + len(items)) < total_count
        response["offset"] = offset
        response["limit"] = limit

    return response


def convert_to_pydantic_models(
    nested_data: list,
    schema_to_select: type[SelectSchemaType],
    nested_schema_to_select: dict[str, type[SelectSchemaType]] | None,
) -> list:
    """
    Converts nested dictionary data to Pydantic model instances.

    This function takes the nested dictionary data structure created by the join processing
    and converts it to properly typed Pydantic models. It handles both the main records
    and any nested joined data, applying the appropriate schemas to each level.

    Args:
        nested_data: List of dictionaries containing the nested data to be converted.
        schema_to_select: The main Pydantic schema class for the base records.
        nested_schema_to_select: Optional mapping of join prefixes to their corresponding schemas.

    Returns:
        List of Pydantic model instances with properly nested related data.

    Example:
        >>> nested_data = [
        ...     {
        ...         "id": 1,
        ...         "name": "Author 1",
        ...         "articles": [{"id": 10, "title": "Article 1"}]
        ...     }
        ... ]
        >>> schemas = {"articles_": ArticleSchema}
        >>> result = convert_to_pydantic_models(nested_data, AuthorSchema, schemas)
        >>> # Returns [AuthorSchema(id=1, name="Author 1", articles=[ArticleSchema(...)])]
    """
    converted_data = []
    for item in nested_data:
        if nested_schema_to_select:
            for prefix, nested_schema in nested_schema_to_select.items():
                prefix_key = prefix.rstrip("_")
                if prefix_key in item:
                    if isinstance(item[prefix_key], list):
                        item[prefix_key] = [
                            nested_schema(**nested_item)
                            for nested_item in item[prefix_key]
                        ]
                    else:
                        item[prefix_key] = (
                            nested_schema(**item[prefix_key])
                            if item[prefix_key] is not None
                            else None
                        )

        converted_data.append(schema_to_select(**item))
    return converted_data
