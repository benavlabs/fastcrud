from typing import Any, Optional, Union, Sequence, cast

from sqlalchemy import inspect
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql import ColumnElement
from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import field_validator

from fastcrud.types import ModelType, SelectSchemaType

from ..endpoint.helper import _get_primary_key


class JoinConfig(BaseModel):
    model: Any
    join_on: Any
    join_prefix: Optional[str] = None
    schema_to_select: Optional[type[BaseModel]] = None
    join_type: str = "left"
    alias: Optional[AliasedClass] = None
    filters: Optional[dict] = None
    relationship_type: Optional[str] = "one-to-one"
    sort_columns: Optional[Union[str, list[str]]] = None
    sort_orders: Optional[Union[str, list[str]]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("relationship_type")
    def check_valid_relationship_type(cls, value):
        valid_relationship_types = {"one-to-one", "one-to-many"}
        if value is not None and value not in valid_relationship_types:
            raise ValueError(f"Invalid relationship type: {value}")  # pragma: no cover
        return value

    @field_validator("join_type")
    def check_valid_join_type(cls, value):
        valid_join_types = {"left", "inner"}
        if value not in valid_join_types:
            raise ValueError(f"Unsupported join type: {value}")
        return value

    @field_validator("sort_columns")
    def check_valid_sort_columns(cls, value):
        if value is not None and not isinstance(value, (str, list)):
            raise ValueError("sort_columns must be a string or a list of strings")
        return value

    @field_validator("sort_orders")
    def check_valid_sort_orders(cls, value):
        if value is not None:
            if isinstance(value, str):
                if value not in ["asc", "desc"]:
                    raise ValueError("Invalid sort order: {value}. Only 'asc' or 'desc' are allowed.")
            elif isinstance(value, list):
                for order in value:
                    if order not in ["asc", "desc"]:
                        raise ValueError("Invalid sort order: {order}. Only 'asc' or 'desc' are allowed.")
            else:
                raise ValueError("sort_orders must be a string or a list of strings")
        return value


def _extract_matching_columns_from_schema(
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
    if not hasattr(model, "__table__"):  # pragma: no cover
        raise AttributeError(f"{model.__name__} does not have a '__table__' attribute.")

    model_or_alias = alias if alias else model
    columns = []
    temp_prefix = (
        temp_prefix if use_temporary_prefix and temp_prefix is not None else ""
    )

    mapper = inspect(model).mapper

    if schema:
        for field in schema.model_fields.keys():
            if hasattr(model_or_alias, field):
                column = getattr(model_or_alias, field)
                if prefix is not None or use_temporary_prefix:
                    column_label = (
                        f"{temp_prefix}{prefix}{field}"
                        if prefix
                        else f"{temp_prefix}{field}"
                    )
                    column = column.label(column_label)
                columns.append(column)
    else:
        for prop in mapper.column_attrs:
            column = getattr(model_or_alias, prop.key)
            if prefix is not None or use_temporary_prefix:
                column_label = (
                    f"{temp_prefix}{prefix}{prop.key}"
                    if prefix
                    else f"{temp_prefix}{prop.key}"
                )
                column = column.label(column_label)
            columns.append(column)

    return columns


def _auto_detect_join_condition(
    base_model: ModelType,
    join_model: ModelType,
) -> Optional[ColumnElement]:
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
    if not hasattr(base_model, "__table__"):  # pragma: no cover
        raise AttributeError(
            f"{base_model.__name__} does not have a '__table__' attribute."
        )
    if not hasattr(join_model, "__table__"):  # pragma: no cover
        raise AttributeError(
            f"{join_model.__name__} does not have a '__table__' attribute."
        )

    inspector = inspect(base_model)
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

        if join_on is None:  # pragma: no cover
            raise ValueError(
                "Could not automatically determine join condition. Please provide join_on."
            )
    else:  # pragma: no cover
        raise ValueError("Could not automatically get model columns.")

    return join_on


def _handle_one_to_one(nested_data, nested_key, nested_field, value):
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


def _handle_one_to_many(nested_data, nested_key, nested_field, value):
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


def _sort_nested_list(nested_list: list[dict], sort_columns: Union[str, list[str]], sort_orders: Optional[Union[str, list[str]]] = None) -> list[dict]:
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
        >>> _sort_nested_list([{"id": 2, "name": "B"}, {"id": 1, "name": "A"}], "name")
        [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]

        Sorting by multiple columns with different orders:
        >>> _sort_nested_list([{"id": 1, "name": "A"}, {"id": 2, "name": "A"}], ["name", "id"], ["asc", "desc"])
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
                raise ValueError(f"Invalid sort order: {order}. Only 'asc' or 'desc' are allowed.")
    else:
        sort_orders = ["asc"] * len(sort_columns)

    # Create a list of (column, order) tuples for sorting
    sort_specs = [(col, 1 if order == "asc" else -1) for col, order in zip(sort_columns, sort_orders)]

    # Sort the list using the sort_specs
    sorted_list = nested_list.copy()
    for col, direction in reversed(sort_specs):
        sorted_list.sort(key=lambda x: (x.get(col) is None, x.get(col)), reverse=direction == -1)

    return sorted_list


def _nest_join_data(
    data: dict,
    join_definitions: list[JoinConfig],
    temp_prefix: str = "joined__",
    nested_data: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Nests joined data based on join definitions provided. This function processes the input `data` dictionary, identifying keys
    that correspond to joined tables using the provided `join_definitions` and nest them under their respective table keys.

    Args:
        data: The flat dictionary containing data with potentially prefixed keys from joined tables.
        join_definitions: A list of `JoinConfig` instances defining the join configurations, including prefixes.
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

    for key, value in data.items():
        nested = False
        for join in join_definitions:
            join_prefix = join.join_prefix or ""
            full_prefix = f"{temp_prefix}{join_prefix}"

            if isinstance(key, str) and key.startswith(full_prefix):
                nested_key = (
                    join_prefix.rstrip("_") if join_prefix else join.model.__tablename__
                )
                nested_field = key[len(full_prefix) :]

                if join.relationship_type == "one-to-many":
                    nested_data = _handle_one_to_many(
                        nested_data, nested_key, nested_field, value
                    )
                else:
                    nested_data = _handle_one_to_one(
                        nested_data, nested_key, nested_field, value
                    )

                nested = True
                break

        if not nested:
            stripped_key = (
                key[len(temp_prefix) :]
                if isinstance(key, str) and key.startswith(temp_prefix)
                else key
            )
            if nested_data is None:  # pragma: no cover
                nested_data = {}

            nested_data[stripped_key] = value

    if nested_data is None:  # pragma: no cover
        nested_data = {}

    for join in join_definitions:
        join_primary_key = _get_primary_key(join.model)
        nested_key = (
            join.join_prefix.rstrip("_")
            if join.join_prefix
            else join.model.__tablename__
        )
        if join.relationship_type == "one-to-many" and nested_key in nested_data:
            if isinstance(nested_data.get(nested_key, []), list):
                if any(
                    item[join_primary_key] is None for item in nested_data[nested_key]
                ):
                    nested_data[nested_key] = []
                # Apply sorting to nested list if sort_columns is specified
                elif join.sort_columns and nested_data[nested_key]:
                    nested_data[nested_key] = _sort_nested_list(
                        nested_data[nested_key], join.sort_columns, join.sort_orders
                    )

        if nested_key in nested_data and isinstance(nested_data[nested_key], dict):
            if (
                join_primary_key in nested_data[nested_key]
                and nested_data[nested_key][join_primary_key] is None
            ):
                nested_data[nested_key] = None

    assert nested_data is not None, "Couldn't nest the data."
    return nested_data


def _nest_multi_join_data(
    base_primary_key: str,
    data: Sequence[Union[dict, BaseModel]],
    joins_config: Sequence[JoinConfig],
    return_as_model: bool = False,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    nested_schema_to_select: Optional[dict[str, type[SelectSchemaType]]] = None,
) -> Sequence[Union[dict, SelectSchemaType]]:
    """
    Nests joined data based on join definitions provided for multiple records. This function processes the input list of
    dictionaries, identifying keys that correspond to joined tables using the provided `joins_config`, and nests them
    under their respective table keys.

    Args:
        base_primary_key: The primary key of the base model.
        data: The list of dictionaries containing the records with potentially nested data.
        joins_config: The list of join configurations containing the joined model classes and related settings.
        schema_to_select: Pydantic schema for selecting specific columns from the primary model. Used for converting
                          dictionaries back to Pydantic models.
        return_as_model: If `True`, converts the fetched data to Pydantic models based on `schema_to_select`. Defaults to `False`.
        nested_schema_to_select: A dictionary mapping join prefixes to their corresponding Pydantic schemas.

    Returns:
        Sequence[Union[dict, SelectSchemaType]]: A list of dictionaries with nested structures for joined table data or Pydantic models.

    Example:

        Input:

        ```python
        data = [
            {'id': 1, 'title': 'Test Author', 'articles': [{'id': 1, 'title': 'Article 1', 'author_id': 1}]},
            {'id': 2, 'title': 'Test Author 2', 'articles': [{'id': 2, 'title': 'Article 2', 'author_id': 2}]},
            {'id': 2, 'title': 'Test Author 2', 'articles': [{'id': 3, 'title': 'Article 3', 'author_id': 2}]},
            {'id': 3, 'title': 'Test Author 3', 'articles': [{'id': None, 'title': None, 'author_id': None}]},
        ]

        joins_config = [
            JoinConfig(model=Article, join_prefix='articles_', relationship_type='one-to-many')
        ]
        ```

        Output:

        ```json
        [
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
            },
            {
                'id': 2,
                'title': 'Test Author 2',
                'articles': [
                    {
                        'id': 2,
                        'title': 'Article 2',
                        'author_id': 2
                    },
                    {
                        'id': 3,
                        'title': 'Article 3',
                        'author_id': 2
                    }
                ]
            },
            {
                'id': 3,
                'title': 'Test Author 3',
                'articles': []
            }
        ]
        ```
    """
    pre_nested_data = {}

    for row in data:
        if isinstance(row, BaseModel):
            new_row = {
                key: ([] if isinstance(value, list) else value)
                for key, value in row.model_dump().items()
            }
        else:
            new_row = {
                key: ([] if isinstance(value, list) else value)
                for key, value in row.items()
            }

        primary_key_value = new_row[base_primary_key]
        if primary_key_value not in pre_nested_data:
            pre_nested_data[primary_key_value] = new_row

    for join_config in joins_config:
        join_primary_key = _get_primary_key(join_config.model)
        join_prefix = (
            join_config.join_prefix.rstrip("_")
            if join_config.join_prefix
            else join_config.model.__tablename__
        )

        for row in data:
            row_dict = row if isinstance(row, dict) else row.model_dump()
            primary_key_value = row_dict[base_primary_key]

            if join_config.relationship_type == "one-to-many":
                if join_prefix in row_dict:
                    value = row_dict[join_prefix]
                    if isinstance(value, list):
                        if any(
                            item[join_primary_key] is None for item in value
                        ):  # pragma: no cover
                            pre_nested_data[primary_key_value][join_prefix] = []
                        else:
                            existing_items = {
                                item[join_primary_key]
                                for item in pre_nested_data[primary_key_value][
                                    join_prefix
                                ]
                            }
                            for item in value:
                                if item[join_primary_key] not in existing_items:
                                    pre_nested_data[primary_key_value][
                                        join_prefix
                                    ].append(item)
                                    existing_items.add(item[join_primary_key])

                            # Apply sorting to nested list if sort_columns is specified
                            if join_config.sort_columns and pre_nested_data[primary_key_value][join_prefix]:
                                pre_nested_data[primary_key_value][join_prefix] = _sort_nested_list(
                                    pre_nested_data[primary_key_value][join_prefix],
                                    join_config.sort_columns,
                                    join_config.sort_orders
                                )
            else:  # pragma: no cover
                if join_prefix in row_dict:
                    value = row_dict[join_prefix]
                    if isinstance(value, dict) and value.get(join_primary_key) is None:
                        pre_nested_data[primary_key_value][join_prefix] = None
                    elif isinstance(value, dict):
                        pre_nested_data[primary_key_value][join_prefix] = value

    nested_data: list = list(pre_nested_data.values())

    if return_as_model:
        if not schema_to_select:  # pragma: no cover
            raise ValueError(
                "schema_to_select must be provided when return_as_model is True."
            )

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
                        else:  # pragma: no cover
                            item[prefix_key] = (
                                nested_schema(**item[prefix_key])
                                if item[prefix_key] is not None
                                else None
                            )

            converted_data.append(schema_to_select(**item))
        return converted_data

    return nested_data


def _handle_null_primary_key_multi_join(
    data: list[Union[dict[str, Any], SelectSchemaType]],
    join_definitions: list[JoinConfig],
) -> list[Union[dict[str, Any], SelectSchemaType]]:
    for item in data:
        item_dict = item if isinstance(item, dict) else item.model_dump()

        for join in join_definitions:
            join_prefix = join.join_prefix or ""
            nested_key = (
                join_prefix.rstrip("_") if join_prefix else join.model.__tablename__
            )

            if nested_key in item_dict and isinstance(item_dict[nested_key], dict):
                join_primary_key = _get_primary_key(join.model)

                primary_key = join_primary_key
                if join_primary_key:
                    if (
                        primary_key in item_dict[nested_key]
                        and item_dict[nested_key][primary_key] is None
                    ):  # pragma: no cover
                        item_dict[nested_key] = None

        if isinstance(item, BaseModel):
            for key, value in item_dict.items():
                setattr(item, key, value)

    return data
