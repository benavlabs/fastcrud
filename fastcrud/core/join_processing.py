"""
Complex join processing with stateful caching and multi-join operations.

This module provides classes and functions for handling complex multi-join scenarios
with proper caching of model introspection results and composite primary key handling.
"""

from typing import Sequence, Any, TYPE_CHECKING
from pydantic import BaseModel

from .introspection import (
    get_model_inspector,
    get_first_primary_key,
    create_composite_key,
)
from .data import (
    sort_nested_list,
    convert_to_pydantic_models,
)
from ..types import ModelType, SelectSchemaType

if TYPE_CHECKING:  # pragma: no cover
    from .config import JoinConfig


class JoinProcessor:
    """
    Stateful processor for complex multi-join operations with caching.

    This class manages multiple model inspectors and provides efficient processing
    of multi-join scenarios by caching expensive model introspection operations.

    Attributes:
        base_model: The base SQLAlchemy model for joins.
        base_inspector: Cached inspector for the base model.

    Example:
        >>> processor = JoinProcessor(Author)
        >>> result = processor.process_multi_join(data, joins_config)
    """

    def __init__(self, base_model: ModelType):
        self.base_model = base_model
        self.base_inspector = get_model_inspector(base_model)
        self._join_inspectors: dict[ModelType, Any] = {}

    def get_join_inspector(self, model: ModelType):
        """
        Get cached inspector for join model.

        Args:
            model: The SQLAlchemy model to get inspector for.

        Returns:
            Cached ModelInspector instance for the model.
        """
        if model not in self._join_inspectors:
            self._join_inspectors[model] = get_model_inspector(model)
        return self._join_inspectors[model]

    def initialize_pre_nested_data(
        self,
        base_primary_key: str,
        data: Sequence[dict | BaseModel],
    ) -> dict:
        """
        Initializes a dictionary for organizing multi-record joined data by primary key.

        This function creates a base structure for handling multiple records with joined data,
        organizing them by their primary key values to facilitate deduplication and proper
        nesting of related data.

        Args:
            base_primary_key: The name of the primary key field for the base model.
            data: A sequence of records (dictionaries or Pydantic models) to be organized.

        Returns:
            A dictionary mapping primary key values to their corresponding record data.

        Example:
            >>> data = [
            ...     {"id": 1, "name": "Author 1", "articles": [{"title": "Article 1"}]},
            ...     {"id": 2, "name": "Author 2", "articles": [{"title": "Article 2"}]},
            ...     {"id": 1, "name": "Author 1", "articles": [{"title": "Article 3"}]}  # Duplicate
            ... ]
            >>> result = processor.initialize_pre_nested_data("id", data)
            >>> # Returns: {1: {"id": 1, "name": "Author 1", ...}, 2: {"id": 2, "name": "Author 2", ...}}
        """
        pre_nested_data = {}
        for row in data:
            if isinstance(row, BaseModel):
                new_row = row.model_dump()
            else:
                new_row = dict(row)

            primary_key_value = new_row[base_primary_key]
            if primary_key_value not in pre_nested_data:
                pre_nested_data[primary_key_value] = new_row

        return pre_nested_data

    def deduplicate_and_sort_join_items(
        self,
        existing_items: set,
        value: list,
        join_primary_key_names: list[str],
        join_config,
        target_list: list,
    ) -> None:
        """
        Deduplicates joined items using composite primary keys and applies sorting if configured.

        This function handles the deduplication of joined records in one-to-many relationships
        by creating composite keys from all primary key fields, ensuring that records with
        the same composite primary key are not duplicated. It also applies sorting based on
        the join configuration if specified.

        Args:
            existing_items: A set of composite keys representing items already processed.
            value: List of new items to be deduplicated and potentially added.
            join_primary_key_names: List of primary key field names for creating composite keys.
            join_config: The join configuration containing sorting configuration.
            target_list: The target list where deduplicated items will be added.

        Returns:
            None. The function modifies target_list in place.

        Example:
            >>> existing_items = {(1, 1), (1, 2)}
            >>> new_items = [{"id": 1, "version": 3, "name": "New"}, {"id": 1, "version": 1, "name": "Duplicate"}]
            >>> processor.deduplicate_and_sort_join_items(
            ...     existing_items, new_items, ["id", "version"], join_config, target_list
            ... )
            >>> # Only the item with (1, 3) composite key is added, (1, 1) is skipped as duplicate
        """
        for item in value:
            item_composite_key = create_composite_key(item, join_primary_key_names)
            if item_composite_key not in existing_items:
                target_list.append(item)
                existing_items.add(item_composite_key)

        if join_config.sort_columns and target_list:
            target_list[:] = sort_nested_list(
                target_list, join_config.sort_columns, join_config.sort_orders
            )

    def process_one_to_many_join(
        self,
        join_config,
        data: Sequence[dict | BaseModel],
        pre_nested_data: dict,
        base_primary_key: str,
        join_primary_key: str,
        join_primary_key_names: list[str],
        join_prefix: str,
    ) -> None:
        """
        Processes one-to-many join relationships with proper deduplication using composite primary keys.

        This function handles the complex logic of merging one-to-many joined data while ensuring
        that duplicate records (based on composite primary keys) are not included multiple times.
        It's specifically designed to handle the composite primary key deduplication bug where
        records with the same first primary key but different second primary keys were incorrectly
        considered duplicates.

        Args:
            join_config: The join configuration defining the join relationship and sorting options.
            data: Sequence of records containing the joined data to be processed.
            pre_nested_data: Dictionary mapping base record primary keys to their data.
            base_primary_key: The primary key field name of the base model.
            join_primary_key: The primary key field name of the joined model (first key only).
            join_primary_key_names: List of all primary key field names for composite key creation.
            join_prefix: The prefix used to identify fields belonging to this join.

        Returns:
            None. The function modifies pre_nested_data in place.

        Example:
            Processing a parent with children having composite PKs (child_id, version):
            >>> # Child 1: (child_id=1, version=1)
            >>> # Child 2: (child_id=1, version=2)  # Different version, should be separate
            >>> processor.process_one_to_many_join(...)
            >>> # Results in both children being included, not deduplicated incorrectly
        """
        for row in data:
            row_dict = row if isinstance(row, dict) else row.model_dump()
            primary_key_value = row_dict[base_primary_key]

            if join_prefix in row_dict:
                value = row_dict[join_prefix]
                if isinstance(value, list):
                    if any(item[join_primary_key] is None for item in value):
                        pre_nested_data[primary_key_value][join_prefix] = []
                    else:
                        existing_items = {
                            create_composite_key(item, join_primary_key_names)
                            for item in pre_nested_data[primary_key_value][join_prefix]
                        }
                        self.deduplicate_and_sort_join_items(
                            existing_items,
                            value,
                            join_primary_key_names,
                            join_config,
                            pre_nested_data[primary_key_value][join_prefix],
                        )

    def process_one_to_one_join(
        self,
        data: Sequence[dict | BaseModel],
        pre_nested_data: dict,
        base_primary_key: str,
        join_primary_key: str,
        join_prefix: str,
    ) -> None:
        """
        Processes one-to-one join relationships by merging related record data.

        This function handles the merging of one-to-one joined data, where each base record
        can have at most one related record. It properly handles cases where the related
        record might be null (indicated by a null primary key).

        Args:
            data: Sequence of records containing the joined data to be processed.
            pre_nested_data: Dictionary mapping base record primary keys to their data.
            base_primary_key: The primary key field name of the base model.
            join_primary_key: The primary key field name of the joined model.
            join_prefix: The prefix used to identify fields belonging to this join.

        Returns:
            None. The function modifies pre_nested_data in place.

        Example:
            Processing an author with their profile (one-to-one):
            >>> # Author 1 -> Profile {"id": 10, "bio": "Author bio"}
            >>> # Author 2 -> Profile {"id": None, "bio": None}  # No profile
            >>> processor.process_one_to_one_join(...)
            >>> # Author 1 gets profile data, Author 2 gets profile set to None
        """
        for row in data:
            row_dict = row if isinstance(row, dict) else row.model_dump()
            primary_key_value = row_dict[base_primary_key]

            if join_prefix in row_dict:
                value = row_dict[join_prefix]
                if isinstance(value, dict) and value.get(join_primary_key) is None:
                    pre_nested_data[primary_key_value][join_prefix] = None
                elif isinstance(value, dict):
                    pre_nested_data[primary_key_value][join_prefix] = value

    def validate_schema_compatibility(
        self,
        joins_config: Sequence,
        schema_to_select: type[SelectSchemaType],
    ) -> None:
        """
        Validates that join prefixes are compatible with the target Pydantic schema fields.

        This function ensures that when return_as_model=True is used, all join prefixes
        correspond to actual fields in the target Pydantic schema. This prevents runtime
        errors when trying to construct Pydantic models with unexpected nested keys.

        Args:
            joins_config: Sequence of join configuration instances with join prefix configurations.
            schema_to_select: The target Pydantic schema class for validation.

        Raises:
            ValueError: If any join prefix creates a key that doesn't exist in the schema.

        Example:
            >>> class AuthorSchema(BaseModel):
            ...     id: int
            ...     name: str
            ...     articles: List[ArticleSchema] = []
            ...
            >>> join_config = JoinConfig(model=Article, join_prefix="articles_")
            >>> processor.validate_schema_compatibility([join_config], AuthorSchema)
            >>> # Passes validation since "articles" field exists in AuthorSchema

            >>> bad_join_config = JoinConfig(model=Article, join_prefix="posts_")
            >>> processor.validate_schema_compatibility([bad_join_config], AuthorSchema)
            >>> # Raises ValueError: join_prefix 'posts_' creates key 'posts' which is not in schema
        """
        schema_fields = set(schema_to_select.model_fields.keys())
        for join_config in joins_config:
            if join_config.join_prefix:
                join_key = join_config.join_prefix.rstrip("_")
                if join_key not in schema_fields:
                    raise ValueError(
                        f"join_prefix '{join_config.join_prefix}' creates key '{join_key}' "
                        f"which is not a field in schema {schema_to_select.__name__}. "
                        f"Available fields: {sorted(schema_fields)}. "
                        f"Either change join_prefix to match a schema field or use return_as_model=False."
                    )

    def process_multi_join(
        self,
        data: Sequence[dict | BaseModel],
        joins_config: Sequence["JoinConfig"],
        return_as_model: bool = False,
        schema_to_select: type[SelectSchemaType] | None = None,
        nested_schema_to_select: dict[str, type[SelectSchemaType]] | None = None,
    ) -> Sequence[dict | SelectSchemaType]:
        """
        Nests joined data based on join definitions provided for multiple records. This function processes the input list of
        dictionaries, identifying keys that correspond to joined tables using the provided `joins_config`, and nests them
        under their respective table keys.

        Args:
            data: The list of dictionaries containing the records with potentially nested data.
            joins_config: The list of join configurations containing the joined model classes and related settings.
            return_as_model: If `True`, converts the fetched data to Pydantic models based on `schema_to_select`. Defaults to `False`.
            schema_to_select: Pydantic schema for selecting specific columns from the primary model. Used for converting
                              dictionaries back to Pydantic models.
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
        base_primary_key = self.base_inspector.first_primary_key
        pre_nested_data = self.initialize_pre_nested_data(base_primary_key, data)

        for join_config in joins_config:
            join_inspector = self.get_join_inspector(join_config.model)
            join_primary_key = join_inspector.first_primary_key
            join_primary_key_names = join_inspector.primary_key_names
            join_prefix = (
                join_config.join_prefix.rstrip("_")
                if join_config.join_prefix
                else join_config.model.__tablename__
            )

            if join_config.relationship_type == "one-to-many":
                self.process_one_to_many_join(
                    join_config,
                    data,
                    pre_nested_data,
                    base_primary_key,
                    join_primary_key,
                    join_primary_key_names,
                    join_prefix,
                )
            else:
                self.process_one_to_one_join(
                    data,
                    pre_nested_data,
                    base_primary_key,
                    join_primary_key,
                    join_prefix,
                )

        nested_data: list = list(pre_nested_data.values())

        if return_as_model:
            if not schema_to_select:
                raise ValueError(
                    "schema_to_select must be provided when return_as_model is True."
                )

            self.validate_schema_compatibility(joins_config, schema_to_select)
            return convert_to_pydantic_models(
                nested_data, schema_to_select, nested_schema_to_select
            )

        return nested_data


def handle_null_primary_key_multi_join(
    data: list[dict[str, Any] | SelectSchemaType],
    join_definitions: list,
) -> list[dict[str, Any] | SelectSchemaType]:
    """
    Handles null primary keys in multi-join results by cleaning up invalid nested data.

    This function post-processes multi-join results to handle cases where joined
    records have null primary keys, indicating no actual related data exists.

    Args:
        data: List of result records (dicts or Pydantic models).
        join_definitions: List of join configuration instances.

    Returns:
        Cleaned list with null primary key entries properly handled.
    """
    for item in data:
        item_dict = item if isinstance(item, dict) else item.model_dump()

        for join in join_definitions:
            join_prefix = join.join_prefix or ""
            nested_key = (
                join_prefix.rstrip("_") if join_prefix else join.model.__tablename__
            )

            if nested_key in item_dict and isinstance(item_dict[nested_key], dict):
                join_primary_key = get_first_primary_key(join.model)

                if join_primary_key:
                    if (
                        join_primary_key in item_dict[nested_key]
                        and item_dict[nested_key][join_primary_key] is None
                    ):
                        item_dict[nested_key] = None

        if isinstance(item, BaseModel):
            for key, value in item_dict.items():
                setattr(item, key, value)

    return data
