"""
Configuration classes for join operations in FastCRUD.

This module defines the configuration classes used for specifying join relationships
and count operations in multi-table queries.
"""

from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import field_validator
from sqlalchemy.orm.util import AliasedClass


class JoinConfig(BaseModel):
    """
    Configuration for join operations in FastCRUD queries.

    This class defines how tables should be joined in multi-table queries,
    including the relationship type, join conditions, and optional sorting/pagination.

    Attributes:
        model: The SQLAlchemy model to join with.
        join_on: The join condition (SQLAlchemy expression).
        join_prefix: Optional prefix for joined columns in results.
        schema_to_select: Optional Pydantic schema for the joined model.
        join_type: Type of SQL join ("left" or "inner").
        alias: Optional SQLAlchemy alias for the joined model.
        filters: Optional filters to apply to the joined model.
        relationship_type: Type of relationship ("one-to-one" or "one-to-many").
        sort_columns: Optional column(s) to sort joined results by.
        sort_orders: Optional sort order(s) for the sort columns.
        nested_limit: Optional limit for nested items in one-to-many relationships.
            When set, only the first N nested items are returned (after sorting).
            Use None for no limit (default).

    Example:
        >>> join_config = JoinConfig(
        ...     model=Article,
        ...     join_on=Article.author_id == Author.id,
        ...     join_prefix="articles_",
        ...     relationship_type="one-to-many",
        ...     sort_columns=["created_at", "title"],
        ...     sort_orders=["desc", "asc"],
        ...     nested_limit=10  # Only return first 10 articles per author
        ... )
    """

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
    nested_limit: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("relationship_type")
    def check_valid_relationship_type(cls, value):
        """Validate that relationship_type is valid."""
        valid_relationship_types = {"one-to-one", "one-to-many"}
        if value is not None and value not in valid_relationship_types:
            raise ValueError(f"Invalid relationship type: {value}")
        return value

    @field_validator("join_type")
    def check_valid_join_type(cls, value):
        """Validate that join_type is valid."""
        valid_join_types = {"left", "inner"}
        if value not in valid_join_types:
            raise ValueError(f"Unsupported join type: {value}")
        return value


class CountConfig(BaseModel):
    """
    Configuration for counting related objects in joined queries.

    This allows you to annotate query results with counts of related objects,
    particularly useful for many-to-many relationships. The count is implemented
    as a scalar subquery, which means all records from the primary model will be
    returned with their respective counts (including 0 for records with no related objects).

    Attributes:
        model: The SQLAlchemy model to count.
        join_on: The join condition for the count query.
        alias: Optional alias for the count column in the result. Defaults to "{model.__tablename__}_count".
        filters: Optional filters to apply to the count query.

    Example:
        ```python
        from fastcrud import FastCRUD, CountConfig

        # Count videos for each search through a many-to-many relationship
        count_config = CountConfig(
            model=Video,
            join_on=(Video.id == VideoSearchAssociation.video_id)
                   & (VideoSearchAssociation.search_id == Search.id),
            alias='videos_count'
        )

        search_crud = FastCRUD(Search)
        results = await search_crud.get_multi_joined(
            db=session,
            counts_config=[count_config],
        )
        # Results will include 'videos_count' field for each search
        # Example: [{"id": 1, "term": "cats", "videos_count": 5}, ...]
        ```
    """

    model: Any
    join_on: Any
    alias: Optional[str] = None
    filters: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
