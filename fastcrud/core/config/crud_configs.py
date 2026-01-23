"""
Configuration classes for CRUD operations in FastCRUD.

This module defines configuration classes for customizing create, update, delete,
and filter operations, including automatic field injection and endpoint customization.
"""

from typing import Any, Callable, Sequence, Annotated
from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator
from fastapi import Depends, Query

from ...types import ModelType


class CRUDMethods(BaseModel):
    """
    Configuration for valid CRUD methods in endpoint creation.

    Attributes:
        valid_methods: Sequence of allowed CRUD method names.

    Example:
        >>> methods = CRUDMethods(valid_methods=["create", "read", "update"])
        >>> # Only create, read, and update endpoints will be generated
    """

    valid_methods: Annotated[
        Sequence[str],
        Field(
            default=[
                "create",
                "read",
                "read_multi",
                "update",
                "delete",
                "db_delete",
            ]
        ),
    ]

    @field_validator("valid_methods")
    def check_valid_method(cls, values: Sequence[str]) -> Sequence[str]:
        """Validate that all specified methods are valid CRUD methods."""
        valid_methods = {
            "create",
            "read",
            "read_multi",
            "update",
            "delete",
            "db_delete",
        }

        for v in values:
            if v not in valid_methods:
                raise ValueError(f"Invalid CRUD method: {v}")

        return values


class CreateConfig(BaseModel):
    """
    Configuration for create operations with automatic field injection.

    Allows you to automatically inject fields before data is written to the database.
    Perfect for:
    - Adding user_id from authentication context
    - Setting timestamps (created_at)
    - Adding audit fields (created_by)
    - Preventing clients from setting sensitive fields

    Attributes:
        auto_fields: Dictionary mapping field names to callables that provide values.
                     The callables will be invoked (with dependency injection if needed)
                     and their return values will be injected into the data.
        exclude_from_schema: List of field names to exclude from the request schema.
                            These fields won't appear in API documentation.

    Examples:
        Inject user_id and timestamps:
        ```python
        from datetime import datetime
        from fastapi import Depends, Cookie
        from fastcrud import crud_router, CreateConfig

        # Functions that return values (can use Depends for DI)
        async def get_current_user_id(session_token: str = Cookie(None)):
            user = await verify_token(session_token)
            return user.id

        def get_current_timestamp():
            return datetime.utcnow()

        create_config = CreateConfig(
            auto_fields={
                "user_id": get_current_user_id,      # Injected from cookie
                "created_by": get_current_user_id,   # Same user
                "created_at": get_current_timestamp, # Timestamp
            },
            exclude_from_schema=["user_id", "created_by", "created_at"]
        )

        router = crud_router(
            session=get_db,
            model=Item,
            create_schema=CreateItemSchema,  # Does NOT include auto fields
            update_schema=UpdateItemSchema,
            create_config=create_config,
        )
        ```
    """

    auto_fields: Annotated[dict[str, Callable[..., Any]], Field(default_factory=dict)]
    exclude_from_schema: Annotated[list[str], Field(default_factory=list)]

    @field_validator("auto_fields")
    @classmethod
    def check_auto_fields(
        cls, auto_fields: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., Any]]:
        """Validate that all auto_fields values are callable."""
        for key, value in auto_fields.items():
            if not callable(value):  # pragma: no cover
                raise ValueError(
                    f"auto_fields['{key}'] must be callable, got {type(value).__name__}"
                )
        return auto_fields


class UpdateConfig(BaseModel):
    """
    Configuration for update operations with automatic field injection.

    Allows you to automatically inject fields before data is written to the database.
    Perfect for:
    - Adding updated_by from authentication context
    - Setting timestamps (updated_at)
    - Preventing clients from modifying sensitive fields

    Attributes:
        auto_fields: Dictionary mapping field names to callables that provide values.
                     The callables will be invoked (with dependency injection if needed)
                     and their return values will be injected into the data.
        exclude_from_schema: List of field names to exclude from the request schema.
                            These fields won't appear in API documentation.

    Examples:
        Inject updated_by and timestamps:
        ```python
        from datetime import datetime
        from fastapi import Depends, Cookie
        from fastcrud import crud_router, UpdateConfig

        # Functions that return values (can use Depends for DI)
        async def get_current_user_id(session_token: str = Cookie(None)):
            user = await verify_token(session_token)
            return user.id

        def get_current_timestamp():
            return datetime.utcnow()

        update_config = UpdateConfig(
            auto_fields={
                "updated_by": get_current_user_id,
                "updated_at": get_current_timestamp,
            },
            exclude_from_schema=["updated_by", "updated_at", "user_id"]
        )

        router = crud_router(
            session=get_db,
            model=Item,
            create_schema=CreateItemSchema,
            update_schema=UpdateItemSchema,
            update_config=update_config,
        )
        ```
    """

    auto_fields: Annotated[dict[str, Callable[..., Any]], Field(default_factory=dict)]
    exclude_from_schema: Annotated[list[str], Field(default_factory=list)]

    @field_validator("auto_fields")
    @classmethod
    def check_auto_fields(
        cls, auto_fields: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., Any]]:
        """Validate that all auto_fields values are callable."""
        for key, value in auto_fields.items():
            if not callable(value):  # pragma: no cover
                raise ValueError(
                    f"auto_fields['{key}'] must be callable, got {type(value).__name__}"
                )
        return auto_fields


class DeleteConfig(BaseModel):
    """
    Configuration for delete operations with automatic field injection.

    Allows you to automatically inject fields before an item is soft-deleted.
    Perfect for:
    - Adding deleted_by from authentication context
    - Setting timestamps (deleted_at)
    - Adding audit fields for compliance
    - Soft delete tracking

    Attributes:
        auto_fields: Dictionary mapping field names to callables that provide values.
                     The callables will be invoked (with dependency injection if needed)
                     and their return values will be injected into the soft delete data.

    Examples:
        Inject deleted_by and timestamps for soft deletes:
        ```python
        from datetime import datetime
        from fastapi import Depends, Cookie
        from fastcrud import crud_router, DeleteConfig

        # Functions that return values (can use Depends for DI)
        async def get_current_user_id(session_token: str = Cookie(None)):
            user = await verify_token(session_token)
            return user.id

        def get_current_timestamp():
            return datetime.utcnow()

        delete_config = DeleteConfig(
            auto_fields={
                "deleted_by": get_current_user_id,
                "deleted_at": get_current_timestamp,
            }
        )

        router = crud_router(
            session=get_db,
            model=Item,
            create_schema=CreateItemSchema,
            update_schema=UpdateItemSchema,
            delete_config=delete_config,
        )
        ```

        Authorization check before deletion:
        ```python
        async def check_can_delete(
            session_token: str = Cookie(None),
            item_id: int = Path(...)
        ):
            user = await verify_token(session_token)
            if not user.can_delete:
                raise HTTPException(403, "Not authorized to delete")
            return user.id

        delete_config = DeleteConfig(
            auto_fields={
                "deleted_by": check_can_delete,
            }
        )
        ```
    """

    auto_fields: Annotated[dict[str, Callable[..., Any]], Field(default_factory=dict)]

    @field_validator("auto_fields")
    @classmethod
    def check_auto_fields(
        cls, auto_fields: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., Any]]:
        """Validate that all auto_fields values are callable."""
        for key, value in auto_fields.items():
            if not callable(value):  # pragma: no cover
                raise ValueError(
                    f"auto_fields['{key}'] must be callable, got {type(value).__name__}"
                )
        return auto_fields


class FilterConfig(BaseModel):
    """
    Configuration for query filtering in FastCRUD endpoints.

    This class allows you to define available query filters for endpoints,
    including support for joined model filtering using dot notation.

    Attributes:
        filters: Dictionary mapping filter names to their default values or dependency functions.

    Example:
        Basic filters:
        ```python
        filter_config = FilterConfig(
            name=None,              # ?name=john
            age__gte=None,          # ?age__gte=18
            is_active=True          # ?is_active=false (default: true)
        )
        ```

        Joined model filters:
        ```python
        filter_config = FilterConfig(
            **{
                "user.company.name": None,           # ?user.company.name=acme
                "user.company.name__contains": None, # ?user.company.name__contains=corp
            }
        )
        ```

        Dependency-based filters:
        ```python
        def get_current_user_company():
            return "current_user_company_id"

        filter_config = FilterConfig(
            company_id=get_current_user_company  # Automatically injected
        )
        ```
    """

    filters: Annotated[dict[str, Any], Field(default={})]

    @field_validator("filters")
    def check_filter_types(cls, filters: dict[str, Any]) -> dict[str, Any]:
        """Validate that filter values are of acceptable types."""
        for key, value in filters.items():
            if not (
                isinstance(value, (type(None), str, int, float, bool))
                or callable(value)
            ):
                raise ValueError(f"Invalid default value for '{key}': {value}")
        return filters

    def __init__(self, **kwargs: Any) -> None:
        """Initialize FilterConfig with filters from kwargs."""
        filters = kwargs.pop("filters", {})
        filters.update(kwargs)
        super().__init__(filters=filters)

    def get_params(self) -> dict[str, Any]:
        """
        Get FastAPI parameter definitions for the configured filters.

        Returns:
            Dictionary mapping parameter names to FastAPI parameter objects.
        """
        params = {}
        for key, value in self.filters.items():
            if callable(value):
                params[key] = Depends(value)
            else:
                params[key] = Query(value)
        return params

    def is_joined_filter(self, filter_key: str) -> bool:
        """
        Check if a filter key represents a joined model filter (contains dot notation).

        Args:
            filter_key: The filter key to check.

        Returns:
            True if the filter is for a joined model, False otherwise.

        Example:
            >>> config = FilterConfig()
            >>> config.is_joined_filter("user.company.name")
            True
            >>> config.is_joined_filter("name")
            False
        """
        field_path = filter_key.split("__")[0] if "__" in filter_key else filter_key
        return "." in field_path

    def parse_joined_filter(self, filter_key: str) -> tuple[list[str], str, str | None]:
        """
        Parse a joined filter key into its components.

        Args:
            filter_key: Filter key like "user.company.name" or "user.company.name__eq"

        Returns:
            tuple: (relationship_path, final_field, operator)
            e.g., (["user", "company"], "name", "eq") or (["user", "company"], "name", None)

        Example:
            >>> config = FilterConfig()
            >>> config.parse_joined_filter("user.company.name__contains")
            (['user', 'company'], 'name', 'contains')
        """
        if "__" in filter_key:
            field_path, operator = filter_key.rsplit("__", 1)
        else:
            field_path, operator = filter_key, None

        path_parts = field_path.split(".")
        if len(path_parts) < 2:
            raise ValueError(f"Invalid joined filter format: {filter_key}")

        relationship_path = path_parts[:-1]
        final_field = path_parts[-1]

        return relationship_path, final_field, operator


def validate_joined_filter_path(
    model: ModelType, relationship_path: list[str], final_field: str
) -> bool:
    """
    Validate that a joined filter path exists in the model relationships.

    Args:
        model: The base SQLAlchemy model
        relationship_path: List of relationship names to traverse (e.g., ["user", "company"])
        final_field: The final field name to filter on

    Returns:
        bool: True if the path is valid, False otherwise

    Example:
        >>> # For User -> Company -> Address relationship
        >>> validate_joined_filter_path(Task, ["user", "company"], "name")
        True
        >>> validate_joined_filter_path(Task, ["user", "invalid"], "name")
        False
    """
    from sqlalchemy import inspect as sa_inspect

    current_model = model

    for relationship_name in relationship_path:
        inspector = sa_inspect(current_model)
        if inspector is None:  # pragma: no cover
            return False

        if not hasattr(inspector, "relationships"):  # pragma: no cover
            return False

        relationship = inspector.relationships.get(relationship_name)
        if relationship is None:  # pragma: no cover
            return False

        current_model = relationship.mapper.class_

    final_inspector = sa_inspect(current_model)
    if final_inspector is None:  # pragma: no cover
        return False

    return (
        hasattr(current_model, final_field)
        and hasattr(final_inspector.mapper, "columns")
        and final_field in [col.name for col in final_inspector.mapper.columns]
    )
