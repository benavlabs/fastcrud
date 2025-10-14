import inspect
from uuid import UUID
from typing import Optional, Union, Annotated, Sequence, Callable, TypeVar, Any

from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator
from fastapi import Depends, Query, Path, params

from sqlalchemy import Column, inspect as sa_inspect
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.types import TypeEngine
from sqlalchemy.sql.elements import KeyedColumnElement

from fastcrud.types import ModelType

F = TypeVar("F", bound=Callable[..., Any])


class CRUDMethods(BaseModel):
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
        for key, value in auto_fields.items():
            if not callable(value):
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
        for key, value in auto_fields.items():
            if not callable(value):
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
        for key, value in auto_fields.items():
            if not callable(value):
                raise ValueError(
                    f"auto_fields['{key}'] must be callable, got {type(value).__name__}"
                )
        return auto_fields


class FilterConfig(BaseModel):
    filters: Annotated[dict[str, Any], Field(default={})]

    @field_validator("filters")
    def check_filter_types(cls, filters: dict[str, Any]) -> dict[str, Any]:
        for key, value in filters.items():
            if not (
                isinstance(value, (type(None), str, int, float, bool))
                or callable(value)
            ):
                raise ValueError(f"Invalid default value for '{key}': {value}")
        return filters

    def __init__(self, **kwargs: Any) -> None:
        filters = kwargs.pop("filters", {})
        filters.update(kwargs)
        super().__init__(filters=filters)

    def get_params(self) -> dict[str, Any]:
        params = {}
        for key, value in self.filters.items():
            if callable(value):
                params[key] = Depends(value)
            else:
                params[key] = Query(value)
        return params

    def is_joined_filter(self, filter_key: str) -> bool:
        """Check if a filter key represents a joined model filter (contains dot notation)."""
        field_path = filter_key.split("__")[0] if "__" in filter_key else filter_key
        return "." in field_path

    def parse_joined_filter(
        self, filter_key: str
    ) -> tuple[list[str], str, Optional[str]]:
        """
        Parse a joined filter key into its components.

        Args:
            filter_key: Filter key like "user.company.name" or "user.company.name__eq"

        Returns:
            tuple: (relationship_path, final_field, operator)
            e.g., (["user", "company"], "name", "eq") or (["user", "company"], "name", None)
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


def _validate_joined_filter_path(
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
    """
    current_model = model

    for relationship_name in relationship_path:
        inspector = sa_inspect(current_model)
        if inspector is None:
            return False

        if not hasattr(inspector, "relationships"):
            return False

        relationship = inspector.relationships.get(relationship_name)
        if relationship is None:
            return False

        current_model = relationship.mapper.class_

    final_inspector = sa_inspect(current_model)
    if final_inspector is None:
        return False

    return (
        hasattr(current_model, final_field)
        and hasattr(final_inspector.mapper, "columns")
        and final_field in [col.name for col in final_inspector.mapper.columns]
    )


def _get_primary_key(
    model: ModelType,
) -> Union[str, None]:  # pragma: no cover
    key: Optional[str] = _get_primary_keys(model)[0].name
    return key


def _get_primary_keys(
    model: ModelType,
) -> Sequence[Column]:
    """Get the primary key of a SQLAlchemy model."""
    inspector_result = sa_inspect(model)
    if inspector_result is None:  # pragma: no cover
        raise ValueError("Model inspection failed, resulting in None.")
    primary_key_columns: Sequence[Column] = inspector_result.mapper.primary_key

    return primary_key_columns


def _is_uuid_type(column_type: TypeEngine) -> bool:  # pragma: no cover
    """
    Check if a SQLAlchemy column type represents a UUID.
    Handles various SQL dialects and common UUID implementations.
    """
    if isinstance(column_type, PostgresUUID):
        return True

    type_name = getattr(column_type, "__visit_name__", "").lower()
    if "uuid" in type_name:
        return True

    if hasattr(column_type, "impl"):
        return _is_uuid_type(column_type.impl)

    return False


def _get_python_type(column: Column) -> Optional[type]:
    """Get the Python type for a SQLAlchemy column, with special handling for UUIDs."""
    try:
        if _is_uuid_type(column.type):
            return UUID

        direct_type: Optional[type] = column.type.python_type
        return direct_type
    except NotImplementedError:
        if hasattr(column.type, "impl") and hasattr(column.type.impl, "python_type"):
            if _is_uuid_type(column.type.impl):  # pragma: no cover
                return UUID
            indirect_type: Optional[type] = column.type.impl.python_type
            return indirect_type
        else:  # pragma: no cover
            raise NotImplementedError(
                f"The primary key column {column.name} uses a custom type without a defined `python_type` or suitable `impl` fallback."
            )


def _get_column_types(
    model: ModelType,
) -> dict[str, Union[type, None]]:
    """Get a dictionary of column names and their corresponding Python types from a SQLAlchemy model."""
    inspector_result = sa_inspect(model)
    if inspector_result is None or inspector_result.mapper is None:  # pragma: no cover
        raise ValueError("Model inspection failed, resulting in None.")
    column_types = {}
    for column in inspector_result.mapper.columns:
        column_type = _get_python_type(column)
        if (
            hasattr(column.type, "__visit_name__")
            and column.type.__visit_name__ == "uuid"
        ):
            column_type = UUID
        column_types[column.name] = column_type
    return column_types


def _extract_unique_columns(
    model: ModelType,
) -> Sequence[KeyedColumnElement]:
    """Extracts columns from a SQLAlchemy model that are marked as unique."""
    if not hasattr(model, "__table__"):  # pragma: no cover
        raise AttributeError(f"{model.__name__} does not have a '__table__' attribute.")
    unique_columns = [column for column in model.__table__.columns if column.unique]
    return unique_columns


def _inject_dependencies(
    funcs: Optional[Sequence[Callable]] = None,
) -> Optional[Sequence[params.Depends]]:
    """Wraps a list of functions in FastAPI's Depends."""
    if funcs is None:
        return None

    for func in funcs:
        if not callable(func):
            raise TypeError(
                f"All dependencies must be callable. Got {type(func)} instead."
            )

    return [Depends(func) for func in funcs]


def _apply_model_pk(**pkeys: dict[str, type]):
    """
    This decorator injects positional arguments into a fastCRUD endpoint.
    It dynamically changes the endpoint signature and allows to use
    multiple primary keys without defining them explicitly.
    """

    def wrapper(endpoint):
        signature = inspect.signature(endpoint)
        parameters = [
            p
            for p in signature.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        extra_positional_params = []
        for k, v in pkeys.items():
            if v == UUID:
                extra_positional_params.append(
                    inspect.Parameter(
                        name=k,
                        annotation=Annotated[UUID, Path(...)],
                        kind=inspect.Parameter.POSITIONAL_ONLY,
                    )
                )
            else:
                extra_positional_params.append(
                    inspect.Parameter(
                        name=k, annotation=v, kind=inspect.Parameter.POSITIONAL_ONLY
                    )
                )

        endpoint.__signature__ = signature.replace(
            parameters=extra_positional_params + parameters
        )
        return endpoint

    return wrapper


def _create_auto_field_injector(
    config: Optional[Union[CreateConfig, UpdateConfig, DeleteConfig]],
) -> Callable[..., dict[str, Any]]:
    """
    Creates a dynamic dependency function that resolves auto_fields.

    Similar to _create_dynamic_filters but for CreateConfig/UpdateConfig/DeleteConfig.
    Returns a function that can be used with Depends() to inject auto field values.
    """
    if config is None or not config.auto_fields:
        return lambda: {}

    def auto_fields_resolver(**kwargs: Any) -> dict[str, Any]:
        """Receives resolved dependency values and returns dict of field:value."""
        return kwargs

    params = []
    for field_name, func in config.auto_fields.items():
        params.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Depends(func),
            )
        )

    sig = inspect.Signature(params)
    setattr(auto_fields_resolver, "__signature__", sig)

    return auto_fields_resolver


def _create_modified_schema(
    original_schema: type[BaseModel],
    exclude_fields: list[str],
    schema_name: str = "ModifiedSchema",
) -> type[BaseModel]:
    """
    Creates a new Pydantic schema with specified fields excluded.

    Args:
        original_schema: The original Pydantic schema
        exclude_fields: List of field names to exclude
        schema_name: Name for the new schema class

    Returns:
        A new Pydantic schema class without the excluded fields
    """
    if not exclude_fields:
        return original_schema

    from pydantic import create_model

    field_definitions: dict[str, Any] = {}
    for field_name, field_info in original_schema.model_fields.items():
        if field_name not in exclude_fields:
            field_definitions[field_name] = (field_info.annotation, field_info)

    new_schema: type[BaseModel] = create_model(
        schema_name,
        **field_definitions,  # type: ignore[arg-type]
    )

    return new_schema


def _create_dynamic_filters(
    filter_config: Optional[FilterConfig], column_types: dict[str, type]
) -> Callable[..., dict[str, Any]]:
    if filter_config is None:
        return lambda: {}

    param_to_filter_key = {}
    for original_key in filter_config.filters.keys():
        param_name = original_key.replace(".", "_")
        param_to_filter_key[param_name] = original_key

    def filters(
        **kwargs: Any,
    ) -> dict[str, Any]:
        filtered_params = {}
        for param_name, value in kwargs.items():
            if value is not None:
                original_key = param_to_filter_key.get(param_name, param_name)
                key_without_op = original_key.rsplit("__", 1)[0]
                parse_func = column_types.get(key_without_op)
                if parse_func:
                    try:
                        filtered_params[original_key] = parse_func(value)
                    except (ValueError, TypeError):
                        filtered_params[original_key] = value
                else:
                    filtered_params[original_key] = value
        return filtered_params

    params = []
    for key, value in filter_config.filters.items():
        param_name = key.replace(".", "_")

        if callable(value):
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=Depends(value),
                )
            )
        else:
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=Query(value, alias=key),
                )
            )

    sig = inspect.Signature(params)
    setattr(filters, "__signature__", sig)

    return filters
