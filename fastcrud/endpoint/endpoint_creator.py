from typing import Type, Optional, Callable, Sequence, Union, Any, cast
from enum import Enum

from fastapi import Depends, Body, Query, APIRouter
from pydantic import ValidationError, BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from fastcrud.crud.fast_crud import FastCRUD
from fastcrud.paginated import ListResponse, PaginatedListResponse
from fastcrud.types import (
    CreateSchemaType,
    DeleteSchemaType,
    ModelType,
    SelectSchemaType,
    UpdateSchemaType,
)
from ..exceptions.http_exceptions import (
    DuplicateValueException,
    NotFoundException,
    BadRequestException,
)
from ..paginated.helper import compute_offset
from ..paginated.response import paginated_response
from .helper import (
    CRUDMethods,
    FilterConfig,
    _extract_unique_columns,
    _get_primary_keys,
    _get_python_type,
    _inject_dependencies,
    _apply_model_pk,
    _create_dynamic_filters,
    _get_column_types,
)


class EndpointCreator:
    """
    A class to create and register CRUD endpoints for a FastAPI application.

    This class simplifies the process of adding create, read, update, and delete (CRUD) endpoints
    to a FastAPI router. It is initialized with a SQLAlchemy session, model, CRUD operations,
    and Pydantic schemas, and allows for custom dependency injection for each endpoint.
    The method assumes `id` is the primary key for path parameters.

    Attributes:
        session: The SQLAlchemy async session.
        model: The SQLAlchemy model.
        create_schema: Pydantic schema for creating an item.
        update_schema: Pydantic schema for updating an item.
        crud: An optional FastCRUD instance. If not provided, uses `FastCRUD(model)`.
        include_in_schema: Whether to include the created endpoints in the OpenAPI schema.
        delete_schema: Optional Pydantic schema for deleting an item.
        path: Base path for the CRUD endpoints.
        tags: List of tags for grouping endpoints in the documentation.
        is_deleted_column: Optional column name to use for indicating a soft delete. Defaults to `"is_deleted"`.
        deleted_at_column: Optional column name to use for storing the timestamp of a soft delete. Defaults to `"deleted_at"`.
        updated_at_column: Optional column name to use for storing the timestamp of an update. Defaults to `"updated_at"`.
        endpoint_names: Optional dictionary to customize endpoint names for CRUD operations. Keys are operation types
                        (`"create"`, `"read"`, `"update"`, `"delete"`, `"db_delete"`, `"read_multi"`), and
                        values are the custom names to use. Unspecified operations will use default names.
        filter_config: Optional `FilterConfig` instance or dictionary to configure filters for the `read_multi` endpoint.
        select_schema: Optional Pydantic schema for selecting an item.

    Raises:
        ValueError: If both `included_methods` and `deleted_methods` are provided.

    Examples:
        Basic Setup:

        ??? example "`mymodel/model.py`"

            ```python
            --8<--
            fastcrud/examples/mymodel/model.py:imports
            fastcrud/examples/mymodel/model.py:model_simple
            --8<--
            ```

        ??? example "`mymodel/schemas.py`"

            ```python
            --8<--
            fastcrud/examples/mymodel/schemas.py:imports
            fastcrud/examples/mymodel/schemas.py:createschema
            fastcrud/examples/mymodel/schemas.py:updateschema
            --8<--
            ```

        ```python
        from fastapi import FastAPI
        from fastcrud import EndpointCreator

        from .database import async_session
        from .mymodel.model import MyModel
        from .mymodel.schemas import CreateMyModelSchema, UpdateMyModelSchema

        app = FastAPI()
        endpoint_creator = EndpointCreator(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
        )
        endpoint_creator.add_routes_to_router()
        app.include_router(endpoint_creator.router, prefix="/mymodel")
        ```

        With Custom Dependencies:

        ```python
        from fastapi.security import OAuth2PasswordBearer

        oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

        def get_current_user(token: str = Depends(oauth2_scheme)):
            return ...

        endpoint_creator.add_routes_to_router(
            read_deps=[get_current_user],
            update_deps=[get_current_user],
        )
        ```

        Selective Endpoint Creation (inclusion):

        ```python
        # Only create 'create' and 'read' endpoints
        endpoint_creator.add_routes_to_router(
            included_methods=["create", "read"],
        )
        ```

        Selective Endpoint Creation (deletion):

        ```python
        # Create all but 'update' and 'delete' endpoints
        endpoint_creator.add_routes_to_router(
            deleted_methods=["update", "delete"],
        )
        ```

        Integrating with Multiple Models:

        ```python
        # Assuming definitions for OtherModel, OtherModelCRUD, etc.

        other_model_crud = OtherModelCRUD(OtherModel)
        other_endpoint_creator = EndpointCreator(
            session=async_session,
            model=OtherModel,
            create_schema=CreateOtherModelSchema,
            update_schema=UpdateOtherModelSchema,
            crud=other_model_crud,
        )
        other_endpoint_creator.add_routes_to_router()
        app.include_router(other_endpoint_creator.router, prefix="/othermodel")
        ```

        Customizing Endpoint Names:

        ```python
        endpoint_creator = EndpointCreator(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            path="/mymodel",
            tags=["MyModel"],
            endpoint_names={
                "create": "add",  # Custom endpoint name for creating items
                "read": "fetch",  # Custom endpoint name for reading a single item
                "update": "change",  # Custom endpoint name for updating items
                # The delete operation will use the default name "delete"
            },
        )
        endpoint_creator.add_routes_to_router()
        ```

        Using `filter_config` with `dict`:

        ```python
        from fastapi import FastAPI
        from fastcrud import EndpointCreator, FilterConfig

        from .database import async_session
        from .mymodel.model import MyModel
        from .mymodel.schemas import CreateMyModelSchema, UpdateMyModelSchema

        app = FastAPI()
        endpoint_creator = EndpointCreator(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            filter_config=FilterConfig(filters={"id": None, "name": "default"}),
        )
        # Adds CRUD routes with filtering capabilities
        endpoint_creator.add_routes_to_router()
        # Include the internal router into the FastAPI app
        app.include_router(endpoint_creator.router, prefix="/mymodel")

        # Explanation:
        # The FilterConfig specifies that 'id' should be a query parameter with no default value
        # and 'name' should be a query parameter with a default value of 'default'.
        # When fetching multiple items, you can filter by these parameters.
        # Example GET request: /mymodel/get_multi?id=1&name=example
        ```

        Using `filter_config` with keyword arguments:

        ```python
        from fastapi import FastAPI
        from fastcrud import EndpointCreator, FilterConfig

        from .database import async_session
        from .mymodel.model import MyModel
        from .mymodel.schemas import CreateMyModelSchema, UpdateMyModelSchema

        app = FastAPI()
        endpoint_creator = EndpointCreator(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            filter_config=FilterConfig(id=None, name="default"),
        )
        # Adds CRUD routes with filtering capabilities
        endpoint_creator.add_routes_to_router()
        # Include the internal router into the FastAPI app
        app.include_router(endpoint_creator.router, prefix="/mymodel")

        # Explanation:
        # The FilterConfig specifies that 'id' should be a query parameter with no default value
        # and 'name' should be a query parameter with a default value of 'default'.
        # When fetching multiple items, you can filter by these parameters.
        # Example GET request: /mymodel/get_multi?id=1&name=example
        ```
    """

    def __init__(
        self,
        session: Callable,
        model: ModelType,
        create_schema: Type[CreateSchemaType],
        update_schema: Type[UpdateSchemaType],
        crud: Optional[FastCRUD] = None,
        include_in_schema: bool = True,
        delete_schema: Optional[Type[DeleteSchemaType]] = None,
        path: str = "",
        tags: Optional[list[Union[str, Enum]]] = None,
        is_deleted_column: str = "is_deleted",
        deleted_at_column: str = "deleted_at",
        updated_at_column: str = "updated_at",
        endpoint_names: Optional[dict[str, str]] = None,
        filter_config: Optional[Union[FilterConfig, dict]] = None,
        select_schema: Optional[Type[SelectSchemaType]] = None,
    ) -> None:
        self._primary_keys = _get_primary_keys(model)
        self._primary_keys_types = {
            pk.name: _get_python_type(pk) for pk in self._primary_keys
        }
        self.primary_key_names = [pk.name for pk in self._primary_keys]
        self.session = session
        self.crud = crud or FastCRUD(
            model=model,
            is_deleted_column=is_deleted_column,
            deleted_at_column=deleted_at_column,
            updated_at_column=updated_at_column,
        )
        self.model = model
        self.create_schema = create_schema
        self.update_schema = update_schema
        self.delete_schema = delete_schema
        self.select_schema = select_schema
        self.include_in_schema = include_in_schema
        self.path = path
        self.tags = tags or []
        self.router = APIRouter()
        self.is_deleted_column = is_deleted_column
        self.deleted_at_column = deleted_at_column
        self.updated_at_column = updated_at_column
        self.default_endpoint_names = {
            "create": "",
            "read": "",
            "update": "",
            "delete": "",
            "db_delete": "db_delete",
            "read_multi": "",
        }
        self.endpoint_names = {**self.default_endpoint_names, **(endpoint_names or {})}
        if filter_config:
            if isinstance(filter_config, dict):
                filter_config = FilterConfig(**filter_config)
            self._validate_filter_config(filter_config)
        self.filter_config = filter_config
        self.column_types = _get_column_types(model)

        if select_schema is not None:
            self.list_response_model: Optional[Type[ListResponse[Any]]] = type(
                "DynamicListResponse",
                (ListResponse[BaseModel],),
                {"__annotations__": {"data": list[select_schema]}},  # type: ignore
            )
            self.paginated_response_model: Optional[
                Type[PaginatedListResponse[Any]]
            ] = type(
                "DynamicPaginatedResponse",
                (PaginatedListResponse[BaseModel],),
                {
                    "__annotations__": {
                        "data": list[select_schema],  # type: ignore
                        "total_count": int,
                        "has_more": bool,
                        "page": Optional[int],
                        "items_per_page": Optional[int],
                    }
                },
            )
        else:
            self.list_response_model = None
            self.paginated_response_model = None

    def _validate_filter_config(self, filter_config: FilterConfig) -> None:
        model_columns = self.crud.model_col_names
        supported_filters = self.crud._SUPPORTED_FILTERS
        for key, value in filter_config.filters.items():
            # Skip validation for callable values as they will be resolved at runtime
            if callable(value):
                continue

            if "__" in key:
                field_name, op = key.rsplit("__", 1)
                if op not in supported_filters:
                    raise ValueError(
                        f"Invalid filter op '{op}': following filter ops are allowed: {supported_filters.keys()}"
                    )
            else:
                field_name = key

            if field_name not in model_columns:
                raise ValueError(
                    f"Invalid filter column '{key}': not found in model '{self.model.__name__}' columns"
                )

    def _create_item(self):
        """Creates an endpoint for creating items in the database."""

        async def endpoint(
            db: AsyncSession = Depends(self.session),
            item: self.create_schema = Body(...),  # type: ignore
        ):
            unique_columns = _extract_unique_columns(self.model)

            for column in unique_columns:
                col_name = column.name
                if hasattr(item, col_name):
                    value = getattr(item, col_name)
                    exists = await self.crud.exists(db, **{col_name: value})
                    if exists:  # pragma: no cover
                        raise DuplicateValueException(
                            f"Value {value} is already registered"
                        )

            return await self.crud.create(db, item)

        return endpoint

    def _read_item(self):
        """Creates an endpoint for reading a single item from the database."""

        @_apply_model_pk(**self._primary_keys_types)
        async def endpoint(db: AsyncSession = Depends(self.session), **pkeys):
            if self.select_schema is not None:
                item = await self.crud.get(
                    db,
                    schema_to_select=cast(Type[BaseModel], self.select_schema),
                    return_as_model=True,
                    **pkeys,
                )
            else:
                item = await self.crud.get(db, **pkeys)
            if not item:  # pragma: no cover
                raise NotFoundException(detail="Item not found")
            return item  # pragma: no cover

        return endpoint

    def _read_items(self):
        """Creates an endpoint for reading multiple items from the database."""
        dynamic_filters = _create_dynamic_filters(self.filter_config, self.column_types)

        async def endpoint(
            db: AsyncSession = Depends(self.session),
            offset: Optional[int] = Query(
                None, description="Offset for unpaginated queries"
            ),
            limit: Optional[int] = Query(
                None, description="Limit for unpaginated queries"
            ),
            page: Optional[int] = Query(None, alias="page", description="Page number"),
            items_per_page: Optional[int] = Query(
                None, alias="itemsPerPage", description="Number of items per page"
            ),
            filters: dict = Depends(dynamic_filters),
        ) -> Union[dict[str, Any], PaginatedListResponse, ListResponse]:
            is_paginated = (page is not None) or (items_per_page is not None)
            has_offset_limit = (offset is not None) and (limit is not None)

            if is_paginated and has_offset_limit:
                raise BadRequestException(
                    detail="Conflicting parameters: Use either 'page' and 'itemsPerPage' for paginated results or 'offset' and 'limit' for specific range queries."
                )

            if is_paginated:
                if not page:
                    page = 1
                if not items_per_page:
                    items_per_page = 10
                offset = compute_offset(page=page, items_per_page=items_per_page)  # type: ignore
                limit = items_per_page
            elif not has_offset_limit:
                offset = 0
                limit = 100

            if self.select_schema is not None:
                crud_data = await self.crud.get_multi(
                    db,
                    offset=offset,  # type: ignore
                    limit=limit,  # type: ignore
                    schema_to_select=self.select_schema,
                    return_as_model=True,
                    **filters,
                )
            else:
                crud_data = await self.crud.get_multi(
                    db,
                    offset=offset,  # type: ignore
                    limit=limit,  # type: ignore
                    **filters,
                )

            if is_paginated:
                return paginated_response(
                    crud_data=crud_data,
                    page=page,  # type: ignore
                    items_per_page=items_per_page,  # type: ignore
                )

            return crud_data  # pragma: no cover

        return endpoint

    def _update_item(self):
        """Creates an endpoint for updating an existing item in the database."""

        @_apply_model_pk(**self._primary_keys_types)
        async def endpoint(
            item: self.update_schema = Body(...),  # type: ignore
            db: AsyncSession = Depends(self.session),
            **pkeys,
        ):
            return await self.crud.update(db, item, **pkeys)

        return endpoint

    def _delete_item(self):
        """Creates an endpoint for deleting an item from the database."""

        @_apply_model_pk(**self._primary_keys_types)
        async def endpoint(db: AsyncSession = Depends(self.session), **pkeys):
            await self.crud.delete(db, **pkeys)
            return {"message": "Item deleted successfully"}  # pragma: no cover

        return endpoint

    def _db_delete(self):
        """
        Creates an endpoint for hard deleting an item from the database.

        This endpoint is only added if the `delete_schema` is provided during initialization.
        The endpoint expects an item ID as a path parameter and uses the provided SQLAlchemy
        async session to permanently delete the item from the database.
        """

        @_apply_model_pk(**self._primary_keys_types)
        async def endpoint(db: AsyncSession = Depends(self.session), **pkeys):
            await self.crud.db_delete(db, **pkeys)
            return {
                "message": "Item permanently deleted from the database"
            }  # pragma: no cover

        return endpoint

    def _get_endpoint_path(self, operation: str):
        endpoint_name = self.endpoint_names.get(
            operation, self.default_endpoint_names.get(operation, operation)
        )
        path = f"{self.path}/{endpoint_name}" if endpoint_name else self.path

        if operation in {"read", "update", "delete", "db_delete"}:
            _primary_keys_path_suffix = "/".join(
                f"{{{n}}}" for n in self.primary_key_names
            )
            path = f"{path}/{_primary_keys_path_suffix}"

        return path

    def add_routes_to_router(
        self,
        create_deps: Sequence[Callable] = [],
        read_deps: Sequence[Callable] = [],
        read_multi_deps: Sequence[Callable] = [],
        update_deps: Sequence[Callable] = [],
        delete_deps: Sequence[Callable] = [],
        db_delete_deps: Sequence[Callable] = [],
        included_methods: Optional[Sequence[str]] = None,
        deleted_methods: Optional[Sequence[str]] = None,
    ):
        """
        Adds CRUD operation routes to the FastAPI router with specified dependencies for each type of operation.

        This method registers routes for create, read, update, and delete operations with the FastAPI router,
        allowing for custom dependency injection for each type of operation.

        Args:
            create_deps: List of functions to be injected as dependencies for the create endpoint.
            read_deps: List of functions to be injected as dependencies for the read endpoint.
            read_multi_deps: List of functions to be injected as dependencies for the read multiple items endpoint.
            update_deps: List of functions to be injected as dependencies for the update endpoint.
            delete_deps: List of functions to be injected as dependencies for the delete endpoint.
            db_delete_deps: List of functions to be injected as dependencies for the hard delete endpoint.
            included_methods: Optional list of methods to include. Defaults to all CRUD methods.
            deleted_methods: Optional list of methods to exclude. Defaults to `None`.

        Raises:
            ValueError: If both `included_methods` and `deleted_methods` are provided.

        Examples:
            Selective Endpoint Creation:

            ```python
            # Only create 'create' and 'read' endpoints
            endpoint_creator.add_routes_to_router(
                included_methods=["create", "read"],
            )
            ```

            Excluding Specific Endpoints:

            ```python
            # Create all endpoints except 'delete' and 'db_delete'
            endpoint_creator.add_routes_to_router(
                deleted_methods=["delete", "db_delete"],
            )
            ```

            With Custom Dependencies and Selective Endpoints:

            ```python
            def get_current_user(...):
                ...

            # Create only 'read' and 'update' endpoints with custom dependencies
            endpoint_creator.add_routes_to_router(
                read_deps=[get_current_user],
                update_deps=[get_current_user],
                included_methods=["read", "update"],
            )
            ```

        Note:
            This method should be called to register the endpoints with the FastAPI application.
            If `delete_schema` is provided on class instantiation, a hard delete endpoint is also registered.
            This method assumes `id` is the primary key for path parameters.
        """
        if (included_methods is not None) and (deleted_methods is not None):
            raise ValueError(
                "Cannot use both 'included_methods' and 'deleted_methods' simultaneously."
            )

        if included_methods is None:
            included_methods = [
                "create",
                "read",
                "read_multi",
                "update",
                "delete",
                "db_delete",
            ]
        else:
            try:
                included_methods = CRUDMethods(
                    valid_methods=included_methods
                ).valid_methods
            except ValidationError as e:
                raise ValueError(f"Invalid CRUD methods in included_methods: {e}")

        if deleted_methods is None:
            deleted_methods = []
        else:
            try:
                deleted_methods = CRUDMethods(
                    valid_methods=deleted_methods
                ).valid_methods
            except ValidationError as e:
                raise ValueError(f"Invalid CRUD methods in deleted_methods: {e}")

        delete_description = "Delete a"
        if self.delete_schema:
            delete_description = "Soft delete a"

        if ("create" in included_methods) and ("create" not in deleted_methods):
            self.router.add_api_route(
                self._get_endpoint_path(operation="create"),
                self._create_item(),
                methods=["POST"],
                include_in_schema=self.include_in_schema,
                tags=self.tags,
                dependencies=_inject_dependencies(create_deps),
                description=f"Create a new {self.model.__name__} row in the database.",
            )

        if ("read" in included_methods) and ("read" not in deleted_methods):
            self.router.add_api_route(
                self._get_endpoint_path(operation="read"),
                self._read_item(),
                methods=["GET"],
                include_in_schema=self.include_in_schema,
                tags=self.tags,
                dependencies=_inject_dependencies(read_deps),
                response_model=self.select_schema if self.select_schema else None,
                description=f"Read a single {self.model.__name__} row from the database by its primary keys: {self.primary_key_names}.",
            )

        if ("read_multi" in included_methods) and ("read_multi" not in deleted_methods):
            if self.select_schema is not None:
                response_model: Optional[
                    Type[Union[PaginatedListResponse[Any], ListResponse[Any]]]
                ] = Union[
                    self.paginated_response_model,  # type: ignore
                    self.list_response_model,  # type: ignore
                ]
            else:
                response_model = None

            self.router.add_api_route(
                self._get_endpoint_path(operation="read_multi"),
                self._read_items(),
                methods=["GET"],
                include_in_schema=self.include_in_schema,
                tags=self.tags,
                dependencies=_inject_dependencies(read_multi_deps),
                response_model=response_model,
                description=(
                    f"Read multiple {self.model.__name__} rows from the database.\n\n"
                    f"- Use page & itemsPerPage for paginated results\n"
                    f"- Use offset & limit for specific ranges\n"
                    f"- Returns paginated response when using page/itemsPerPage\n"
                    f"- Returns simple list response when using offset/limit"
                ),
            )

        if ("update" in included_methods) and ("update" not in deleted_methods):
            self.router.add_api_route(
                self._get_endpoint_path(operation="update"),
                self._update_item(),
                methods=["PATCH"],
                include_in_schema=self.include_in_schema,
                tags=self.tags,
                dependencies=_inject_dependencies(update_deps),
                description=f"Update an existing {self.model.__name__} row in the database by its primary keys: {self.primary_key_names}.",
            )

        if ("delete" in included_methods) and ("delete" not in deleted_methods):
            path = self._get_endpoint_path(operation="delete")
            self.router.add_api_route(
                path,
                self._delete_item(),
                methods=["DELETE"],
                include_in_schema=self.include_in_schema,
                tags=self.tags,
                dependencies=_inject_dependencies(delete_deps),
                description=f"{delete_description} {self.model.__name__} row from the database by its primary keys: {self.primary_key_names}.",
            )

        if (
            ("db_delete" in included_methods)
            and ("db_delete" not in deleted_methods)
            and self.delete_schema
        ):
            self.router.add_api_route(
                self._get_endpoint_path(operation="db_delete"),
                self._db_delete(),
                methods=["DELETE"],
                include_in_schema=self.include_in_schema,
                tags=self.tags,
                dependencies=_inject_dependencies(db_delete_deps),
                description=f"Permanently delete a {self.model.__name__} row from the database by its primary keys: {self.primary_key_names}.",
            )

    def add_custom_route(
        self,
        endpoint: Callable,
        methods: Optional[Union[set[str], list[str]]],
        path: Optional[str] = None,
        dependencies: Optional[Sequence[Callable]] = None,
        include_in_schema: bool = True,
        tags: Optional[list[Union[str, Enum]]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
    ) -> None:
        """
        Adds a custom route to the FastAPI router.

        Args:
            endpoint: The endpoint function to execute when the route is called.
            methods: A list of HTTP methods for the route (e.g., `['GET', 'POST']`).
            path: URL path for the custom route.
            dependencies: A list of functions to be injected as dependencies for the route.
            include_in_schema: Whether to include this route in the OpenAPI schema.
            tags: Tags for grouping and categorizing the route in documentation.
            summary: A short summary of the route, for documentation.
            description: A detailed description of the route, for documentation.
            response_description: A description of the response, for documentation.

        Example:
            ```python
            async def custom_endpoint(foo: int, bar: str):
                # custom logic here
                return {"foo": foo, "bar": bar}

            endpoint_creator.add_custom_route(
                endpoint=custom_endpoint,
                methods=["GET"],
                path="/custom",
                tags=["custom"],
                summary="Custom Endpoint",
                description="This is a custom endpoint.",
            )
            ```
        """
        path = path or self.path
        full_path = f"{self.path}{path}"
        self.router.add_api_route(
            path=full_path,
            endpoint=endpoint,
            methods=methods,
            dependencies=_inject_dependencies(dependencies) or [],
            include_in_schema=include_in_schema,
            tags=tags or self.tags,
            summary=summary,
            description=description,
            response_description=response_description,
        )
