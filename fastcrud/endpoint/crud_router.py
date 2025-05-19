from typing import Type, Optional, Union, Sequence, Callable
from enum import Enum

from fastapi import APIRouter

from fastcrud.crud.fast_crud import FastCRUD
from fastcrud.types import (
    CreateSchemaType,
    DeleteSchemaType,
    ModelType,
    UpdateSchemaType,
    SelectSchemaType,
)
from .endpoint_creator import EndpointCreator
from .helper import FilterConfig


def crud_router(
    session: Callable,
    model: ModelType,
    create_schema: Type[CreateSchemaType],
    update_schema: Type[UpdateSchemaType],
    crud: Optional[FastCRUD] = None,
    delete_schema: Optional[Type[DeleteSchemaType]] = None,
    path: str = "",
    tags: Optional[list[Union[str, Enum]]] = None,
    include_in_schema: bool = True,
    create_deps: Sequence[Callable] = [],
    read_deps: Sequence[Callable] = [],
    read_multi_deps: Sequence[Callable] = [],
    update_deps: Sequence[Callable] = [],
    delete_deps: Sequence[Callable] = [],
    db_delete_deps: Sequence[Callable] = [],
    included_methods: Optional[list[str]] = None,
    deleted_methods: Optional[list[str]] = None,
    endpoint_creator: Optional[Type[EndpointCreator]] = None,
    is_deleted_column: str = "is_deleted",
    deleted_at_column: str = "deleted_at",
    updated_at_column: str = "updated_at",
    endpoint_names: Optional[dict[str, str]] = None,
    filter_config: Optional[Union[FilterConfig, dict]] = None,
    select_schema: Optional[Type[SelectSchemaType]] = None,
) -> APIRouter:
    """
    Creates and configures a FastAPI router with CRUD endpoints for a given model.

    This utility function streamlines the process of setting up a router for CRUD operations,
    using a custom `EndpointCreator` if provided, and managing dependency injections as well
    as selective method inclusions or exclusions.

    Args:
        session: The SQLAlchemy async session.
        model: The SQLAlchemy model.
        create_schema: Pydantic schema for creating an item.
        update_schema: Pydantic schema for updating an item.
        crud: An optional `FastCRUD` instance. If not provided, uses `FastCRUD(model)`.
        delete_schema: Optional Pydantic schema for deleting an item.
        path: Base path for the CRUD endpoints.
        tags: Optional list of tags for grouping endpoints in the documentation.
        include_in_schema: Whether to include the created endpoints in the OpenAPI schema.
        create_deps: Optional list of functions to be injected as dependencies for the create endpoint.
        read_deps: Optional list of functions to be injected as dependencies for the read endpoint.
        read_multi_deps: Optional list of functions to be injected as dependencies for the read multiple items endpoint.
        update_deps: Optional list of functions to be injected as dependencies for the update endpoint.
        delete_deps: Optional list of functions to be injected as dependencies for the delete endpoint.
        db_delete_deps: Optional list of functions to be injected as dependencies for the hard delete endpoint.
        included_methods: Optional list of CRUD methods to include. If `None`, all methods are included.
        deleted_methods: Optional list of CRUD methods to exclude.
        endpoint_creator: Optional custom class derived from `EndpointCreator` for advanced customization.
        is_deleted_column: Optional column name to use for indicating a soft delete. Defaults to `"is_deleted"`.
        deleted_at_column: Optional column name to use for storing the timestamp of a soft delete. Defaults to `"deleted_at"`.
        updated_at_column: Optional column name to use for storing the timestamp of an update. Defaults to `"updated_at"`.
        endpoint_names: Optional dictionary to customize endpoint names for CRUD operations. Keys are operation types
                        (`"create"`, `"read"`, `"update"`, `"delete"`, `"db_delete"`, `"read_multi"`), and
                        values are the custom names to use. Unspecified operations will use default names.
        filter_config: Optional `FilterConfig` instance or dictionary to configure filters for the `read_multi` endpoint.
        select_schema: Optional Pydantic schema for selecting an item.

    Returns:
        Configured `APIRouter` instance with the CRUD endpoints.

    Raises:
        ValueError: If both `included_methods` and `deleted_methods` are provided.

    Examples:
        ??? example "Models and Schemas Used Below"

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
                fastcrud/examples/mymodel/schemas.py:deleteschema
                --8<--
                ```

            ---

            ??? example "`customer/model.py`"

                ```python
                --8<--
                fastcrud/examples/customer/model.py:imports
                fastcrud/examples/customer/model.py:model
                --8<--
                ```

            ??? example "`customer/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/customer/schemas.py:imports
                fastcrud/examples/customer/schemas.py:createschema
                fastcrud/examples/customer/schemas.py:readschema
                fastcrud/examples/customer/schemas.py:updateschema
                fastcrud/examples/customer/schemas.py:deleteschema
                --8<--
                ```

            ??? example "`product/model.py`"

                ```python
                --8<--
                fastcrud/examples/product/model.py:imports
                fastcrud/examples/product/model.py:model
                --8<--
                ```

            ??? example "`product/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/product/schemas.py:imports
                fastcrud/examples/product/schemas.py:createschema
                fastcrud/examples/product/schemas.py:readschema
                fastcrud/examples/product/schemas.py:updateschema
                fastcrud/examples/product/schemas.py:deleteschema
                --8<--
                ```

            ??? example "`order/model.py`"

                ```python
                --8<--
                fastcrud/examples/order/model.py:imports
                fastcrud/examples/order/model.py:model
                --8<--
                ```

            ??? example "`order/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/order/schemas.py:imports
                fastcrud/examples/order/schemas.py:createschema
                fastcrud/examples/order/schemas.py:readschema
                fastcrud/examples/order/schemas.py:updateschema
                fastcrud/examples/order/schemas.py:deleteschema
                --8<--
                ```

            ---

            ??? example "`tier/model.py`"

                ```python
                --8<--
                fastcrud/examples/tier/model.py:imports
                fastcrud/examples/tier/model.py:model
                --8<--
                ```

            ??? example "`tier/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/tier/schemas.py:imports
                fastcrud/examples/tier/schemas.py:readschema
                --8<--
                ```

            ??? example "`department/model.py`"

                ```python
                --8<--
                fastcrud/examples/department/model.py:imports
                fastcrud/examples/department/model.py:model
                --8<--
                ```

            ??? example "`department/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/department/schemas.py:imports
                fastcrud/examples/department/schemas.py:readschema
                --8<--
                ```

            ??? example "`user/model.py`"

                ```python
                --8<--
                fastcrud/examples/user/model.py:imports
                fastcrud/examples/user/model.py:model
                --8<--
                ```

            ??? example "`user/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/user/schemas.py:imports
                fastcrud/examples/user/schemas.py:createschema
                fastcrud/examples/user/schemas.py:readschema
                fastcrud/examples/user/schemas.py:updateschema
                fastcrud/examples/user/schemas.py:deleteschema
                --8<--
                ```

            ??? example "`story/model.py`"

                ```python
                --8<--
                fastcrud/examples/story/model.py:imports
                fastcrud/examples/story/model.py:model
                --8<--
                ```

            ??? example "`story/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/story/schemas.py:imports
                fastcrud/examples/story/schemas.py:createschema
                fastcrud/examples/story/schemas.py:readschema
                fastcrud/examples/story/schemas.py:updateschema
                fastcrud/examples/story/schemas.py:deleteschema
                --8<--
                ```

            ??? example "`task/model.py`"

                ```python
                --8<--
                fastcrud/examples/task/model.py:imports
                fastcrud/examples/task/model.py:model
                --8<--
                ```

            ??? example "`task/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/task/schemas.py:imports
                fastcrud/examples/task/schemas.py:createschema
                fastcrud/examples/task/schemas.py:readschema
                fastcrud/examples/task/schemas.py:updateschema
                fastcrud/examples/task/schemas.py:deleteschema
                --8<--
                ```

        Basic Setup:

        ```python
        mymodel_router = crud_router(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            path="/mymodel",
            tags=["MyModel"],
        )
        ```

        With Custom Dependencies:

        ```python
        def get_current_user(token: str = Depends(oauth2_scheme)):
            # Implement user retrieval logic
            return ...

        user_router = crud_router(
            session=async_session,
            model=User,
            create_schema=CreateUserSchema,
            update_schema=UpdateUserSchema,
            read_deps=[get_current_user],
            update_deps=[get_current_user],
            path="/users",
            tags=["Users"],
        )
        ```

        Adding Delete Endpoints:
        ```python
        product_router = crud_router(
            session=async_session,
            model=Product,
            create_schema=CreateProductSchema,
            update_schema=UpdateProductSchema,
            delete_schema=DeleteProductSchema,
            path="/products",
            tags=["Products"],
        )
        ```

        Customizing Path and Tags:
        ```python
        OrderCRUD = FastCRUD[
            Order,
            CreateOrderSchema,
            UpdateOrderSchema,
            UpdateOrderSchema,
            DeleteOrderSchema,
        ]
        order_router = crud_router(
            session=async_session,
            model=Order,
            create_schema=CreateOrderSchema,
            update_schema=UpdateOrderSchema,
            crud=OrderCRUD(Order),
            path="/orders",
            tags=["Orders", "Sales"],
        )
        ```

        Integrating Multiple Models:
        ```python
        ProductCRUD = FastCRUD[
            Product,
            CreateProductSchema,
            UpdateProductSchema,
            UpdateProductSchema,
            DeleteProductSchema,
        ]
        product_router = crud_router(
            session=async_session,
            model=Product,
            create_schema=CreateProductSchema,
            update_schema=UpdateProductSchema,
            crud=ProductCRUD(Product),
            path="/products",
            tags=["Inventory"],
        )

        CustomerCRUD = FastCRUD[
            Customer,
            CreateCustomerSchema,
            UpdateCustomerSchema,
            UpdateCustomerSchema,
            DeleteCustomerSchema,
        ]
        customer_router = crud_router(
            session=async_session,
            model=Customer,
            create_schema=CreateCustomerSchema,
            update_schema=UpdateCustomerSchema,
            crud=CustomerCRUD(Customer),
            path="/customers",
            tags=["CRM"],
        )
        ```

        With Selective CRUD Methods:

        ```python
        MyModelCRUD = FastCRUD[
            MyModel,
            CreateMyModelSchema,
            UpdateMyModelSchema,
            UpdateMyModelSchema,
            DeleteMyModelSchema,
        ]
        # Only include 'create' and 'read' methods
        mymodel_router = crud_router(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            crud=MyModelCRUD(MyModel),
            path="/mymodel",
            tags=["MyModel"],
            included_methods=["create", "read"],
        )
        ```

        Using a Custom `EndpointCreator`:

        ```python
        class CustomEndpointCreator(EndpointCreator):
            def _custom_route(self):
                async def custom_endpoint():
                    # Custom endpoint logic
                    return {"message": "Custom route"}

                return custom_endpoint

                async def add_routes_to_router(self, ...):
                    # First, add standard CRUD routes
                    super().add_routes_to_router(...)

                    # Now, add custom routes
                    self.router.add_api_route(
                        path="/custom",
                        endpoint=self._custom_route(),
                        methods=["GET"],
                        tags=self.tags,
                        # Other parameters as needed
                    )

        MyModelCRUD = FastCRUD[
            MyModel,
            CreateMyModelSchema,
            UpdateMyModelSchema,
            UpdateMyModelSchema,
            DeleteMyModelSchema,
        ]
        mymodel_router = crud_router(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            crud=MyModelCRUD(MyModel),
            path="/mymodel",
            tags=["MyModel"],
            endpoint_creator=CustomEndpointCreator,
        )

        app.include_router(mymodel_router)
        ```

        Customizing Endpoint Names:

        ```python
        task_router = crud_router(
            session=async_session,
            model=Task,
            create_schema=CreateTaskSchema,
            update_schema=UpdateTaskSchema,
            path="/tasks",
            tags=["Task Management"],
            endpoint_names={
                "create": "add_task",
                "read": "get_task",
                "update": "modify_task",
                "delete": "remove_task",
                "db_delete": "permanently_remove_task",
                "read_multi": "list_tasks",
            },
        )
        ```

        Using `FilterConfig` with `dict`:

        ```python
        from fastapi import FastAPI
        from fastcrud import crud_router

        from .database import async_session
        from .mymodel.model import MyModel
        from .mymodel.schemas import CreateMyModelSchema, UpdateMyModelSchema

        app = FastAPI()

        mymodel_router = crud_router(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            filter_config=FilterConfig(filters={"id": None, "name": "default"}),
        )
        # Adds CRUD routes with filtering capabilities
        app.include_router(mymodel_router, prefix="/mymodel")

        # Explanation:
        # The FilterConfig specifies that 'id' should be a query parameter with no default value
        # and 'name' should be a query parameter with a default value of 'default'.
        # When fetching multiple items, you can filter by these parameters.
        # Example GET request: /mymodel/get_multi?id=1&name=example
        ```

        Using `FilterConfig` with keyword arguments:

        ```python
        from fastapi import FastAPI
        from fastcrud import crud_router

        from .database import async_session
        from .mymodel.model import MyModel
        from .mymodel.schemas import CreateMyModelSchema, UpdateMyModelSchema

        app = FastAPI()

        mymodel_router = crud_router(
            session=async_session,
            model=MyModel,
            create_schema=CreateMyModelSchema,
            update_schema=UpdateMyModelSchema,
            filter_config=FilterConfig(id=None, name="default"),
        )
        # Adds CRUD routes with filtering capabilities
        app.include_router(mymodel_router, prefix="/mymodel")

        # Explanation:
        # The FilterConfig specifies that 'id' should be a query parameter with no default value
        # and 'name' should be a query parameter with a default value of 'default'.
        # When fetching multiple items, you can filter by these parameters.
        # Example GET request: /mymodel/get_multi?id=1&name=example
        ```
    """

    crud = crud or FastCRUD(
        model=model,
        is_deleted_column=is_deleted_column,
        deleted_at_column=deleted_at_column,
        updated_at_column=updated_at_column,
    )

    endpoint_creator_class = endpoint_creator or EndpointCreator
    endpoint_creator_instance = endpoint_creator_class(
        session=session,
        crud=crud,
        create_schema=create_schema,  # type: ignore
        update_schema=update_schema,  # type: ignore
        include_in_schema=include_in_schema,
        delete_schema=delete_schema,  # type: ignore
        path=path,
        tags=tags,
        is_deleted_column=is_deleted_column,
        deleted_at_column=deleted_at_column,
        updated_at_column=updated_at_column,
        endpoint_names=endpoint_names,
        filter_config=filter_config,
        select_schema=select_schema,  # type: ignore
    )

    endpoint_creator_instance.add_routes_to_router(
        create_deps=create_deps,
        read_deps=read_deps,
        read_multi_deps=read_multi_deps,
        update_deps=update_deps,
        delete_deps=delete_deps,
        db_delete_deps=db_delete_deps,
        included_methods=included_methods,
        deleted_methods=deleted_methods,
    )

    return endpoint_creator_instance.router
