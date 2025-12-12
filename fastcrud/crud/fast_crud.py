from typing import Any, Generic, Union, Optional, Callable, overload, Literal, cast
from datetime import datetime, timezone

from sqlalchemy import (
    select,
    update,
    delete,
    func,
    column,
    Column,
)
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.sql import Join
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql.elements import BinaryExpression, ColumnElement
from sqlalchemy.sql.selectable import Select

from fastcrud.types import (
    CreateSchemaType,
    DeleteSchemaType,
    ModelType,
    SelectSchemaType,
    UpdateSchemaInternalType,
    UpdateSchemaType,
    GetMultiResponseModel,
    GetMultiResponseDict,
    UpsertMultiResponseModel,
    UpsertMultiResponseDict,
)

from ..core import (
    extract_matching_columns_from_schema,
    auto_detect_join_condition,
    JoinConfig,
    CountConfig,
    get_primary_key_names,
    get_primary_key_columns,
    FilterProcessor,
    SQLQueryBuilder,
    format_multi_response,
    process_joined_data,
    build_joined_query,
    execute_joined_query,
    format_joined_response,
)

from .validation import (
    validate_update_delete_operation,
    validate_pagination_params,
    validate_joined_query_params,
)
from .data_preparation import prepare_update_data
from .execution import (
    execute_update_and_return_response,
    handle_joined_filters_delegation,
)
from ..core.protocols import CRUDInstance
from .database_specific import (
    upsert_multi_postgresql,
    upsert_multi_sqlite,
    upsert_multi_mysql,
)

FilterCallable = Callable[[Column[Any]], Callable[..., ColumnElement[bool]]]


class FastCRUD(
    Generic[
        ModelType,
        CreateSchemaType,
        UpdateSchemaType,
        UpdateSchemaInternalType,
        DeleteSchemaType,
        SelectSchemaType,
    ]
):
    """
    Base class for CRUD operations on a model.

    This class provides a set of methods for create, read, update, and delete operations on a given SQLAlchemy model,
    utilizing Pydantic schemas for data validation and serialization.

    Args:
        model: The SQLAlchemy model type.
        is_deleted_column: Optional column name to use for indicating a soft delete. Defaults to `"is_deleted"`.
        deleted_at_column: Optional column name to use for storing the timestamp of a soft delete. Defaults to `"deleted_at"`.
        updated_at_column: Optional column name to use for storing the timestamp of an update. Defaults to `"updated_at"`.

    Methods:
        create:
            Creates a new record in the database from the provided Pydantic schema.

        select:
            Generates a SQL Alchemy `Select` statement with optional filtering and sorting.

        get:
            Retrieves a single record based on filters. Supports advanced filtering through comparison operators like `__gt`, `__lt`, etc.

        exists:
            Checks if a record exists based on the provided filters.

        count:
            Counts the number of records matching the provided filters.

        get_multi:
            Fetches multiple records with optional sorting, pagination, and model conversion.

        get_joined:
            Performs a join operation with another model, supporting custom join conditions and selection of specific columns.

        get_multi_joined:
            Fetches multiple records with a join on another model, offering pagination and sorting for the joined tables.

        get_multi_by_cursor:
            Implements cursor-based pagination for fetching records, ideal for large datasets and infinite scrolling features.

        update:
            Updates an existing record or multiple records based on specified filters.

        db_delete:
            Hard deletes a record or multiple records from the database based on provided filters.

        delete:
            Soft deletes a record if it has an `"is_deleted"` attribute (or other attribute as defined by `is_deleted_column`); otherwise, performs a hard delete.

    Examples:
        ??? example "Models and Schemas Used Below"

            ??? example "`customer/model.py`"

                ```python
                --8<--
                fastcrud/examples/customer/model.py:imports
                fastcrud/examples/customer/model.py:model
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
                fastcrud/examples/product/schemas.py:readschema
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
                fastcrud/examples/order/schemas.py:readschema
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

            ---

            ??? example "`profile/model.py`"

                ```python
                --8<--
                fastcrud/examples/profile/model.py:imports
                fastcrud/examples/profile/model.py:model
                --8<--
                ```

            ??? example "`profile/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/profile/schemas.py:imports
                fastcrud/examples/profile/schemas.py:readschema
                --8<--
                ```

            ??? example "`author/model.py`"

                ```python
                --8<--
                fastcrud/examples/author/model.py:imports
                fastcrud/examples/author/model.py:model
                --8<--
                ```

            ??? example "`author/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/author/schemas.py:imports
                fastcrud/examples/author/schemas.py:readschema
                --8<--
                ```

            ??? example "`article/model.py`"

                ```python
                --8<--
                fastcrud/examples/article/model.py:imports
                fastcrud/examples/article/model.py:model
                --8<--
                ```

            ??? example "`article/schemas.py`"

                ```python
                --8<--
                fastcrud/examples/article/schemas.py:imports
                fastcrud/examples/article/schemas.py:readschema
                --8<--
                ```

            ---

            ??? example "`Project`, `Participant`, `ProjectsParticipantsAssociation`"

                ```python
                # These models taken from tests/sqlalchemy/conftest.py
                --8<--
                tests/sqlalchemy/conftest.py:model_project
                tests/sqlalchemy/conftest.py:model_participant
                tests/sqlalchemy/conftest.py:model_proj_parts_assoc
                --8<--
                ```

            ??? example "`ReadProjectSchema`"

                ```python
                class ReadProjectSchema(BaseModel):
                    id: int
                    name: str
                    description: Optional[str] = None
                ```

        Example 1: Basic Usage
        ----------------------

        Create a FastCRUD instance for a `User` model and perform basic CRUD operations.

        ```python
        # Assuming you have a User model (either SQLAlchemy or SQLModel)
        # pydantic schemas for creation, update and deletion and an async session `db`
        UserCRUD = FastCRUD[User, CreateUserSchema, UpdateUserSchema, None, DeleteUserSchema]
        user_crud = UserCRUD(User)

        # If you don't care about typing, you can also just ignore the UserCRUD part
        # Straight up define user_crud with FastCRUD
        user_crud = FastCRUD(User)

        # Create a new user
        new_user = await user_crud.create(db, CreateUserSchema(name="Alice"))
        # Read a user
        user = await user_crud.get(db, id=new_user.id)
        # Update a user
        await user_crud.update(db, UpdateUserSchema(email="alice@example.com"), id=new_user.id)
        # Delete a user
        await user_crud.delete(db, id=new_user.id)
        ```

        Example 2: Advanced Filtering and Pagination
        --------------------------------------------

        Use advanced filtering, sorting, and pagination for fetching records.

        ```python
        product_crud = FastCRUD(Product)
        products = await product_crud.get_multi(
            db,
            offset=0,
            limit=10,
            sort_columns=['price'],
            sort_orders=['asc'],
        )
        ```

        Example 3: Join Operations with Custom Schemas
        ----------------------------------------------

        Perform join operations between two models using custom schemas for selection.

        ```python
        order_crud = FastCRUD(Order)
        orders = await order_crud.get_multi_joined(
            db,
            schema_to_select=ReadOrderSchema,
            join_model=Product,
            join_prefix="product_",
            join_schema_to_select=ReadProductSchema,
            offset=0,
            limit=5,
        )
        ```

        Example 4: Cursor Pagination
        ----------------------------

        Implement cursor-based pagination for efficient data retrieval in large datasets.

        ```python
        class Comment(Base):
            id = Column(Integer, primary_key=True)
            user_id = Column(Integer, ForeignKey("user.id"))
            subject = Column(String)
            body = Column(String)

        comment_crud = FastCRUD(Comment)

        first_page = await comment_crud.get_multi_by_cursor(db, limit=10)
        next_cursor = first_page['next_cursor']
        second_page = await comment_crud.get_multi_by_cursor(db, cursor=next_cursor, limit=10)
        ```

        Example 5: Dynamic Filtering and Counting
        -----------------------------------------
        Dynamically filter records based on various criteria and count the results.

        ```python
        task_crud = FastCRUD(Task)
        completed_tasks = await task_crud.get_multi(
            db,
            status='completed',
        )
        high_priority_task_count = await task_crud.count(
            db,
            priority='high',
        )
        ```

        Example 6: Using Custom Column Names for Soft Delete
        ----------------------------------------------------

        If your model uses different column names for indicating a soft delete and its timestamp, you can specify these when creating the `FastCRUD` instance.

        ```python
        --8<--
        fastcrud/examples/user/model.py:model_common
        --8<--
            ...
        --8<--
        fastcrud/examples/user/model.py:model_archived
        --8<--


        custom_user_crud = FastCRUD(
            User,
            is_deleted_column="archived",
            deleted_at_column="archived_at",
        )
        # Now 'archived' and 'archived_at' will be used for soft delete operations.
        ```
    """

    def __init__(
        self,
        model: type[ModelType],
        is_deleted_column: str = "is_deleted",
        deleted_at_column: str = "deleted_at",
        updated_at_column: str = "updated_at",
        multi_response_key: str = "data",
    ) -> None:
        self.model = model
        self.model_col_names = [col.key for col in model.__table__.columns]
        self.is_deleted_column = is_deleted_column
        self.deleted_at_column = deleted_at_column
        self.updated_at_column = updated_at_column
        self.multi_response_key = multi_response_key
        self._primary_keys = get_primary_key_columns(self.model)
        self._filter_processor = FilterProcessor(self.model)
        self._query_builder = SQLQueryBuilder(self.model)

    @overload
    async def create(
        self,
        db: AsyncSession,
        object: CreateSchemaType,
        *,
        commit: bool = True,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[True],
    ) -> SelectSchemaType: ...

    @overload
    async def create(
        self,
        db: AsyncSession,
        object: CreateSchemaType,
        *,
        commit: bool = True,
        schema_to_select: None = None,
        return_as_model: Literal[False] = False,
    ) -> None: ...

    @overload
    async def create(
        self,
        db: AsyncSession,
        object: CreateSchemaType,
        *,
        commit: bool = True,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[False] = False,
    ) -> dict[str, Any]: ...

    @overload
    async def create(
        self,
        db: AsyncSession,
        object: CreateSchemaType,
        *,
        commit: bool = True,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
    ) -> Union[None, SelectSchemaType, dict[str, Any]]: ...

    async def create(
        self,
        db: AsyncSession,
        object: CreateSchemaType,
        commit: bool = True,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
    ) -> Union[None, SelectSchemaType, dict[str, Any]]:
        """
        Create a new record in the database.

        Args:
            db: The SQLAlchemy async session.
            object: The Pydantic schema containing the data to be saved.
            commit: If `True`, commits the transaction immediately. Default is `True`.
            schema_to_select: Pydantic schema for selecting specific columns.
            return_as_model: If `True`, returns data as an instance of `schema_to_select`.

        Returns:
            The created database record or None:
            - When `schema_to_select` is None: `None` (v0.20.0 behavior)
            - When `return_as_model=True` and `schema_to_select` is provided: `SelectSchemaType`
            - When `return_as_model=False` and `schema_to_select` is provided: `Dict[str, Any]`
        """
        if return_as_model and not schema_to_select:
            raise ValueError(
                "schema_to_select must be provided when return_as_model is True."
            )

        object_dict = object.model_dump()
        db_object: ModelType = self.model(**object_dict)
        db.add(db_object)

        if commit:
            await db.commit()
            await db.refresh(db_object)
        else:
            await db.flush()
            await db.refresh(db_object)

        if not schema_to_select:
            return None

        data_dict = {
            col.key: getattr(db_object, col.key) for col in db_object.__table__.columns
        }
        if not return_as_model:
            return data_dict
        return schema_to_select(**data_dict)

    async def select(
        self,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> Select:
        """
        Constructs a SQL Alchemy `Select` statement with optional column selection, filtering, and sorting.

        This method allows for advanced filtering through comparison operators, enabling queries to be refined beyond simple equality checks.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            schema_to_select: Pydantic schema to determine which columns to include in the selection. If not provided, selects all columns of the model.
            sort_columns: A single column name or list of column names to sort the query results by. Must be used in conjunction with `sort_orders`.
            sort_orders: A single sort order (`"asc"` or `"desc"`) or a list of sort orders, corresponding to each column in `sort_columns`. If not specified, defaults to ascending order for all `sort_columns`.
            **kwargs: Filters to apply to the query, including advanced comparison operators for more detailed querying.

        Returns:
            An SQL Alchemy `Select` statement object that can be executed or further modified.

        Examples:
            Selecting specific columns with filtering and sorting:

            ```python
            stmt = await user_crud.select(
                schema_to_select=ReadUserSchema,
                sort_columns=['age', 'name'],
                sort_orders=['asc', 'desc'],
                age__gt=18,
            )
            ```

            Creating a statement to select all users without any filters:

            ```python
            stmt = await user_crud.select()
            ```

            Selecting users with a specific `role`, ordered by `name`:

            ```python
            stmt = await user_crud.select(
                schema_to_select=UserReadSchema,
                sort_columns='name',
                role='admin',
            )
            ```

        Note:
            This method does not execute the generated SQL statement.
            Use `db.execute(stmt)` to run the query and fetch results.
        """
        to_select = extract_matching_columns_from_schema(
            model=self.model, schema=schema_to_select
        )
        filters = self._filter_processor.parse_filters(**kwargs)
        stmt = select(*to_select).filter(*filters)

        if sort_columns:
            stmt = self._query_builder.apply_sorting(stmt, sort_columns, sort_orders)
        return stmt

    @overload
    async def get(
        self,
        db: AsyncSession,
        *,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[True],
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[SelectSchemaType]: ...

    @overload
    async def get(
        self,
        db: AsyncSession,
        *,
        schema_to_select: None = None,
        return_as_model: Literal[False] = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def get(
        self,
        db: AsyncSession,
        *,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[False] = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def get(
        self,
        db: AsyncSession,
        *,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[Union[dict[str, Any], SelectSchemaType]]: ...

    async def get(
        self,
        db: AsyncSession,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[Union[dict[str, Any], SelectSchemaType]]:
        """
        Fetches a single record based on specified filters.

        This method allows for advanced filtering through comparison operators, enabling queries to be refined beyond simple equality checks.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The database session to use for the operation.
            schema_to_select: Optional Pydantic schema for selecting specific columns.
            return_as_model: If `True`, converts the fetched data to Pydantic models based on `schema_to_select`. Defaults to `False`.
            one_or_none: Flag to get strictly one or no result. Multiple results are not allowed.
            **kwargs: Filters to apply to the query, using field names for direct matches or appending comparison operators for advanced queries.

        Raises:
            ValueError: If `return_as_model` is `True` but `schema_to_select` is not provided.

        Returns:
            A dictionary or a Pydantic model instance of the fetched database row, or `None` if no match is found:

            - When `return_as_model=True` and `schema_to_select` is provided: `Optional[SelectSchemaType]`
            - When `return_as_model=False`: `Optional[Dict[str, Any]]`

        Examples:
            Fetch a user by ID:

            ```python
            user = await user_crud.get(db, id=1)
            ```

            Fetch a user with an age greater than 30:

            ```python
            user = await user_crud.get(db, age__gt=30)
            ```

            Fetch a user with a registration date before Jan 1, 2020:

            ```python
            user = await user_crud.get(db, registration_date__lt=datetime(2020, 1, 1))
            ```

            Fetch a user not equal to a specific username:

            ```python
            user = await user_crud.get(db, username__ne='admin')
            ```
        """
        stmt = await self.select(schema_to_select=schema_to_select, **kwargs)

        db_row = await db.execute(stmt)
        result: Optional[Row] = db_row.one_or_none() if one_or_none else db_row.first()
        if result is None:
            return None
        out: dict = dict(result._mapping)
        if not return_as_model:
            return out
        if not schema_to_select:
            raise ValueError(
                "schema_to_select must be provided when return_as_model is True."
            )
        return schema_to_select(**out)

    def _get_pk_dict(self, instance):
        return {pk.name: getattr(instance, pk.name) for pk in self._primary_keys}

    @overload
    async def upsert(
        self,
        db: AsyncSession,
        instance: Union[UpdateSchemaType, CreateSchemaType],
        *,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[True],
    ) -> Optional[SelectSchemaType]: ...

    @overload
    async def upsert(
        self,
        db: AsyncSession,
        instance: Union[UpdateSchemaType, CreateSchemaType],
        *,
        schema_to_select: None = None,
        return_as_model: Literal[False] = False,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def upsert(
        self,
        db: AsyncSession,
        instance: Union[UpdateSchemaType, CreateSchemaType],
        *,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[False] = False,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def upsert(
        self,
        db: AsyncSession,
        instance: Union[UpdateSchemaType, CreateSchemaType],
        *,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
    ) -> Union[SelectSchemaType, dict[str, Any], None]: ...

    async def upsert(
        self,
        db: AsyncSession,
        instance: Union[UpdateSchemaType, CreateSchemaType],
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
    ) -> Union[SelectSchemaType, dict[str, Any], None]:
        """Update the instance or create it if it doesn't exists.

        Note: This method will perform two transactions to the database (get and create or update).

        Args:
            db: The database session to use for the operation.
            instance: A Pydantic schema representing the instance.
            schema_to_select: Optional Pydantic schema for selecting specific columns. Defaults to `None`.
            return_as_model: If `True`, converts the fetched data to Pydantic models based on `schema_to_select`. Defaults to `False`.

        Returns:
            The created or updated instance:

            - When `return_as_model=True` and `schema_to_select` is provided: `SelectSchemaType`
            - When `return_as_model=False`: `Dict[str, Any]`
        """
        _pks = self._get_pk_dict(instance)
        schema_to_select = schema_to_select or type(instance)  # type: ignore
        db_instance = await self.get(
            db,
            schema_to_select=schema_to_select,  # type: ignore
            return_as_model=return_as_model,
            **_pks,
        )
        if db_instance is None:
            db_instance = await self.create(
                db,
                instance,  # type: ignore
                schema_to_select=schema_to_select,  # type: ignore
                return_as_model=return_as_model,
            )
        else:
            await self.update(db, instance)  # type: ignore
            db_instance = await self.get(
                db,
                schema_to_select=schema_to_select,  # type: ignore
                return_as_model=return_as_model,
                **_pks,
            )

        return db_instance

    @overload
    async def upsert_multi(
        self,
        db: AsyncSession,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        *,
        commit: bool = False,
        return_columns: Optional[list[str]] = None,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[True],
        update_override: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[UpsertMultiResponseModel[SelectSchemaType]]: ...

    @overload
    async def upsert_multi(
        self,
        db: AsyncSession,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        *,
        commit: bool = False,
        return_columns: Optional[list[str]] = None,
        schema_to_select: None = None,
        return_as_model: Literal[False] = False,
        update_override: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[UpsertMultiResponseDict]: ...

    @overload
    async def upsert_multi(
        self,
        db: AsyncSession,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        *,
        commit: bool = False,
        return_columns: Optional[list[str]] = None,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[False] = False,
        update_override: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[UpsertMultiResponseDict]: ...

    @overload
    async def upsert_multi(
        self,
        db: AsyncSession,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        *,
        commit: bool = False,
        return_columns: Optional[list[str]] = None,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        update_override: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[
        Union[UpsertMultiResponseDict, UpsertMultiResponseModel[SelectSchemaType]]
    ]: ...

    async def upsert_multi(
        self,
        db: AsyncSession,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        commit: bool = False,
        return_columns: Optional[list[str]] = None,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        update_override: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[
        Union[UpsertMultiResponseDict, UpsertMultiResponseModel[SelectSchemaType]]
    ]:
        """
        Upsert multiple records in the database. The underlying implementation varies based on the database dialect.

        Args:
            db: The database session to use for the operation.
            instances: A list of Pydantic schemas representing the instances to upsert.
            commit: If True, commits the transaction immediately. Default is False.
            return_columns: Optional list of column names to return after the upsert operation.
            schema_to_select: Optional Pydantic schema for selecting specific columns. Required if return_as_model is True.
            return_as_model: If True, returns data as instances of the specified Pydantic model.
            update_override: Optional dictionary to override the update values for the upsert operation.
            **kwargs: Filters to identify the record(s) to update on conflict, supporting advanced comparison operators for refined querying.

        Returns:
            The upserted records as a dictionary containing the operation results:

            - When `return_as_model=True` and `schema_to_select` is provided: `UpsertMultiResponseModel[SelectSchemaType]`
              (`Dict[str, List[SelectSchemaType]]`)
            - When `return_as_model=False`: `UpsertMultiResponseDict`
              (`Dict[str, List[Dict[str, Any]]]`)

            The dictionary contains keys like "updated" and "created" with lists of corresponding records.

        Raises:
            ValueError: If the MySQL dialect is used with filters, return_columns, schema_to_select, or return_as_model.
            NotImplementedError: If the database dialect is not supported for upsert multi.
        """
        if update_override is None:
            update_override = {}
        filters = self._filter_processor.parse_filters(**kwargs)

        if db.bind.dialect.name == "postgresql":
            statement, params = await upsert_multi_postgresql(
                self.model,
                [pk.name for pk in self._primary_keys],
                instances,
                filters,
                update_override,
            )
        elif db.bind.dialect.name == "sqlite":
            statement, params = await upsert_multi_sqlite(
                self.model,
                [pk.name for pk in self._primary_keys],
                instances,
                filters,
                update_override,
            )
        elif db.bind.dialect.name in ["mysql", "mariadb"]:
            if filters:
                raise ValueError(
                    "MySQL does not support filtering on insert operations."
                )
            if return_columns or schema_to_select or return_as_model:
                raise ValueError(
                    "MySQL does not support the returning clause for insert operations."
                )
            statement, params = await upsert_multi_mysql(
                self.model,
                instances,
                update_override,
                self.deleted_at_column,
            )
        else:  # pragma: no cover
            raise NotImplementedError(
                f"Upsert multi is not implemented for {db.bind.dialect.name}"
            )

        if return_as_model:
            return_columns = self.model_col_names

        if return_columns:
            statement = statement.returning(*[column(name) for name in return_columns])
            db_row = await db.execute(statement, params)
            if commit:
                await db.commit()
            rows_data = [dict(row) for row in db_row.mappings()]
            formatted_data = format_multi_response(
                rows_data, schema_to_select, return_as_model
            )
            return {"data": formatted_data}

        await db.execute(statement, params)
        if commit:
            await db.commit()
        return None

    async def exists(self, db: AsyncSession, **kwargs: Any) -> bool:
        """
        Checks if any records exist that match the given filter conditions.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The database session to use for the operation.
            **kwargs: Filters to apply to the query, supporting both direct matches and advanced comparison operators for refined search criteria.

        Returns:
            `True` if at least one record matches the filter conditions, `False` otherwise.

        Examples:
            Check if a user with a specific ID exists:

            ```python
            exists = await user_crud.exists(db, id=1)
            ```

            Check if any user is older than 30:

            ```python
            exists = await user_crud.exists(db, age__gt=30)
            ```

            Check if any user was registered before Jan 1, 2020:

            ```python
            exists = await user_crud.exists(db, registration_date__lt=datetime(2020, 1, 1))
            ```

            Check if a username other than `admin` exists:

            ```python
            exists = await user_crud.exists(db, username__ne='admin')
            ```
        """
        filters = self._filter_processor.parse_filters(**kwargs)
        stmt = select(self.model).filter(*filters).limit(1)

        result = await db.execute(stmt)
        return result.first() is not None

    async def count(
        self,
        db: AsyncSession,
        joins_config: Optional[list[JoinConfig]] = None,
        distinct_on_primary: bool = False,
        **kwargs: Any,
    ) -> int:
        """
        Counts records that match specified filters.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Can also count records based on a configuration of joins, useful for complex queries involving relationships.

        Args:
            db: The database session to use for the operation.
            joins_config: Optional configuration for applying joins in the count query.
            distinct_on_primary: If True, counts only distinct base model rows when using joins.
                This is particularly useful for one-to-many relationships to avoid inflated counts
                from multiple joined rows per base row. Defaults to False.
            **kwargs: Filters to apply for the count, including field names for equality checks or with comparison operators for advanced queries.

        Returns:
            The total number of records matching the filter conditions.

        Examples:
            Count users by ID:

            ```python
            count = await user_crud.count(db, id=1)
            ```

            Count users older than 30:

            ```python
            count = await user_crud.count(db, age__gt=30)
            ```

            Count users with a username other than `admin`:

            ```python
            count = await user_crud.count(db, username__ne='admin')
            ```

            Count projects with at least one participant (many-to-many relationship):

            ```python
            joins_config = [
                JoinConfig(
                    model=ProjectsParticipantsAssociation,
                    join_on=Project.id == ProjectsParticipantsAssociation.project_id,
                    join_type="inner",
                ),
                JoinConfig(
                    model=Participant,
                    join_on=ProjectsParticipantsAssociation.participant_id == Participant.id,
                    join_type="inner",
                ),
            ]
            project_crud = FastCRUD(Project)
            count = await project_crud.count(db, joins_config=joins_config)
            ```

            Count projects by a specific participant name (filter applied on a joined model):

            ```python
            joins_config = [
                JoinConfig(
                    model=ProjectsParticipantsAssociation,
                    join_on=Project.id == ProjectsParticipantsAssociation.project_id,
                    join_type="inner",
                ),
                JoinConfig(
                    model=Participant,
                    join_on=ProjectsParticipantsAssociation.participant_id == Participant.id,
                    join_type="inner",
                    filters={'name': 'Jane Doe'},
                ),
            ]
            count = await project_crud.count(db, joins_config=joins_config)
            ```
        """
        primary_filters = self._filter_processor.parse_filters(**kwargs)

        if joins_config is not None:
            primary_keys = list(get_primary_key_names(self.model))
            if not any(primary_keys):  # pragma: no cover
                raise ValueError(
                    f"The model '{self.model.__name__}' does not have a primary key defined, which is required for counting with joins."
                )
            to_select = [
                getattr(self.model, pk).label(f"distinct_{pk}") for pk in primary_keys
            ]
            base_query = select(*to_select)
            base_query = self._query_builder.prepare_joins(
                base_query, joins_config, select_joined_columns=False
            )
            base_query = self._query_builder.apply_filters(base_query, primary_filters)

            if distinct_on_primary:
                base_query = base_query.distinct()
                subquery = base_query.subquery()
                count_query = select(func.count()).select_from(subquery)
            else:
                count_query = select(func.count()).select_from(base_query.subquery())
        else:
            count_query = select(func.count()).select_from(self.model)
            count_query = self._query_builder.apply_filters(
                count_query, primary_filters
            )

        total_count: Optional[int] = await db.scalar(count_query)
        if total_count is None:
            raise ValueError("Could not find the count.")

        return total_count

    @overload
    async def get_multi(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: Optional[int] = 100,
        schema_to_select: type[SelectSchemaType],
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        return_as_model: Literal[True],
        return_total_count: bool = True,
        **kwargs: Any,
    ) -> GetMultiResponseModel[SelectSchemaType]: ...

    @overload
    async def get_multi(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: Optional[int] = 100,
        schema_to_select: None = None,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        return_as_model: Literal[False] = False,
        return_total_count: bool = True,
        **kwargs: Any,
    ) -> GetMultiResponseDict: ...

    @overload
    async def get_multi(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: Optional[int] = 100,
        schema_to_select: type[SelectSchemaType],
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        return_as_model: Literal[False] = False,
        return_total_count: bool = True,
        **kwargs: Any,
    ) -> GetMultiResponseDict: ...

    @overload
    async def get_multi(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: Optional[int] = 100,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        return_as_model: bool = False,
        return_total_count: bool = True,
        **kwargs: Any,
    ) -> Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict]: ...

    async def get_multi(
        self,
        db: AsyncSession,
        offset: int = 0,
        limit: Optional[int] = 100,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        return_as_model: bool = False,
        return_total_count: bool = True,
        **kwargs: Any,
    ) -> Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict]:
        """
        Fetches multiple records based on filters, supporting sorting, pagination.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The database session to use for the operation.
            offset: Starting index for records to fetch, useful for pagination.
            limit: Maximum number of records to fetch in one call. Use `None` for "no limit", fetching all matching rows. Note that in order to use `limit=None`, you'll have to provide a custom endpoint to facilitate it, which you should only do if you really seriously want to allow the user to get all the data at once.
            schema_to_select: Optional Pydantic schema for selecting specific columns. Required if `return_as_model` is True.
            sort_columns: Column names to sort the results by.
            sort_orders: Corresponding sort orders (`"asc"`, `"desc"`) for each column in `sort_columns`.
            return_as_model: If `True`, returns data as instances of the specified Pydantic model.
            return_total_count: If `True`, also returns the total count of rows with the selected filters. Useful for pagination.
            **kwargs: Filters to apply to the query, including advanced comparison operators for more detailed querying.

        Returns:
            A dictionary containing the data list and optionally the total count:
            - With return_as_model=True: Dict with "data": List[SelectSchemaType]
            - With return_as_model=False: Dict with "data": List[Dict[str, Any]]
            - If return_total_count=True, includes "total_count": int

        Raises:
            ValueError: If `limit` or `offset` is negative, or if `schema_to_select` is required but not provided or invalid.

        Examples:
            Fetch the first 10 users:

            ```python
            users = await user_crud.get_multi(
                db,
                0,
                10,
            )
            ```

            Fetch next 10 users with sorted by username:

            ```python
            users = await user_crud.get_multi(
                db,
                10,
                10,
                sort_columns='username',
                sort_orders='desc',
            )
            ```

            Fetch 10 users older than 30, sorted by age in descending order:

            ```python
            users = await user_crud.get_multi(
                db,
                offset=0,
                limit=10,
                sort_columns='age',
                sort_orders='desc',
                age__gt=30,
            )
            ```

            Fetch 10 users with a registration date before Jan 1, 2020:
            ```python
            users = await user_crud.get_multi(
                db,
                offset=0,
                limit=10,
                registration_date__lt=datetime(2020, 1, 1),
            )
            ```

            Fetch 10 users with a username other than `admin`, returning as model instances (ensure appropriate schema is passed):

            ```python
            users = await user_crud.get_multi(
                db,
                offset=0,
                limit=10,
                schema_to_select=ReadUserSchema,
                return_as_model=True,
                username__ne='admin',
            )
            ```

            Fetch users with filtering and multiple column sorting:

            ```python
            users = await user_crud.get_multi(
                db,
                0,
                10,
                sort_columns=['username', 'email'],
                sort_orders=['asc', 'desc'],
                is_active=True,
            )
            ```
        """
        validate_pagination_params(offset, limit)
        regular_filters, joined_filters_info = (
            self._filter_processor.separate_joined_filters(**kwargs)
        )

        if joined_filters_info:
            return await handle_joined_filters_delegation(
                crud_instance=cast(CRUDInstance, self),
                joined_filters_info=joined_filters_info,
                db=db,
                offset=offset,
                limit=limit,
                schema_to_select=schema_to_select,
                sort_columns=sort_columns,
                sort_orders=sort_orders,
                return_as_model=return_as_model,
                return_total_count=return_total_count,
                **regular_filters,
            )

        stmt = await self.select(
            schema_to_select=schema_to_select,
            sort_columns=sort_columns,
            sort_orders=sort_orders,
            **kwargs,
        )

        stmt = self._query_builder.apply_pagination(stmt, offset, limit)
        result = await db.execute(stmt)
        data = [dict(row) for row in result.mappings()]
        formatted_data = format_multi_response(data, schema_to_select, return_as_model)

        response: dict[str, Any] = {self.multi_response_key: formatted_data}
        if return_total_count:
            total_count = await self.count(db=db, **kwargs)
            response["total_count"] = total_count

        return cast(
            Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict],
            response,
        )

    @overload
    async def get_joined(
        self,
        db: AsyncSession,
        *,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[True],
        join_model: Optional[ModelType] = None,
        join_on: Optional[Union[Join, BinaryExpression]] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass] = None,
        join_filters: Optional[dict] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        nest_joins: bool = False,
        relationship_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[SelectSchemaType]: ...

    @overload
    async def get_joined(
        self,
        db: AsyncSession,
        *,
        schema_to_select: None = None,
        return_as_model: Literal[False] = False,
        join_model: Optional[ModelType] = None,
        join_on: Optional[Union[Join, BinaryExpression]] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass] = None,
        join_filters: Optional[dict] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        nest_joins: bool = False,
        relationship_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def get_joined(
        self,
        db: AsyncSession,
        *,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[False] = False,
        join_model: Optional[ModelType] = None,
        join_on: Optional[Union[Join, BinaryExpression]] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass] = None,
        join_filters: Optional[dict] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        nest_joins: bool = False,
        relationship_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def get_joined(
        self,
        db: AsyncSession,
        *,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        join_model: Optional[ModelType] = None,
        join_on: Optional[Union[Join, BinaryExpression]] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass] = None,
        join_filters: Optional[dict] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        nest_joins: bool = False,
        relationship_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Union[dict[str, Any], SelectSchemaType]]: ...

    async def get_joined(
        self,
        db: AsyncSession,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        join_model: Optional[ModelType] = None,
        join_on: Optional[Union[Join, BinaryExpression]] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass] = None,
        join_filters: Optional[dict] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        nest_joins: bool = False,
        relationship_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Union[dict[str, Any], SelectSchemaType]]:
        """
        Fetches a single record with one or multiple joins on other models. If `join_on` is not provided, the method attempts
        to automatically detect the join condition using foreign key relationships. For multiple joins, use `joins_config` to
        specify each join configuration.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The SQLAlchemy async session.
            schema_to_select: Pydantic schema for selecting specific columns from the primary model. Required if `return_as_model` is True.
            return_as_model: If `True`, returns data as a Pydantic model instance based on `schema_to_select`. Defaults to `False`.
            join_model: The model to join with.
            join_on: SQLAlchemy Join object for specifying the `ON` clause of the join. If `None`, the join condition is auto-detected based on foreign keys.
            join_prefix: Optional prefix to be added to all columns of the joined model. If `None`, no prefix is added.
            join_schema_to_select: Pydantic schema for selecting specific columns from the joined model.
            join_type: Specifies the type of join operation to perform. Can be `"left"` for a left outer join or `"inner"` for an inner join.
            alias: An instance of `AliasedClass` for the join model, useful for self-joins or multiple joins on the same model. Result of `aliased(join_model)`.
            join_filters: Filters applied to the joined model, specified as a dictionary mapping column names to their expected values.
            joins_config: A list of `JoinConfig` instances, each specifying a model to join with, join condition, optional prefix for column names, schema for selecting specific columns, and the type of join. This parameter enables support for multiple joins.
            nest_joins: If `True`, nested data structures will be returned where joined model data are nested under the `join_prefix` as a dictionary.
            relationship_type: Specifies the relationship type, such as `"one-to-one"` or `"one-to-many"`. Used to determine how to nest the joined data. If `None`, uses `"one-to-one"`.
            **kwargs: Filters to apply to the primary model query, supporting advanced comparison operators for refined searching.

        Returns:
            A dictionary or Pydantic model instance representing the joined record, or `None` if no record matches the criteria:

            - When `return_as_model=True` and `schema_to_select` is provided: `Optional[SelectSchemaType]`
            - When `return_as_model=False`: `Optional[Dict[str, Any]]`

        Raises:
            ValueError: If both single join parameters and `joins_config` are used simultaneously.
            ArgumentError: If any provided model in `joins_config` is not recognized or invalid.
            NoResultFound: If no record matches the criteria with the provided filters.

        Examples:
            Simple example: Joining `User` and `Tier` models without explicitly providing `join_on`

            ```python
            result = await user_crud.get_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_schema_to_select=ReadTierSchema,
            )
            ```

            Fetch a user and their associated tier, filtering by user ID:

            ```python
            result = await user_crud.get_joined(
                db,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_schema_to_select=ReadTierSchema,
                id=1,
            )
            ```

            Fetch a user and their associated tier, where the user's age is greater than 30:

            ```python
            result = await user_crud.get_joined(
                db,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_schema_to_select=ReadTierSchema,
                age__gt=30,
            )
            ```

            Fetch a user and their associated tier, excluding users with the `admin` username:

            ```python
            result = await user_crud.get_joined(
                db,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_schema_to_select=ReadTierSchema,
                username__ne='admin',
            )
            ```

            Complex example: Joining with a custom join condition, additional filter parameters, and a prefix

            ```python
            from sqlalchemy import and_
            result = await user_crud.get_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_on=and_(User.tier_id == Tier.id, User.is_superuser == True),
                join_prefix="tier_",
                join_schema_to_select=ReadTierSchema,
                username="john_doe",
            )
            ```

            Example of using `joins_config` for multiple joins:

            ```python
            from fastcrud import JoinConfig

            # Using same User/Tier/Department models/schemas as above.

            result = await user_crud.get_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                joins_config=[
                    JoinConfig(
                        model=Tier,
                        join_on=User.tier_id == Tier.id,
                        join_prefix="tier_",
                        schema_to_select=ReadTierSchema,
                        join_type="left",
                    ),
                    JoinConfig(
                        model=Department,
                        join_on=User.department_id == Department.id,
                        join_prefix="dept_",
                        schema_to_select=ReadDepartmentSchema,
                        join_type="inner",
                    ),
                ],
            )
            ```

            Using `alias` for joining the same model multiple times:
            ```python
            from fastcrud import aliased

            owner_alias = aliased(ModelTest, name="owner")
            user_alias = aliased(ModelTest, name="user")

            result = await crud.get_joined(
                db=session,
                schema_to_select=BookingSchema,
                joins_config=[
                    JoinConfig(
                        model=ModelTest,
                        join_on=BookingModel.owner_id == owner_alias.id,
                        join_prefix="owner_",
                        alias=owner_alias,
                        schema_to_select=UserSchema,
                    ),
                    JoinConfig(
                        model=ModelTest,
                        join_on=BookingModel.user_id == user_alias.id,
                        join_prefix="user_",
                        alias=user_alias,
                        schema_to_select=UserSchema,
                    ),
                ],
                id=1,
            )
            ```

            Fetching a single project and its associated participants where a participant has a specific role:

            ```python
            joins_config = [
                JoinConfig(
                    model=ProjectsParticipantsAssociation,
                    join_on=Project.id == ProjectsParticipantsAssociation.project_id,
                    join_type="inner",
                ),
                JoinConfig(
                    model=Participant,
                    join_on=ProjectsParticipantsAssociation.participant_id == Participant.id,
                    join_type="inner",
                    filters={'role': 'Designer'},
                ),
            ]

            project_crud = FastCRUD(Project)

            project = await project_crud.get_joined(
                db=session,
                schema_to_select=ReadProjectSchema,
                joins_config=joins_config,
            )
            ```

            Example of using `joins_config` for multiple joins with nested joins enabled:

            ```python
            from fastcrud import JoinConfig

            result = await user_crud.get_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                joins_config=[
                    JoinConfig(
                        model=Tier,
                        join_on=User.tier_id == Tier.id,
                        join_prefix="tier_",
                        schema_to_select=ReadTierSchema,
                        join_type="left",
                    ),
                    JoinConfig(
                        model=Department,
                        join_on=User.department_id == Department.id,
                        join_prefix="dept_",
                        schema_to_select=ReadDepartmentSchema,
                        join_type="inner",
                    ),
                ],
                nest_joins=True,
            )
            # Expect 'result' to have 'tier' and 'dept' as nested dictionaries
            ```

            Example using one-to-one relationship:

            ```python
            author_crud = FastCRUD(Author)
            result = await author_crud.get_joined(
                db=session,
                schema_to_select=ReadAuthorSchema,
                join_model=Profile,
                join_on=Author.profile_id == Profile.id,
                join_schema_to_select=ReadProfileSchema,
                nest_joins=True,
                relationship_type='one-to-one', # note that this is the default behavior
            )
            # Expect 'result' to have 'profile' as a nested dictionary
            ```

            Example using one-to-many relationship:

            ```python
            result = await author_crud.get_joined(
                db=session,
                schema_to_select=ReadAuthorSchema,
                join_model=Article,
                join_on=Author.id == Article.author_id,
                join_schema_to_select=ReadArticleSchema,
                nest_joins=True,
                relationship_type='one-to-many',
            )
            # Expect 'result' to have 'posts' as a nested list of dictionaries
            ```
        """
        if joins_config and (
            join_model or join_prefix or join_on or join_schema_to_select or alias
        ):
            raise ValueError(
                "Cannot use both single join parameters and joins_config simultaneously."
            )
        elif not joins_config and not join_model:
            raise ValueError("You need one of join_model or joins_config.")

        primary_select = extract_matching_columns_from_schema(
            model=self.model,
            schema=schema_to_select,
        )
        stmt = self._query_builder.build_base_select(primary_select)

        join_definitions = joins_config if joins_config else []
        if join_model:
            join_definitions.append(
                JoinConfig(
                    model=join_model,
                    join_on=join_on
                    if join_on is not None
                    else auto_detect_join_condition(self.model, join_model),
                    join_prefix=join_prefix,
                    schema_to_select=join_schema_to_select,
                    join_type=join_type,
                    alias=alias,
                    filters=join_filters,
                    relationship_type=relationship_type,
                )
            )

        stmt = self._query_builder.prepare_joins(
            stmt=stmt, joins_config=join_definitions, use_temporary_prefix=nest_joins
        )
        primary_filters = self._filter_processor.parse_filters(**kwargs)
        stmt = self._query_builder.apply_filters(stmt, primary_filters)

        db_rows = await db.execute(stmt)
        if any(join.relationship_type == "one-to-many" for join in join_definitions):
            if nest_joins is False:  # pragma: no cover
                raise ValueError(
                    "Cannot use one-to-many relationship with nest_joins=False"
                )
            results = db_rows.fetchall()
            data_list = [dict(row._mapping) for row in results]
        else:
            result = db_rows.first()
            if result is not None:
                data_list = [dict(result._mapping)]
            else:
                data_list = []

        processed_data = process_joined_data(
            data_list, join_definitions, nest_joins, self.model
        )

        if processed_data is None or not return_as_model:
            return processed_data

        if not schema_to_select:
            raise ValueError(
                "schema_to_select must be provided when return_as_model is True."
            )

        return schema_to_select(**processed_data)

    @overload
    async def get_multi_joined(
        self,
        db: AsyncSession,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[True],
        join_model: Optional[type[ModelType]] = None,
        join_on: Optional[Any] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass[Any]] = None,
        join_filters: Optional[dict] = None,
        nest_joins: bool = False,
        offset: int = 0,
        limit: Optional[int] = 100,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        counts_config: Optional[list[CountConfig]] = None,
        return_total_count: bool = True,
        relationship_type: Optional[str] = None,
        nested_schema_to_select: Optional[dict[str, type[SelectSchemaType]]] = None,
        **kwargs: Any,
    ) -> GetMultiResponseModel[SelectSchemaType]: ...

    @overload
    async def get_multi_joined(
        self,
        db: AsyncSession,
        schema_to_select: None = None,
        return_as_model: Literal[False] = False,
        join_model: Optional[type[ModelType]] = None,
        join_on: Optional[Any] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass[Any]] = None,
        join_filters: Optional[dict] = None,
        nest_joins: bool = False,
        offset: int = 0,
        limit: Optional[int] = 100,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        counts_config: Optional[list[CountConfig]] = None,
        return_total_count: bool = True,
        relationship_type: Optional[str] = None,
        nested_schema_to_select: Optional[dict[str, type[SelectSchemaType]]] = None,
        **kwargs: Any,
    ) -> GetMultiResponseDict: ...

    @overload
    async def get_multi_joined(
        self,
        db: AsyncSession,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[False] = False,
        join_model: Optional[type[ModelType]] = None,
        join_on: Optional[Any] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass[Any]] = None,
        join_filters: Optional[dict] = None,
        nest_joins: bool = False,
        offset: int = 0,
        limit: Optional[int] = 100,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        counts_config: Optional[list[CountConfig]] = None,
        return_total_count: bool = True,
        relationship_type: Optional[str] = None,
        nested_schema_to_select: Optional[dict[str, type[SelectSchemaType]]] = None,
        **kwargs: Any,
    ) -> GetMultiResponseDict: ...

    @overload
    async def get_multi_joined(
        self,
        db: AsyncSession,
        *,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        join_model: Optional[type[ModelType]] = None,
        join_on: Optional[Any] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass[Any]] = None,
        join_filters: Optional[dict] = None,
        nest_joins: bool = False,
        offset: int = 0,
        limit: Optional[int] = 100,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        counts_config: Optional[list[CountConfig]] = None,
        return_total_count: bool = True,
        relationship_type: Optional[str] = None,
        nested_schema_to_select: Optional[dict[str, type[SelectSchemaType]]] = None,
        **kwargs: Any,
    ) -> Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict]: ...

    async def get_multi_joined(
        self,
        db: AsyncSession,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        join_model: Optional[type[ModelType]] = None,
        join_on: Optional[Any] = None,
        join_prefix: Optional[str] = None,
        join_schema_to_select: Optional[type[SelectSchemaType]] = None,
        join_type: str = "left",
        alias: Optional[AliasedClass[Any]] = None,
        join_filters: Optional[dict] = None,
        nest_joins: bool = False,
        offset: int = 0,
        limit: Optional[int] = 100,
        sort_columns: Optional[Union[str, list[str]]] = None,
        sort_orders: Optional[Union[str, list[str]]] = None,
        joins_config: Optional[list[JoinConfig]] = None,
        counts_config: Optional[list[CountConfig]] = None,
        return_total_count: bool = True,
        relationship_type: Optional[str] = None,
        nested_schema_to_select: Optional[dict[str, type[SelectSchemaType]]] = None,
        **kwargs: Any,
    ) -> Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict]:
        """
        Fetch multiple records with a join on another model, allowing for pagination, optional sorting, and model conversion.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The SQLAlchemy async session.
            schema_to_select: Pydantic schema for selecting specific columns from the primary model. Required if `return_as_model` is True.
            join_model: The model to join with.
            join_on: SQLAlchemy Join object for specifying the `ON` clause of the join. If `None`, the join condition is auto-detected based on foreign keys.
            join_prefix: Optional prefix to be added to all columns of the joined model. If `None`, no prefix is added.
            join_schema_to_select: Pydantic schema for selecting specific columns from the joined model.
            join_type: Specifies the type of join operation to perform. Can be `"left"` for a left outer join or `"inner"` for an inner join.
            alias: An instance of `AliasedClass` for the join model, useful for self-joins or multiple joins on the same model. Result of `aliased(join_model)`.
            join_filters: Filters applied to the joined model, specified as a dictionary mapping column names to their expected values.
            nest_joins: If `True`, nested data structures will be returned where joined model data are nested under the `join_prefix` as a dictionary.
            offset: The offset (number of records to skip) for pagination.
            limit: Maximum number of records to fetch in one call. Use `None` for "no limit", fetching all matching rows. Note that in order to use `limit=None`, you'll have to provide a custom endpoint to facilitate it, which you should only do if you really seriously want to allow the user to get all the data at once.
            sort_columns: A single column name or a list of column names on which to apply sorting.
            sort_orders: A single sort order (`"asc"` or `"desc"`) or a list of sort orders corresponding to the columns in `sort_columns`. If not provided, defaults to `"asc"` for each column.
            return_as_model: If `True`, converts the fetched data to Pydantic models based on `schema_to_select`. Defaults to `False`.
            joins_config: List of `JoinConfig` instances for specifying multiple joins. Each instance defines a model to join with, join condition, optional prefix for column names, schema for selecting specific columns, and join type.
            counts_config: List of `CountConfig` instances for counting related objects. Each instance defines a model to count, join condition, and optional alias for the count column. Useful for many-to-many relationships.
            return_total_count: If `True`, also returns the total count of rows with the selected filters. Useful for pagination.
            relationship_type: Specifies the relationship type, such as `"one-to-one"` or `"one-to-many"`. Used to determine how to nest the joined data. If `None`, uses `"one-to-one"`.
            nested_schema_to_select: A dictionary mapping join prefixes to their corresponding Pydantic schemas for nested data conversion. If not provided, schemas are auto-detected from `joins_config`.
            **kwargs: Filters to apply to the primary query, including advanced comparison operators for refined searching.

        Returns:
            A dictionary containing the fetched rows under `"data"` key and total count under `"total_count"`:

            - When `return_as_model=True` and `schema_to_select` is provided: `GetMultiResponseModel[SelectSchemaType]`
              (`Dict[str, Union[List[SelectSchemaType], int]]`)
            - When `return_as_model=False`: `GetMultiResponseDict`
              (`Dict[str, Union[List[Dict[str, Any]], int]]`)

        Raises:
            ValueError: If either `limit` or `offset` are negative, or if `schema_to_select` is required but not provided or invalid.
                        Also if both `joins_config` and any of the single join parameters are provided or none of `joins_config` and `join_model` is provided.

        Examples:
            Fetching multiple `User` records joined with `Tier` records, using left join, returning raw data:

            ```python
            users = await user_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_prefix="tier_",
                join_schema_to_select=ReadTierSchema,
                offset=0,
                limit=10,
            )
            ```

            Fetch users joined with their tiers, sorted by username, where user's age is greater than 30:

            ```python
            users = await user_crud.get_multi_joined(
                db,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_schema_to_select=ReadTierSchema,
                sort_columns='username',
                sort_orders='asc',
                age__gt=30,
            )
            ```

            Fetch users joined with their tiers, excluding users with `admin` username, returning as model instances:

            ```python
            users = await user_crud.get_multi_joined(
                db,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_schema_to_select=ReadTierSchema,
                return_as_model=True,
                username__ne='admin',
            )
            ```

            Fetching and sorting by username in descending order, returning as Pydantic model:

            ```python
            users = await user_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_prefix="tier_",
                join_schema_to_select=ReadTierSchema,
                offset=0,
                limit=10,
                sort_columns=['username'],
                sort_orders=['desc'],
                return_as_model=True,
            )
            ```

            Fetching with complex conditions and custom join, returning as Pydantic model:

            ```python
            users = await user_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                join_model=Tier,
                join_on=User.tier_id == Tier.id,
                join_prefix="tier_",
                join_schema_to_select=ReadTierSchema,
                offset=0,
                limit=10,
                return_as_model=True,
                is_active=True,
            )
            ```

            Example using `joins_config` for multiple joins:

            ```python
            from fastcrud import JoinConfig

            users = await user_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadUserSchema,
                joins_config=[
                    JoinConfig(
                        model=Tier,
                        join_on=User.tier_id == Tier.id,
                        join_prefix="tier_",
                        schema_to_select=ReadTierSchema,
                        join_type="left",
                    ),
                    JoinConfig(
                        model=Department,
                        join_on=User.department_id == Department.id,
                        join_prefix="dept_",
                        schema_to_select=ReadDepartmentSchema,
                        join_type="inner",
                    ),
                ],
                offset=0,
                limit=10,
                sort_columns='username',
                sort_orders='asc',
            )
            ```

            Example using `alias` for multiple joins, with pagination, sorting, and model conversion:
            ```python
            from fastcrud import JoinConfig, FastCRUD, aliased

            # Aliasing for self-joins or multiple joins on the same table
            owner_alias = aliased(ModelTest, name="owner")
            user_alias = aliased(ModelTest, name="user")

            # Initialize your FastCRUD instance for BookingModel
            crud = FastCRUD(BookingModel)

            result = await crud.get_multi_joined(
                db=session,
                schema_to_select=BookingSchema,  # Primary model schema
                joins_config=[
                    JoinConfig(
                        model=ModelTest,
                        join_on=BookingModel.owner_id == owner_alias.id,
                        join_prefix="owner_",
                        schema_to_select=UserSchema,  # Schema for the joined model
                        alias=owner_alias,
                    ),
                    JoinConfig(
                        model=ModelTest,
                        join_on=BookingModel.user_id == user_alias.id,
                        join_prefix="user_",
                        schema_to_select=UserSchema,
                        alias=user_alias,
                    )
                ],
                offset=10,  # Skip the first 10 records
                limit=5,  # Fetch up to 5 records
                sort_columns=['booking_date'],  # Sort by booking_date
                sort_orders=['desc'],  # In descending order
            )
            ```

            Fetching multiple project records and their associated participants where participants have a specific role:

            ```python
            joins_config = [
                JoinConfig(
                    model=ProjectsParticipantsAssociation,
                    join_on=Project.id == ProjectsParticipantsAssociation.project_id,
                    join_type="inner",
                ),
                JoinConfig(
                    model=Participant,
                    join_on=ProjectsParticipantsAssociation.participant_id == Participant.id,
                    join_type="inner",
                    filters={'role': 'Developer'},
                ),
            ]

            project_crud = FastCRUD(Project)

            projects = await project_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadProjectSchema,
                limit=10,
                joins_config=joins_config,
            )
            ```

            Fetching a list of stories, each with nested details of associated tasks and task creators, using nested joins:

            ```python
            story_crud = FastCRUD(Story)
            stories = await story_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadStorySchema,
                joins_config=[
                    JoinConfig(
                        model=Task,
                        join_on=Story.id == Task.story_id,
                        join_prefix="task_",
                        schema_to_select=ReadTaskSchema,
                        join_type="left",
                    ),
                    JoinConfig(
                        model=User,
                        join_on=Task.creator_id == User.id,
                        join_prefix="creator_",
                        schema_to_select=ReadUserSchema,
                        join_type="left",
                        alias=aliased(User, name="task_creator"),
                    ),
                ],
                nest_joins=True,
                offset=0,
                limit=5,
                sort_columns='name',
                sort_orders='asc',
            )
            ```

            Example using one-to-one relationship:

            ```python
            author_crud = FastCRUD(Author)
            results = await author_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadAuthorSchema,
                join_model=Profile,
                join_on=Author.profile_id == Profile.id,
                join_schema_to_select=ReadProfileSchema,
                nest_joins=True,
                offset=0,
                limit=10,
                relationship_type='one-to-one', # note that this is the default behavior
            )
            # Expect 'profile' to be nested as a dictionary under each user
            ```

            Example using one-to-many relationship:

            ```python
            results = await author_crud.get_multi_joined(
                db=session,
                schema_to_select=ReadAuthorSchema,
                join_model=Article,
                join_on=Author.id == Article.author_id,
                join_schema_to_select=ReadArticleSchema,
                nest_joins=True,
                offset=0,
                limit=10,
                relationship_type='one-to-many',
            )
            # Expect 'posts' to be nested as a list of dictionaries under each user
            ```

            Example using counts_config to count related objects (e.g., many-to-many relationships):

            ```python
            from fastcrud import FastCRUD, CountConfig

            search_crud = FastCRUD(Search)

            # Count videos for each search through a many-to-many relationship
            results = await search_crud.get_multi_joined(
                db=session,
                counts_config=[
                    CountConfig(
                        model=Video,
                        join_on=(Video.id == VideoSearchAssociation.video_id)
                               & (VideoSearchAssociation.search_id == Search.id),
                        alias='videos_count'
                    )
                ],
            )
            # Results will include 'videos_count' field for each search
            # Example result:
            # {
            #     "data": [
            #         {"id": 1, "term": "cats", "videos_count": 5},
            #         {"id": 2, "term": "dogs", "videos_count": 3},
            #         {"id": 3, "term": "birds", "videos_count": 0}
            #     ],
            #     "total_count": 3
            # }
            ```
        """
        config = validate_joined_query_params(
            primary_model=self.model,
            joins_config=joins_config,
            join_model=join_model,
            join_prefix=join_prefix,
            join_on=join_on,
            join_schema_to_select=join_schema_to_select,
            alias=alias,
            relationship_type=relationship_type,
            join_type=join_type,
            join_filters=join_filters,
            counts_config=counts_config,
            limit=limit,
            offset=offset,
        )

        stmt = build_joined_query(
            model=self.model,
            query_builder=self._query_builder,
            filter_processor=self._filter_processor,
            config=config,
            schema_to_select=schema_to_select,
            nest_joins=nest_joins,
            **kwargs,
        )
        raw_data = await execute_joined_query(
            db=db,
            stmt=stmt,
            query_builder=self._query_builder,
            limit=limit,
            offset=offset,
            sort_columns=sort_columns,
            sort_orders=sort_orders,
        )

        return cast(
            Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict],
            await format_joined_response(
                primary_model=self.model,
                raw_data=raw_data,
                config=config,
                schema_to_select=schema_to_select,
                return_as_model=return_as_model,
                nest_joins=nest_joins,
                return_total_count=return_total_count,
                db=db,
                nested_schema_to_select=nested_schema_to_select,
                count_func=self.count if return_total_count else None,
                **kwargs,
            ),
        )

    @overload
    async def get_multi_by_cursor(
        self,
        db: AsyncSession,
        *,
        cursor: Any = None,
        limit: int = 100,
        schema_to_select: type[SelectSchemaType],
        sort_column: str = "id",
        sort_order: str = "asc",
        return_as_model: Literal[True],
        **kwargs: Any,
    ) -> dict[str, Union[list[SelectSchemaType], Any]]: ...

    @overload
    async def get_multi_by_cursor(
        self,
        db: AsyncSession,
        *,
        cursor: Any = None,
        limit: int = 100,
        schema_to_select: None = None,
        sort_column: str = "id",
        sort_order: str = "asc",
        return_as_model: Literal[False] = False,
        **kwargs: Any,
    ) -> dict[str, Union[list[dict[str, Any]], Any]]: ...

    @overload
    async def get_multi_by_cursor(
        self,
        db: AsyncSession,
        *,
        cursor: Any = None,
        limit: int = 100,
        schema_to_select: type[SelectSchemaType],
        sort_column: str = "id",
        sort_order: str = "asc",
        return_as_model: Literal[False] = False,
        **kwargs: Any,
    ) -> dict[str, Union[list[dict[str, Any]], Any]]: ...

    @overload
    async def get_multi_by_cursor(
        self,
        db: AsyncSession,
        *,
        cursor: Any = None,
        limit: int = 100,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        sort_column: str = "id",
        sort_order: str = "asc",
        return_as_model: bool = False,
        **kwargs: Any,
    ) -> dict[str, Union[list[Union[dict[str, Any], SelectSchemaType]], Any]]: ...

    async def get_multi_by_cursor(
        self,
        db: AsyncSession,
        cursor: Any = None,
        limit: int = 100,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        sort_column: str = "id",
        sort_order: str = "asc",
        return_as_model: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Implements cursor-based pagination for fetching records. This method is designed for efficient data retrieval in large datasets and is ideal for features like infinite scrolling.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The SQLAlchemy async session.
            cursor: The cursor value to start fetching records from. Defaults to `None`.
            limit: Maximum number of rows to fetch.
            schema_to_select: Pydantic schema for selecting specific columns. Required if `return_as_model` is True.
            sort_column: Column name to use for sorting and cursor pagination.
            sort_order: Sorting direction, either `"asc"` or `"desc"`.
            return_as_model: If `True`, converts the fetched data to Pydantic models based on `schema_to_select`. Defaults to `False`.
            **kwargs: Filters to apply to the query, including advanced comparison operators for detailed querying.

        Returns:
            A dictionary containing the fetched rows under `"data"` key and the next cursor value under `"next_cursor"`:

            - When `return_as_model=True` and `schema_to_select` is provided: `Dict[str, Union[List[SelectSchemaType], Any]]`
            - When `return_as_model=False`: `Dict[str, Union[List[Dict[str, Any]], Any]]`

        Examples:
            Fetch the first set of records (e.g., the first page in an infinite scrolling scenario):

            ```python
            first_page = await user_crud.get_multi_by_cursor(
                db,
                limit=10,
                sort_column='registration_date',
            )

            # Fetch the next set of records using the cursor from the first page
            second_page = await user_crud.get_multi_by_cursor(
                db,
                cursor=next_cursor,
                limit=10,
                sort_column='registration_date',
                sort_order='desc',
            )
            ```

            Fetch records as Pydantic models with cursor pagination:

            ```python
            # Returns typed Pydantic models
            first_page = await user_crud.get_multi_by_cursor(
                db,
                schema_to_select=ReadUserSchema,
                return_as_model=True,
                limit=10,
                sort_column='registration_date',
                age__gt=30,
            )
            ```

            Fetch records excluding a specific username using cursor-based pagination:

            ```python
            result = await user_crud.get_multi_by_cursor(
                db,
                limit=10,
                sort_column='username',
                sort_order='asc',
                username__ne='admin',
            )
            ```

            Fetch records as Pydantic model instances using cursor-based pagination:

            ```python
            result = await user_crud.get_multi_by_cursor(
                db,
                limit=10,
                schema_to_select=ReadUserSchema,
                return_as_model=True,
                sort_column='created_at',
                sort_order='desc',
            )
            ```

        Note:
            This method is designed for efficient pagination in large datasets and is ideal for infinite scrolling features.
            Make sure the column used for cursor pagination is indexed for performance.
        """
        if limit == 0:
            return {"data": [], "next_cursor": None}

        stmt = await self.select(schema_to_select=schema_to_select, **kwargs)

        if cursor:
            cursor_filter = []
            if sort_order == "asc":
                cursor_filter = self._filter_processor.parse_filters(
                    **{f"{sort_column}__gt": cursor}
                )
            else:
                cursor_filter = self._filter_processor.parse_filters(
                    **{f"{sort_column}__lt": cursor}
                )
            stmt = self._query_builder.apply_filters(stmt, cursor_filter)

        stmt = self._query_builder.apply_sorting(stmt, sort_column, sort_order)
        stmt = self._query_builder.apply_pagination(stmt, 0, limit)

        result = await db.execute(stmt)
        data = [dict(row) for row in result.mappings()]
        next_cursor = data[-1][sort_column] if len(data) == limit else None

        formatted_data = format_multi_response(data, schema_to_select, return_as_model)

        return {"data": formatted_data, "next_cursor": next_cursor}

    @overload
    async def update(
        self,
        db: AsyncSession,
        object: Union[UpdateSchemaType, dict[str, Any]],
        *,
        allow_multiple: bool = False,
        commit: bool = True,
        return_columns: Optional[list[str]] = None,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[True],
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[SelectSchemaType]: ...

    @overload
    async def update(
        self,
        db: AsyncSession,
        object: Union[UpdateSchemaType, dict[str, Any]],
        *,
        allow_multiple: bool = False,
        commit: bool = True,
        return_columns: Optional[list[str]] = None,
        schema_to_select: None = None,
        return_as_model: Literal[False] = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def update(
        self,
        db: AsyncSession,
        object: Union[UpdateSchemaType, dict[str, Any]],
        *,
        allow_multiple: bool = False,
        commit: bool = True,
        return_columns: Optional[list[str]] = None,
        schema_to_select: type[SelectSchemaType],
        return_as_model: Literal[False] = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def update(
        self,
        db: AsyncSession,
        object: Union[UpdateSchemaType, dict[str, Any]],
        *,
        allow_multiple: bool = False,
        commit: bool = True,
        return_columns: Optional[list[str]] = None,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[Union[dict[str, Any], SelectSchemaType]]: ...

    async def update(
        self,
        db: AsyncSession,
        object: Union[UpdateSchemaType, dict[str, Any]],
        allow_multiple: bool = False,
        commit: bool = True,
        return_columns: Optional[list[str]] = None,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[Union[dict, SelectSchemaType]]:
        """
        Updates an existing record or multiple records in the database based on specified filters. This method allows for precise targeting of records to update.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The database session to use for the operation.
            object: A Pydantic schema or dictionary containing the update data.
            allow_multiple: If `True`, allows updating multiple records that match the filters. If `False`, raises an error if more than one record matches the filters.
            commit: If `True`, commits the transaction immediately. Default is `True`.
            return_columns: A list of column names to return after the update. If `return_as_model` is True, all columns are returned.
            schema_to_select: Pydantic schema for selecting specific columns from the updated record(s). Required if `return_as_model` is `True`.
            return_as_model: If `True`, returns the updated record(s) as Pydantic model instances based on `schema_to_select`. Default is False.
            one_or_none: If `True`, returns a single record if only one record matches the filters. Default is `False`.
            **kwargs: Filters to identify the record(s) to update, supporting advanced comparison operators for refined querying.

        Returns:
            The updated record(s) as a dictionary or Pydantic model instance or `None`:

            - When `return_as_model=True` and `schema_to_select` is provided: `Optional[SelectSchemaType]`
            - When `return_as_model=False`: `Optional[Dict[str, Any]]`

        Raises:
            MultipleResultsFound: If `allow_multiple` is `False` and more than one record matches the filters.
            NoResultFound: If no record matches the filters. (on version 0.15.3)
            ValueError: If extra fields not present in the model are provided in the update data.
            ValueError: If `return_as_model` is `True` but `schema_to_select` is not provided.

        Examples:
            Update a user's email based on their ID:

            ```python
            await user_crud.update(db, {'email': 'new_email@example.com'}, id=1)
            ```

            Update users to be inactive where age is greater than 30 and allow updates to multiple records:

            ```python
            await user_crud.update(
                db,
                {'is_active': False},
                allow_multiple=True,
                age__gt=30,
            )
            ```

            Update a user's username excluding specific user ID and prevent multiple updates:

            ```python
            await user_crud.update(
                db,
                {'username': 'new_username'},
                allow_multiple=False,
                id__ne=1,
            )
            ```

            Update a user's email and return the updated record as a Pydantic model instance:

            ```python
            user = await user_crud.update(
                db,
                {'email': 'new_email@example.com'},
                schema_to_select=ReadUserSchema,
                return_as_model=True,
                id=1,
            )
            ```

            Update a user's email and return the updated record as a dictionary:
            ```python
            user = await user_crud.update(
                db,
                {'email': 'new_email@example.com'},
                return_columns=['id', 'email'],
                id=1,
            )
            ```
        """
        await validate_update_delete_operation(
            self.count, db, allow_multiple, "update", **kwargs
        )
        update_data = prepare_update_data(
            object, self.model_col_names, self.updated_at_column, self.model
        )

        filters = self._filter_processor.parse_filters(**kwargs)
        stmt = update(self.model).filter(*filters).values(update_data)

        if return_as_model:
            return_columns = self.model_col_names

        return await execute_update_and_return_response(
            db=db,
            stmt=stmt,
            commit=commit,
            return_columns=return_columns,
            schema_to_select=schema_to_select,
            return_as_model=return_as_model,
            allow_multiple=allow_multiple,
            one_or_none=one_or_none,
        )

    async def db_delete(
        self,
        db: AsyncSession,
        allow_multiple: bool = False,
        commit: bool = True,
        filters: Optional[DeleteSchemaType] = None,
        **kwargs: Any,
    ) -> None:
        """
        Deletes a record or multiple records from the database based on specified filters.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The database session to use for the operation.
            allow_multiple: If `True`, allows deleting multiple records that match the filters. If `False`, raises an error if more than one record matches the filters.
            commit: If `True`, commits the transaction immediately. Default is `True`.
            filters: Optional Pydantic schema instance containing filters to identify the record(s) to delete.
            **kwargs: Additional filters to identify the record(s) to delete, including advanced comparison operators for detailed querying.

        Returns:
            None

        Raises:
            ValueError: If no filters are provided (to prevent accidental deletion of all records).
            MultipleResultsFound: If `allow_multiple` is `False` and more than one record matches the filters.

        Examples:
            Delete a user based on their ID using kwargs:

            ```python
            await user_crud.db_delete(db, id=1)
            ```

            Delete a user using a Pydantic schema:

            ```python
            delete_filters = DeleteUserSchema(id=1)
            await user_crud.db_delete(db, filters=delete_filters)
            ```

            Delete users older than 30 years and allow deletion of multiple records:

            ```python
            await user_crud.db_delete(
                db,
                allow_multiple=True,
                age__gt=30,
            )
            ```

            Delete a user with a specific username, ensuring only one record is deleted:

            ```python
            await user_crud.db_delete(
                db,
                allow_multiple=False,
                username='unique_username',
            )
            ```

            Combine schema filters with kwargs:

            ```python
            delete_filters = DeleteUserSchema(status='inactive')
            await user_crud.db_delete(
                db,
                filters=delete_filters,
                allow_multiple=True,
                created_at__lt=datetime(2020, 1, 1),
            )
            ```
        """
        combined_filters = {}
        if filters:
            combined_filters.update(filters.model_dump(exclude_unset=True))
        combined_filters.update(kwargs)

        if not combined_filters:
            raise ValueError(
                "No filters provided. To prevent accidental deletion of all records, at least one filter must be specified."
            )

        if (
            not allow_multiple
            and (total_count := await self.count(db, **combined_filters)) > 1
        ):
            raise MultipleResultsFound(
                f"Expected exactly one record to delete, found {total_count}."
            )

        parsed_filters = self._filter_processor.parse_filters(**combined_filters)
        stmt = delete(self.model).filter(*parsed_filters)
        await db.execute(stmt)
        if commit:
            await db.commit()

    async def delete(
        self,
        db: AsyncSession,
        db_row: Optional[Row] = None,
        allow_multiple: bool = False,
        commit: bool = True,
        filters: Optional[DeleteSchemaType] = None,
        **kwargs: Any,
    ) -> None:
        """
        Soft deletes a record or optionally multiple records if it has an `"is_deleted"` attribute, otherwise performs a hard delete, based on specified filters.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The database session to use for the operation.
            db_row: Optional existing database row to delete. If provided, the method will attempt to delete this specific row, ignoring other filters.
            allow_multiple: If `True`, allows deleting multiple records that match the filters. If `False`, raises an error if more than one record matches the filters.
            commit: If `True`, commits the transaction immediately. Default is `True`.
            filters: Optional Pydantic schema instance containing filters to identify the record(s) to delete.
            **kwargs: Additional filters to identify the record(s) to delete, supporting advanced comparison operators for refined querying.

        Raises:
            ValueError: If no filters are provided and db_row is None (to prevent accidental deletion of all records).
            MultipleResultsFound: If `allow_multiple` is `False` and more than one record matches the filters.
            NoResultFound: If no record matches the filters.

        Returns:
            None

        Examples:
            Soft delete a specific user by ID using kwargs:

            ```python
            await user_crud.delete(db, id=1)
            ```

            Soft delete a user using a Pydantic schema:

            ```python
            delete_filters = DeleteUserSchema(id=1)
            await user_crud.delete(db, filters=delete_filters)
            ```

            Soft delete users with account registration dates before 2020, allowing deletion of multiple records:

            ```python
            await user_crud.delete(
                db,
                allow_multiple=True,
                creation_date__lt=datetime(2020, 1, 1),
            )
            ```

            Soft delete a user with a specific email, ensuring only one record is deleted:

            ```python
            await user_crud.delete(
                db,
                allow_multiple=False,
                email='unique@example.com',
            )
            ```

            Combine schema filters with kwargs:

            ```python
            delete_filters = DeleteUserSchema(status='inactive')
            await user_crud.delete(
                db,
                filters=delete_filters,
                allow_multiple=True,
                last_login__lt=datetime(2023, 1, 1),
            )
            ```
        """
        combined_filters = {}
        if filters:
            combined_filters.update(filters.model_dump(exclude_unset=True))
        combined_filters.update(kwargs)

        if db_row:
            has_soft_delete = hasattr(db_row, self.is_deleted_column) and hasattr(
                db_row, self.deleted_at_column
            )
            if has_soft_delete:
                setattr(db_row, self.is_deleted_column, True)
                setattr(db_row, self.deleted_at_column, datetime.now(timezone.utc))
            else:
                await db.delete(db_row)
            if commit:
                await db.commit()
            return

        if not combined_filters:
            raise ValueError(
                "No filters provided. To prevent accidental deletion of all records, at least one filter must be specified."
            )

        await validate_update_delete_operation(
            self.count, db, allow_multiple, "delete", **combined_filters
        )

        parsed_filters = self._filter_processor.parse_filters(**combined_filters)

        update_values: dict[str, Union[bool, datetime]] = {}
        if self.deleted_at_column in self.model_col_names:
            update_values[self.deleted_at_column] = datetime.now(timezone.utc)
        if self.is_deleted_column in self.model_col_names:
            update_values[self.is_deleted_column] = True

        if update_values:
            update_stmt = (
                update(self.model).filter(*parsed_filters).values(**update_values)
            )
            await db.execute(update_stmt)

        else:
            delete_stmt = self.model.__table__.delete().where(*parsed_filters)
            await db.execute(delete_stmt)
        if commit:
            await db.commit()
