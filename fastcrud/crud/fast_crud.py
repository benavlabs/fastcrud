from typing import Any, Generic, Union, Optional, Callable, cast
from datetime import datetime, timezone

from pydantic import ValidationError
from sqlalchemy import (
    Insert,
    Result,
    and_,
    select,
    update,
    delete,
    func,
    inspect,
    asc,
    desc,
    or_,
    column,
    not_,
    Column,
)
from sqlalchemy.exc import ArgumentError, MultipleResultsFound, NoResultFound
from sqlalchemy.sql import Join
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql.elements import BinaryExpression, ColumnElement
from sqlalchemy.sql.selectable import Select
from sqlalchemy.dialects import postgresql, sqlite, mysql

from fastcrud.types import (
    CreateSchemaType,
    DeleteSchemaType,
    ModelType,
    SelectSchemaType,
    UpdateSchemaInternalType,
    UpdateSchemaType,
    GetMultiResponseModel,
    GetMultiResponseDict,
)

from .helper import (
    _extract_matching_columns_from_schema,
    _auto_detect_join_condition,
    _nest_join_data,
    _nest_multi_join_data,
    _handle_null_primary_key_multi_join,
    JoinConfig,
)

from ..endpoint.helper import _get_primary_keys

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

    _SUPPORTED_FILTERS = {
        "eq": lambda column: column.__eq__,
        "gt": lambda column: column.__gt__,
        "lt": lambda column: column.__lt__,
        "gte": lambda column: column.__ge__,
        "lte": lambda column: column.__le__,
        "ne": lambda column: column.__ne__,
        "is": lambda column: column.is_,
        "is_not": lambda column: column.is_not,
        "like": lambda column: column.like,
        "notlike": lambda column: column.notlike,
        "ilike": lambda column: column.ilike,
        "notilike": lambda column: column.notilike,
        "startswith": lambda column: column.startswith,
        "endswith": lambda column: column.endswith,
        "contains": lambda column: column.contains,
        "match": lambda column: column.match,
        "between": lambda column: column.between,
        "in": lambda column: column.in_,
        "not_in": lambda column: column.not_in,
        "or": lambda column: column.or_,
        "not": lambda column: column.not_,
    }

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
        self._primary_keys = _get_primary_keys(self.model)

    def _get_sqlalchemy_filter(
        self,
        operator: str,
        value: Any,
    ) -> Optional[FilterCallable]:
        if operator in {"in", "not_in", "between"}:
            if not isinstance(value, (tuple, list, set)):
                raise ValueError(f"<{operator}> filter must be tuple, list or set")
        return cast(Optional[FilterCallable], self._SUPPORTED_FILTERS.get(operator))

    def _parse_filters(
            self,
            model: Optional[Union[type[ModelType], AliasedClass]] = None,
            **kwargs
    ) -> list[ColumnElement]:
        """Parse and convert filter arguments into SQLAlchemy filter conditions.

        Args:
            model: The model to apply filters to. Defaults to self.model
            **kwargs: Filter arguments in the format field_name__operator=value

        Returns:
            List of SQLAlchemy filter conditions
        """
        model = model or self.model
        filters = []

        if "_or" in kwargs:
            filters.extend(self._handle_multi_field_or_filter(model, kwargs.pop("_or")))

        for key, value in kwargs.items():
            if "__" not in key:
                filters.extend(self._handle_simple_filter(model, key, value))
                continue

            field_name, operator = key.rsplit("__", 1)
            model_column = self._get_column(model, field_name)

            if operator == "or":
                filters.extend(self._handle_or_filter(model_column, value))
            elif operator == "not":
                filters.extend(self._handle_not_filter(model_column, value))
            else:
                filters.extend(self._handle_standard_filter(model_column, operator, value))

        return filters

    def _handle_simple_filter(
            self,
            model: Union[type[ModelType], AliasedClass],
            key: str,
            value: Any
    ) -> list[ColumnElement]:
        """Handle simple equality filters (e.g., name='John')."""
        col = getattr(model, key, None)
        return [col == value] if col is not None else []

    def _handle_or_filter(
            self,
            col: Column,
            value: dict
    ) -> list[ColumnElement]:
        """Handle OR conditions (e.g., age__or={'gt': 18, 'lt': 65})."""
        if not isinstance(value, dict):  # pragma: no cover
            raise ValueError("OR filter value must be a dictionary")

        or_conditions = []
        for or_op, or_value in value.items():
            sqlalchemy_filter = self._get_sqlalchemy_filter(or_op, or_value)
            if sqlalchemy_filter:
                condition = (
                    sqlalchemy_filter(col)(*or_value)
                    if or_op == "between"
                    else sqlalchemy_filter(col)(or_value)
                )
                or_conditions.append(condition)

        return [or_(*or_conditions)] if or_conditions else []

    def _handle_not_filter(
            self,
            col: Column,
            value: dict
    ) -> list[ColumnElement[bool]]:
        """Handle NOT conditions (e.g., age__not={'eq': 20, 'between': (30, 40)})."""
        if not isinstance(value, dict):  # pragma: no cover
            raise ValueError("NOT filter value must be a dictionary")

        not_conditions = []
        for not_op, not_value in value.items():
            sqlalchemy_filter = self._get_sqlalchemy_filter(not_op, not_value)
            if sqlalchemy_filter is None:  # pragma: no cover
                continue

            condition = (
                sqlalchemy_filter(col)(*not_value)
                if not_op == "between"
                else sqlalchemy_filter(col)(not_value)
            )
            not_conditions.append(condition)

        return [and_(*(not_(cond) for cond in not_conditions))] if not_conditions else []

    def _handle_standard_filter(
            self,
            col: Column[Any],
            operator: str,
            value: Any
    ) -> list[ColumnElement[bool]]:
        """Handle standard comparison operators (e.g., age__gt=18)."""
        sqlalchemy_filter = self._get_sqlalchemy_filter(operator, value)
        if sqlalchemy_filter is None:  # pragma: no cover
            return []

        condition = (
            sqlalchemy_filter(col)(*value)
            if operator == "between"
            else sqlalchemy_filter(col)(value)
        )
        return [condition]

    def _handle_multi_field_or_filter(
            self,
            model: Union[type[ModelType], AliasedClass],
            value: dict
    ) -> list[ColumnElement]:
        """Handle OR conditions across multiple fields.

        This method allows for OR conditions between different fields, such as:
        _or={'name__ilike': '%keyword%', 'email__ilike': '%keyword%'}

        Args:
            model: The model to apply filters to
            value: Dictionary of field conditions in the format {'field_name__operator': value}

        Returns:
            List containing a single SQLAlchemy OR condition combining all specified filters
        """
        if not isinstance(value, dict):  # pragma: no cover
            raise ValueError("Multi-field OR filter value must be a dictionary")

        or_conditions = []

        for field_condition, condition_value in value.items():
            if "__" not in field_condition:
                col = getattr(model, field_condition, None)
                if col is not None:
                    or_conditions.append(col == condition_value)
                continue

            field_name, operator = field_condition.rsplit("__", 1)
            try:
                model_column = self._get_column(model, field_name)

                sqlalchemy_filter = self._get_sqlalchemy_filter(operator, condition_value)
                if sqlalchemy_filter:
                    condition = (
                        sqlalchemy_filter(model_column)(*condition_value)
                        if operator == "between"
                        else sqlalchemy_filter(model_column)(condition_value)
                    )
                    or_conditions.append(condition)
            except ValueError:
                # TODO: log warning
                continue

        return [or_(*or_conditions)] if or_conditions else []

    def _get_column(
            self,
            model: Union[type[ModelType], AliasedClass],
            field_name: str
    ) -> Column[Any]:
        """Get column from model, raising ValueError if not found."""
        model_column = getattr(model, field_name, None)
        if model_column is None:
            raise ValueError(f"Invalid filter column: {field_name}")
        return cast(Column[Any], model_column)

    def _apply_sorting(
        self,
        stmt: Select,
        sort_columns: Union[str, list[str]],
        sort_orders: Optional[Union[str, list[str]]] = None,
    ) -> Select:
        """
        Apply sorting to a SQLAlchemy query based on specified column names and sort orders.

        Args:
            stmt: The SQLAlchemy `Select` statement to which sorting will be applied.
            sort_columns: A single column name or a list of column names on which to apply sorting.
            sort_orders: A single sort order (`"asc"` or `"desc"`) or a list of sort orders corresponding
                to the columns in `sort_columns`. If not provided, defaults to `"asc"` for each column.

        Raises:
            ValueError: Raised if sort orders are provided without corresponding sort columns,
                or if an invalid sort order is provided (not `"asc"` or `"desc"`).
            ArgumentError: Raised if an invalid column name is provided that does not exist in the model.

        Returns:
            The modified `Select` statement with sorting applied.

        Examples:
            Applying ascending sort on a single column:
            >>> stmt = _apply_sorting(stmt, 'name')

            Applying descending sort on a single column:
            >>> stmt = _apply_sorting(stmt, 'age', 'desc')

            Applying mixed sort orders on multiple columns:
            >>> stmt = _apply_sorting(stmt, ['name', 'age'], ['asc', 'desc'])

            Applying ascending sort on multiple columns:
            >>> stmt = _apply_sorting(stmt, ['name', 'age'])

        Note:
            This method modifies the passed `Select` statement in-place by applying the `order_by` clause
            based on the provided column names and sort orders.
        """
        if sort_orders and not sort_columns:
            raise ValueError("Sort orders provided without corresponding sort columns.")

        if sort_columns:
            if not isinstance(sort_columns, list):
                sort_columns = [sort_columns]

            if sort_orders:
                if not isinstance(sort_orders, list):
                    sort_orders = [sort_orders] * len(sort_columns)
                if len(sort_columns) != len(sort_orders):
                    raise ValueError(
                        "The length of sort_columns and sort_orders must match."
                    )

                for idx, order in enumerate(sort_orders):
                    if order not in ["asc", "desc"]:
                        raise ValueError(
                            f"Invalid sort order: {order}. Only 'asc' or 'desc' are allowed."
                        )

            validated_sort_orders = (
                ["asc"] * len(sort_columns) if not sort_orders else sort_orders
            )

            for idx, column_name in enumerate(sort_columns):
                column = getattr(self.model, column_name, None)
                if not column:
                    raise ArgumentError(f"Invalid column name: {column_name}")

                order = validated_sort_orders[idx]
                stmt = stmt.order_by(asc(column) if order == "asc" else desc(column))

        return stmt

    def _prepare_and_apply_joins(
        self,
        stmt: Select,
        joins_config: list[JoinConfig],
        use_temporary_prefix: bool = False,
    ):
        """
        Applies joins to the given SQL statement based on a list of `JoinConfig` objects.

        Args:
            stmt: The initial SQL statement.
            joins_config: Configurations for all joins.
            use_temporary_prefix: Whether to use or not an additional prefix for joins. Default `False`.

        Returns:
            The modified SQL statement with joins applied.
        """
        for join in joins_config:
            model = join.alias or join.model
            join_select = _extract_matching_columns_from_schema(
                model,
                join.schema_to_select,
                join.join_prefix,
                join.alias,
                use_temporary_prefix,
            )
            joined_model_filters = self._parse_filters(
                model=model, **(join.filters or {})
            )

            if join.join_type == "left":
                stmt = stmt.outerjoin(model, join.join_on).add_columns(*join_select)
            elif join.join_type == "inner":
                stmt = stmt.join(model, join.join_on).add_columns(*join_select)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported join type: {join.join_type}.")
            if joined_model_filters:
                stmt = stmt.filter(*joined_model_filters)

        return stmt

    async def create(
        self, db: AsyncSession, object: CreateSchemaType, commit: bool = True
    ) -> ModelType:
        """
        Create a new record in the database.

        Args:
            db: The SQLAlchemy async session.
            object: The Pydantic schema containing the data to be saved.
            commit: If `True`, commits the transaction immediately. Default is `True`.

        Returns:
            The created database object.
        """
        object_dict = object.model_dump()
        db_object: ModelType = self.model()
        # Set the fields on the model using setattr in case the model has `init=False`
        for field, value in object_dict.items():
            setattr(db_object, field, value)
        db.add(db_object)
        if commit:
            await db.commit()
        return db_object

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
        to_select = _extract_matching_columns_from_schema(
            model=self.model, schema=schema_to_select
        )
        filters = self._parse_filters(**kwargs)
        stmt = select(*to_select).filter(*filters)

        if sort_columns:
            stmt = self._apply_sorting(stmt, sort_columns, sort_orders)
        return stmt

    async def get(
        self,
        db: AsyncSession,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        one_or_none: bool = False,
        **kwargs: Any,
    ) -> Optional[Union[dict, SelectSchemaType]]:
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
            A dictionary or a Pydantic model instance of the fetched database row, or `None` if no match is found.

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
            The created or updated instance
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
            db_instance = await self.create(db, instance)  # type: ignore
            db_instance = schema_to_select.model_validate(  # type: ignore
                db_instance, from_attributes=True
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
    ) -> Optional[dict[str, Any]]:
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
            The updated record(s) as a dictionary or Pydantic model instance or None, depending on the value of `return_as_model` and `return_columns`.

        Raises:
            ValueError: If the MySQL dialect is used with filters, return_columns, schema_to_select, or return_as_model.
            NotImplementedError: If the database dialect is not supported for upsert multi.
        """
        if update_override is None:
            update_override = {}
        filters = self._parse_filters(**kwargs)

        if db.bind.dialect.name == "postgresql":
            statement, params = await self._upsert_multi_postgresql(
                instances, filters, update_override
            )
        elif db.bind.dialect.name == "sqlite":
            statement, params = await self._upsert_multi_sqlite(
                instances, filters, update_override
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
            statement, params = await self._upsert_multi_mysql(
                instances, update_override
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
            return self._as_multi_response(
                db_row,
                schema_to_select=schema_to_select,
                return_as_model=return_as_model,
            )

        await db.execute(statement, params)
        if commit:
            await db.commit()
        return None

    async def _upsert_multi_postgresql(
        self,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        filters: list[ColumnElement],
        update_set_override: dict[str, Any],
    ) -> tuple[Insert, list[dict]]:
        statement = postgresql.insert(self.model)
        statement = statement.on_conflict_do_update(
            index_elements=self._primary_keys,
            set_={
                column.name: getattr(statement.excluded, column.name)
                for column in self.model.__table__.columns
                if not column.primary_key and not column.unique
            }
            | update_set_override,
            where=and_(*filters) if filters else None,
        )
        params = [
            self.model(**instance.model_dump()).__dict__ for instance in instances
        ]
        return statement, params

    async def _upsert_multi_sqlite(
        self,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        filters: list[ColumnElement],
        update_set_override: dict[str, Any],
    ) -> tuple[Insert, list[dict]]:
        statement = sqlite.insert(self.model)
        statement = statement.on_conflict_do_update(
            index_elements=self._primary_keys,
            set_={
                column.name: getattr(statement.excluded, column.name)
                for column in self.model.__table__.columns
                if not column.primary_key and not column.unique
            }
            | update_set_override,
            where=and_(*filters) if filters else None,
        )
        params = [
            self.model(**instance.model_dump()).__dict__ for instance in instances
        ]
        return statement, params

    async def _upsert_multi_mysql(
        self,
        instances: list[Union[UpdateSchemaType, CreateSchemaType]],
        update_set_override: dict[str, Any],
    ) -> tuple[Insert, list[dict]]:
        statement = mysql.insert(self.model)
        statement = statement.on_duplicate_key_update(
            {
                column.name: getattr(statement.inserted, column.name)
                for column in self.model.__table__.columns
                if not column.primary_key
                and not column.unique
                and column.name != self.deleted_at_column
            }
            | update_set_override,
        )
        params = [
            self.model(**instance.model_dump()).__dict__ for instance in instances
        ]
        return statement, params

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
        filters = self._parse_filters(**kwargs)
        stmt = select(self.model).filter(*filters).limit(1)

        result = await db.execute(stmt)
        return result.first() is not None

    async def count(
        self,
        db: AsyncSession,
        joins_config: Optional[list[JoinConfig]] = None,
        **kwargs: Any,
    ) -> int:
        """
        Counts records that match specified filters.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Can also count records based on a configuration of joins, useful for complex queries involving relationships.

        Args:
            db: The database session to use for the operation.
            joins_config: Optional configuration for applying joins in the count query.
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
        primary_filters = self._parse_filters(**kwargs)

        if joins_config is not None:
            primary_keys = [p.name for p in _get_primary_keys(self.model)]
            if not any(primary_keys):  # pragma: no cover
                raise ValueError(
                    f"The model '{self.model.__name__}' does not have a primary key defined, which is required for counting with joins."
                )
            to_select = [
                getattr(self.model, pk).label(f"distinct_{pk}") for pk in primary_keys
            ]
            base_query = select(*to_select)

            for join in joins_config:
                join_model = join.alias or join.model
                join_filters = (
                    self._parse_filters(model=join_model, **join.filters)
                    if join.filters
                    else []
                )

                if join.join_type == "inner":
                    base_query = base_query.join(join_model, join.join_on)
                else:
                    base_query = base_query.outerjoin(join_model, join.join_on)

                if join_filters:
                    base_query = base_query.where(*join_filters)

            if primary_filters:
                base_query = base_query.where(*primary_filters)

            subquery = base_query.subquery()
            count_query = select(func.count()).select_from(subquery)
        else:
            count_query = select(func.count()).select_from(self.model)
            if primary_filters:
                count_query = count_query.where(*primary_filters)

        total_count: Optional[int] = await db.scalar(count_query)
        if total_count is None:
            raise ValueError("Could not find the count.")

        return total_count

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
        if (limit is not None and limit < 0) or offset < 0:
            raise ValueError("Limit and offset must be non-negative.")

        stmt = await self.select(
            schema_to_select=schema_to_select,
            sort_columns=sort_columns,
            sort_orders=sort_orders,
            **kwargs,
        )

        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        data = [dict(row) for row in result.mappings()]

        response: dict[str, Any] = {self.multi_response_key: data}

        if return_total_count:
            total_count = await self.count(db=db, **kwargs)
            response["total_count"] = total_count

        if return_as_model:
            if not schema_to_select:
                raise ValueError(
                    "schema_to_select must be provided when return_as_model is True."
                )
            try:
                model_data = [schema_to_select(**row) for row in data]
                response[self.multi_response_key] = model_data
            except ValidationError as e:
                raise ValueError(
                    f"Data validation error for schema {schema_to_select.__name__}: {e}"
                )

        return response

    async def get_joined(
        self,
        db: AsyncSession,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
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
    ) -> Optional[dict[str, Any]]:
        """
        Fetches a single record with one or multiple joins on other models. If `join_on` is not provided, the method attempts
        to automatically detect the join condition using foreign key relationships. For multiple joins, use `joins_config` to
        specify each join configuration.

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
            joins_config: A list of `JoinConfig` instances, each specifying a model to join with, join condition, optional prefix for column names, schema for selecting specific columns, and the type of join. This parameter enables support for multiple joins.
            nest_joins: If `True`, nested data structures will be returned where joined model data are nested under the `join_prefix` as a dictionary.
            relationship_type: Specifies the relationship type, such as `"one-to-one"` or `"one-to-many"`. Used to determine how to nest the joined data. If `None`, uses `"one-to-one"`.
            **kwargs: Filters to apply to the primary model query, supporting advanced comparison operators for refined searching.

        Returns:
            A dictionary representing the joined record, or `None` if no record matches the criteria.

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

        primary_select = _extract_matching_columns_from_schema(
            model=self.model,
            schema=schema_to_select,
        )
        stmt: Select = select(*primary_select).select_from(self.model)

        join_definitions = joins_config if joins_config else []
        if join_model:
            join_definitions.append(
                JoinConfig(
                    model=join_model,
                    join_on=join_on,
                    join_prefix=join_prefix,
                    schema_to_select=join_schema_to_select,
                    join_type=join_type,
                    alias=alias,
                    filters=join_filters,
                    relationship_type=relationship_type,
                )
            )

        stmt = self._prepare_and_apply_joins(
            stmt=stmt, joins_config=join_definitions, use_temporary_prefix=nest_joins
        )
        primary_filters = self._parse_filters(**kwargs)
        if primary_filters:
            stmt = stmt.filter(*primary_filters)

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

        if data_list:
            if nest_joins:
                nested_data: dict = {}
                for data in data_list:
                    nested_data = _nest_join_data(
                        data,
                        join_definitions,
                        nested_data=nested_data,
                    )
                return nested_data
            return data_list[0]

        return None

    async def get_multi_joined(
        self,
        db: AsyncSession,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
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
        return_as_model: bool = False,
        joins_config: Optional[list[JoinConfig]] = None,
        return_total_count: bool = True,
        relationship_type: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
            return_total_count: If `True`, also returns the total count of rows with the selected filters. Useful for pagination.
            relationship_type: Specifies the relationship type, such as `"one-to-one"` or `"one-to-many"`. Used to determine how to nest the joined data. If `None`, uses `"one-to-one"`.
            **kwargs: Filters to apply to the primary query, including advanced comparison operators for refined searching.

        Returns:
            A dictionary containing the fetched rows under `"data"` key and total count under `"total_count"`.

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
        """
        if joins_config and (
            join_model
            or join_prefix
            or join_on
            or join_schema_to_select
            or alias
            or relationship_type
        ):
            raise ValueError(
                "Cannot use both single join parameters and joins_config simultaneously."
            )
        elif not joins_config and not join_model:
            raise ValueError("You need one of join_model or joins_config.")

        if (limit is not None and limit < 0) or offset < 0:
            raise ValueError("Limit and offset must be non-negative.")

        if relationship_type is None:
            relationship_type = "one-to-one"

        primary_select = _extract_matching_columns_from_schema(
            model=self.model, schema=schema_to_select
        )
        stmt: Select = select(*primary_select)

        join_definitions = joins_config if joins_config else []
        if join_model:
            try:
                join_definitions.append(
                    JoinConfig(
                        model=join_model,
                        join_on=join_on
                        if join_on is not None
                        else _auto_detect_join_condition(self.model, join_model),
                        join_prefix=join_prefix,
                        schema_to_select=join_schema_to_select,
                        join_type=join_type,
                        alias=alias,
                        filters=join_filters,
                        relationship_type=relationship_type,
                    )
                )
            except ValueError as e:  # pragma: no cover
                raise ValueError(f"Could not configure join: {str(e)}")

        stmt = self._prepare_and_apply_joins(
            stmt=stmt, joins_config=join_definitions, use_temporary_prefix=nest_joins
        )

        primary_filters = self._parse_filters(**kwargs)
        if primary_filters:
            stmt = stmt.filter(*primary_filters)

        if sort_columns:
            stmt = self._apply_sorting(stmt, sort_columns, sort_orders)

        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        data: list[Union[dict, SelectSchemaType]] = []

        for row in result.mappings().all():
            row_dict = dict(row)

            if nest_joins:
                row_dict = _nest_join_data(
                    data=row_dict,
                    join_definitions=join_definitions,
                )

            if return_as_model:
                if schema_to_select is None:
                    raise ValueError(
                        "schema_to_select must be provided when return_as_model is True."
                    )
                try:
                    model_instance = schema_to_select(**row_dict)
                    data.append(model_instance)
                except ValidationError as e:
                    raise ValueError(
                        f"Data validation error for schema {schema_to_select.__name__}: {e}"
                    )
            else:
                data.append(row_dict)

        if nest_joins and any(
            join.relationship_type == "one-to-many" for join in join_definitions
        ):
            nested_data = _nest_multi_join_data(
                base_primary_key=self._primary_keys[0].name,  # type: ignore[misc]
                data=data,
                joins_config=join_definitions,
                return_as_model=return_as_model,
                schema_to_select=schema_to_select if return_as_model else None,
                nested_schema_to_select={
                    (
                        join.join_prefix.rstrip("_")
                        if join.join_prefix
                        else join.model.__tablename__
                    ): join.schema_to_select
                    for join in join_definitions
                    if join.schema_to_select
                },
            )
        else:
            nested_data = _handle_null_primary_key_multi_join(data, join_definitions)

        response: dict[str, Any] = {"data": nested_data}

        if return_total_count:
            total_count: int = await self.count(
                db=db, joins_config=join_definitions, **kwargs
            )
            response["total_count"] = total_count

        return response

    async def get_multi_by_cursor(
        self,
        db: AsyncSession,
        cursor: Any = None,
        limit: int = 100,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        sort_column: str = "id",
        sort_order: str = "asc",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Implements cursor-based pagination for fetching records. This method is designed for efficient data retrieval in large datasets and is ideal for features like infinite scrolling.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The SQLAlchemy async session.
            cursor: The cursor value to start fetching records from. Defaults to `None`.
            limit: Maximum number of rows to fetch.
            schema_to_select: Pydantic schema for selecting specific columns.
            sort_column: Column name to use for sorting and cursor pagination.
            sort_order: Sorting direction, either `"asc"` or `"desc"`.
            **kwargs: Filters to apply to the query, including advanced comparison operators for detailed querying.

        Returns:
            A dictionary containing the fetched rows under `"data"` key and the next cursor value under `"next_cursor"`.

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

            Fetch records with age greater than 30 using cursor-based pagination:

            ```python
            first_page = await user_crud.get_multi_by_cursor(
                db,
                limit=10,
                sort_column='age',
                sort_order='asc',
            )
            ```

            Fetch records excluding a specific username using cursor-based pagination:

            ```python
                db,
                limit=10,
                sort_column='username',
                sort_order='asc',
                username__ne='admin',
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
            if sort_order == "asc":
                stmt = stmt.filter(getattr(self.model, sort_column) > cursor)
            else:
                stmt = stmt.filter(getattr(self.model, sort_column) < cursor)

        stmt = stmt.order_by(
            asc(getattr(self.model, sort_column))
            if sort_order == "asc"
            else desc(getattr(self.model, sort_column))
        )
        stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        data = [dict(row) for row in result.mappings()]

        next_cursor = data[-1][sort_column] if len(data) == limit else None

        return {"data": data, "next_cursor": next_cursor}

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
            The updated record(s) as a dictionary or Pydantic model instance or `None`, depending on the value of `return_as_model` and `return_columns`.

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
        total_count = await self.count(db, **kwargs)
        if total_count == 0:
            raise NoResultFound("No record found to update.")
        if not allow_multiple and total_count > 1:
            raise MultipleResultsFound(
                f"Expected exactly one record to update, found {total_count}."
            )

        if isinstance(object, dict):
            update_data = object
        else:
            update_data = object.model_dump(exclude_unset=True)

        updated_at_col = getattr(self.model, self.updated_at_column, None)
        if updated_at_col:
            update_data[self.updated_at_column] = datetime.now(timezone.utc)

        update_data_keys = set(update_data.keys())
        model_columns = {_column.name for _column in inspect(self.model).c}
        extra_fields = update_data_keys - model_columns
        if extra_fields:
            raise ValueError(f"Extra fields provided: {extra_fields}")

        filters = self._parse_filters(**kwargs)
        stmt = update(self.model).filter(*filters).values(update_data)

        if return_as_model:
            return_columns = self.model_col_names

        if return_columns:
            stmt = stmt.returning(*[column(name) for name in return_columns])
            db_row = await db.execute(stmt)
            if commit:
                await db.commit()
            if allow_multiple:
                return self._as_multi_response(
                    db_row,
                    schema_to_select=schema_to_select,
                    return_as_model=return_as_model,
                )
            return self._as_single_response(
                db_row,
                schema_to_select=schema_to_select,
                return_as_model=return_as_model,
                one_or_none=one_or_none,
            )

        await db.execute(stmt)
        if commit:
            await db.commit()
        return None

    def _as_single_response(
        self,
        db_row: Result,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
        one_or_none: bool = False,
    ) -> Optional[Union[dict, SelectSchemaType]]:
        if not (result := db_row.one_or_none() if one_or_none else db_row.first()):
            return None

        out = dict(result._mapping)
        if not return_as_model:
            return out
        if not schema_to_select:  # pragma: no cover
            raise ValueError(
                "schema_to_select must be provided when return_as_model is True."
            )
        return schema_to_select(**out)

    def _as_multi_response(
        self,
        db_row: Result,
        schema_to_select: Optional[type[SelectSchemaType]] = None,
        return_as_model: bool = False,
    ) -> dict:
        data = [dict(row) for row in db_row.mappings()]
        
        if not return_as_model:
            return {"data": data}
        
        if not schema_to_select:  # pragma: no cover
            raise ValueError("schema_to_select required when return_as_model is True")
        
        try:
            return {"data": [schema_to_select(**row) for row in data]}
        except ValidationError as e:  # pragma: no cover
            raise ValueError(f"Schema validation error ({schema_to_select.__name__}): {e}")

    async def db_delete(
        self,
        db: AsyncSession,
        allow_multiple: bool = False,
        commit: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Deletes a record or multiple records from the database based on specified filters.

        For filtering details see [the Advanced Filters documentation](../advanced/crud.md/#advanced-filters)

        Args:
            db: The database session to use for the operation.
            allow_multiple: If `True`, allows deleting multiple records that match the filters. If `False`, raises an error if more than one record matches the filters.
            commit: If `True`, commits the transaction immediately. Default is `True`.
            **kwargs: Filters to identify the record(s) to delete, including advanced comparison operators for detailed querying.

        Returns:
            None

        Raises:
            MultipleResultsFound: If `allow_multiple` is `False` and more than one record matches the filters.

        Examples:
            Delete a user based on their ID:

            ```python
            await user_crud.db_delete(db, id=1)
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
        """
        if not allow_multiple and (total_count := await self.count(db, **kwargs)) > 1:
            raise MultipleResultsFound(
                f"Expected exactly one record to delete, found {total_count}."
            )

        filters = self._parse_filters(**kwargs)
        stmt = delete(self.model).filter(*filters)
        await db.execute(stmt)
        if commit:
            await db.commit()

    async def delete(
        self,
        db: AsyncSession,
        db_row: Optional[Row] = None,
        allow_multiple: bool = False,
        commit: bool = True,
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
            **kwargs: Filters to identify the record(s) to delete, supporting advanced comparison operators for refined querying.

        Raises:
            MultipleResultsFound: If `allow_multiple` is `False` and more than one record matches the filters.
            NoResultFound: If no record matches the filters.

        Returns:
            None

        Examples:
            Soft delete a specific user by ID:

            ```python
            await user_crud.delete(db, id=1)
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
        """
        filters = self._parse_filters(**kwargs)
        if db_row:
            has_soft_delete = (
                hasattr(db_row, self.is_deleted_column) and 
                hasattr(db_row, self.deleted_at_column)
            )
            if has_soft_delete:
                setattr(db_row, self.is_deleted_column, True)
                setattr(db_row, self.deleted_at_column, datetime.now(timezone.utc))
            else:
                await db.delete(db_row)
            if commit:
                await db.commit()
            return

        total_count = await self.count(db, **kwargs)
        if total_count == 0:
            raise NoResultFound("No record found to delete.")
        if not allow_multiple and total_count > 1:
            raise MultipleResultsFound(
                f"Expected exactly one record to delete, found {total_count}."
            )

        update_values: dict[str, Union[bool, datetime]] = {}
        if self.deleted_at_column in self.model_col_names:
            update_values[self.deleted_at_column] = datetime.now(timezone.utc)
        if self.is_deleted_column in self.model_col_names:
            update_values[self.is_deleted_column] = True

        if update_values:
            update_stmt = update(self.model).filter(*filters).values(**update_values)
            await db.execute(update_stmt)

        else:
            delete_stmt = self.model.__table__.delete().where(*filters)
            await db.execute(delete_stmt)
        if commit:
            await db.commit()
