# Using FastCRUD for Enhanced CRUD Operations

FastCRUD is a versatile tool for handling CRUD (Create, Read, Update, Delete) operations in FastAPI applications with SQLAlchemy models. It leverages Pydantic schemas for data validation and serialization, offering a streamlined approach to database interactions.

## Key Features

- Simplified CRUD operations with SQLAlchemy models.
- Data validation and serialization using Pydantic.
- Support for complex queries including joins and pagination.

## Getting Started

### Step 1: Define Models and Schemas

Define your SQLAlchemy models and Pydantic schemas for data representation.

??? example "Models and Schemas Used Below"

    ??? example "`item/model.py`"

        ```python
        --8<--
        fastcrud/examples/item/model.py:imports
        fastcrud/examples/item/model.py:model
        --8<--
        ```

    ??? example "`item/schemas.py`"

        ```python
        --8<--
        fastcrud/examples/item/schemas.py:imports
        fastcrud/examples/item/schemas.py:createschema
        fastcrud/examples/item/schemas.py:updateschema
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
        fastcrud/examples/customer/schemas.py:readschema
        --8<--
        ```

    ??? example "`product/model.py`"

        ```python
        --8<--
        fastcrud/examples/product/model.py:imports
        fastcrud/examples/product/model.py:model
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

### Step 2: Initialize FastCRUD

Create a `FastCRUD` instance for your model to handle CRUD operations.

```python
from fastcrud import FastCRUD

# Creating a FastCRUD instance
item_crud = FastCRUD(Item)
order_crud = FastCRUD(Order)
```

### Step 3: Pick your Method

Then you just pick the method you need and use it like this:

```python
# Creating a new record
new_record = await item_crud.create(db_session, create_schema_instance)
```

More on available methods below.

---

## Understanding FastCRUD Methods

FastCRUD offers a comprehensive suite of methods for CRUD operations, each designed to handle different aspects of database interactions efficiently.

### 1. Create

```python
create(
    db: AsyncSession,
    object: CreateSchemaType,
    commit: bool = True,
) -> ModelType
```

**Purpose**: To create a new record in the database.  
**Usage Example**: Creates an item with name `"New Item"`.

```python
new_item = await item_crud.create(db, CreateItemSchema(name="New Item"))
```

!!! WARNING "Deprecated Behavior"

    **Upcoming Changes in Next Major Version**: The `create()` method currently behaves inconsistently compared to other CRUD methods like `update()`. It will change:
    
    - **Currently without `schema_to_select`**: Returns SQLAlchemy model → **Will return `None`**
    - **Currently with `schema_to_select`**: Bypasses `return_as_model` parameter → **Will respect `return_as_model` like other methods**
    
    This makes `create()` consistent with `update()` behavior. Current usage patterns will trigger deprecation warnings. See [changelog](../changelog.md#0192---nov-15-2025) for details.

!!! WARNING

    Note that naive `datetime` such as `datetime.utcnow` is not supported by `FastCRUD` as it was [deprecated](https://github.com/python/cpython/pull/103858).
    
    Use timezone aware `datetime`, such as `datetime.now(UTC)` instead.

### 2. Get

```python
get(
    db: AsyncSession,
    schema_to_select: Optional[type[BaseModel]] = None,
    return_as_model: bool = False,
    one_or_none: bool = False,
    **kwargs: Any,
) -> Optional[Union[dict, BaseModel]]
```

**Purpose**: To fetch a single record based on filters, with an option to select specific columns using a Pydantic schema.  
**Return Types**:
- When `return_as_model=True` and `schema_to_select` is provided: `Optional[SelectSchemaType]`
- When `return_as_model=False`: `Optional[Dict[str, Any]]`

**Usage Examples**: 

Fetch item as dictionary:
```python
item = await item_crud.get(db, id=item_id)
# Returns: Optional[Dict[str, Any]]
```

Fetch item as typed Pydantic model:
```python
from .schemas import ReadItemSchema

item = await item_crud.get(
    db, 
    schema_to_select=ReadItemSchema,
    return_as_model=True,
    id=item_id
)
# Returns: Optional[ReadItemSchema]
```

### 3. Exists

```python
exists(
    db: AsyncSession,
    **kwargs: Any,
) -> bool
```

**Purpose**: To check if a record exists based on provided filters.  
**Usage Example**: Checks whether an item with name `"Existing Item"` exists.

```python
exists = await item_crud.exists(db, name="Existing Item")
```

### 4. Count

```python
count(
    db: AsyncSession,
    joins_config: Optional[list[JoinConfig]] = None,
    **kwargs: Any,
) -> int
```

**Purpose**: To count the number of records matching provided filters.  
**Usage Example**: Counts the number of items with the `"Books"` category.

```python
count = await item_crud.count(db, category="Books")
```

### 5. Get Multi

```python
get_multi(
    db: AsyncSession,
    offset: int = 0,
    limit: Optional[int] = 100,
    schema_to_select: Optional[type[BaseModel]] = None,
    sort_columns: Optional[Union[str, list[str]]] = None,
    sort_orders: Optional[Union[str, list[str]]] = None,
    return_as_model: bool = False,
    return_total_count: bool = True,
    **kwargs: Any,
) -> dict[str, Any]
```

**Purpose**: To fetch multiple records with optional sorting, pagination, and model conversion.  
**Return Types**:
- When `return_as_model=True` and `schema_to_select` is provided: `GetMultiResponseModel[SelectSchemaType]` (i.e., `Dict[str, Union[List[SelectSchemaType], int]]`)
- When `return_as_model=False`: `GetMultiResponseDict` (i.e., `Dict[str, Union[List[Dict[str, Any]], int]]`)

**Usage Examples**: 

Fetch items as dictionaries:
```python
items = await item_crud.get_multi(db, offset=10, limit=5)
# Returns: {"data": [Dict[str, Any], ...], "total_count": int}
```

Fetch items as typed Pydantic models:
```python
from .schemas import ReadItemSchema

items = await item_crud.get_multi(
    db, 
    offset=10, 
    limit=5,
    schema_to_select=ReadItemSchema,
    return_as_model=True
)
# Returns: {"data": [ReadItemSchema, ...], "total_count": int}
```

### 6. Update

```python
update(
    db: AsyncSession, 
    object: Union[UpdateSchemaType, dict[str, Any]], 
    allow_multiple: bool = False,
    commit: bool = True,
    return_columns: Optional[list[str]] = None,
    schema_to_select: Optional[type[BaseModel]] = None,
    return_as_model: bool = False,
    one_or_none: bool = False,
    **kwargs: Any,
) -> Optional[Union[dict, BaseModel]]
```

**Purpose**: To update an existing record in the database.  
**Return Types**:
- When `return_as_model=True` and `schema_to_select` is provided: `Optional[SelectSchemaType]`
- When `return_as_model=False`: `Optional[Dict[str, Any]]`

**Usage Examples**: 

Update and return as dictionary:
```python
updated_item = await item_crud.update(
    db,
    UpdateItemSchema(description="Updated"),
    id=item_id,
)
# Returns: Optional[Dict[str, Any]]
```

Update and return as typed Pydantic model:
```python
from .schemas import ReadItemSchema, UpdateItemSchema

updated_item = await item_crud.update(
    db,
    UpdateItemSchema(description="Updated"),
    schema_to_select=ReadItemSchema,
    return_as_model=True,
    id=item_id,
)
# Returns: Optional[ReadItemSchema]
```

### 7. Delete

```python
delete(
    db: AsyncSession, 
    db_row: Optional[Row] = None, 
    allow_multiple: bool = False,
    commit: bool = True,
    **kwargs: Any,
) -> None
```

**Purpose**: To delete a record from the database, with support for soft delete.  
**Usage Example**: Deletes the item with `item_id` as its `id`, performs a soft delete if the model has the `is_deleted` column.

```python
await item_crud.delete(db, id=item_id)
```

### 8. Hard Delete

```python
db_delete(
    db: AsyncSession, 
    allow_multiple: bool = False,
    commit: bool = True,
    **kwargs: Any,
) -> None
```

**Purpose**: To hard delete a record from the database.  
**Usage Example**: Hard deletes the item with `item_id` as its `id`.

```python
await item_crud.db_delete(db, id=item_id)
```

---

## Advanced Methods for Complex Queries and Joins

FastCRUD extends its functionality with advanced methods tailored for complex query operations and handling joins. These methods cater to specific use cases where more sophisticated data retrieval and manipulation are required.

### 1. Get Multi

```python
get_multi(
    db: AsyncSession,
    offset: int = 0,
    limit: Optional[int] = 100,
    schema_to_select: Optional[type[BaseModel]] = None,
    sort_columns: Optional[Union[str, list[str]]] = None,
    sort_orders: Optional[Union[str, list[str]]] = None,
    return_as_model: bool = False,
    return_total_count: bool = True,
    **kwargs: Any,
) -> dict[str, Any]
```

**Purpose**: To fetch multiple records based on specified filters, with options for sorting and pagination.  
**Usage Example**: Gets the first 10 items sorted by `name` in ascending order.

```python
items = await item_crud.get_multi(
    db,
    offset=0,
    limit=10,
    sort_columns=['name'],
    sort_orders=['asc'],
)
```

### 2. Get Joined

```python
get_joined(
    db: AsyncSession,
    schema_to_select: Optional[type[BaseModel]] = None,
    join_model: Optional[ModelType] = None,
    join_on: Optional[Union[Join, BinaryExpression]] = None,
    join_prefix: Optional[str] = None,
    join_schema_to_select: Optional[type[BaseModel]] = None,
    join_type: str = "left",
    alias: Optional[AliasedClass] = None,
    join_filters: Optional[dict] = None,
    joins_config: Optional[list[JoinConfig]] = None,
    nest_joins: bool = False,
    relationship_type: Optional[str] = None,
    **kwargs: Any,
) -> Optional[dict[str, Any]]
```

**Purpose**: To fetch a single record with one or multiple joins on other models.  
**Usage Example**: Fetches order details for a specific order by joining with the `Customer` table, selecting specific columns as defined in `ReadOrderSchema` and `ReadCustomerSchema`.

```python
order_details = await order_crud.get_joined(
    db,
    schema_to_select=ReadOrderSchema,
    join_model=Customer,
    join_schema_to_select=ReadCustomerSchema,
    id=order_id,
)
```

### 3. Get Multi Joined

```python
get_multi_joined(
    db: AsyncSession,
    schema_to_select: Optional[type[BaseModel]] = None,
    join_model: Optional[type[ModelType]] = None,
    join_on: Optional[Any] = None,
    join_prefix: Optional[str] = None,
    join_schema_to_select: Optional[type[BaseModel]] = None,
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
) -> dict[str, Any]
```

**Purpose**: Similar to `get_joined`, but for fetching multiple records.  
**Return Types**:
- When `return_as_model=True` and `schema_to_select` is provided: `GetMultiResponseModel[SelectSchemaType]` (i.e., `Dict[str, Union[List[SelectSchemaType], int]]`)
- When `return_as_model=False`: `GetMultiResponseDict` (i.e., `Dict[str, Union[List[Dict[str, Any]], int]]`)

**Usage Examples**: 

Fetch joined records as dictionaries:
```python
orders = await order_crud.get_multi_joined(
    db,
    schema_to_select=ReadOrderSchema,
    join_model=Customer,
    join_schema_to_select=ReadCustomerSchema,
    offset=0,
    limit=5,
)
# Returns: {"data": [Dict[str, Any], ...], "total_count": int}
```

Fetch joined records as typed Pydantic models:
```python
orders = await order_crud.get_multi_joined(
    db,
    schema_to_select=ReadOrderSchema,
    join_model=Customer,
    join_schema_to_select=ReadCustomerSchema,
    return_as_model=True,
    offset=0,
    limit=5,
)
# Returns: {"data": [ReadOrderSchema, ...], "total_count": int}
```

### 4. Get Multi By Cursor

```python
get_multi_by_cursor(
    db: AsyncSession,
    cursor: Any = None,
    limit: int = 100,
    schema_to_select: Optional[type[BaseModel]] = None,
    sort_column: str = "id",
    sort_order: str = "asc",
    return_as_model: bool = False,
    **kwargs: Any,
) -> dict[str, Any]
```

**Purpose**: Implements cursor-based pagination for efficient data retrieval in large datasets.  
**Return Types**:
- When `return_as_model=True` and `schema_to_select` is provided: `Dict[str, Union[List[SelectSchemaType], Any]]`
- When `return_as_model=False`: `Dict[str, Union[List[Dict[str, Any]], Any]]`

**Usage Examples**: 

Cursor pagination with dictionaries:
```python
paginated_items = await item_crud.get_multi_by_cursor(
    db,
    cursor=last_cursor,
    limit=10,
    sort_column='created_at',
    sort_order='desc',
)
# Returns: {"data": [Dict[str, Any], ...], "next_cursor": Any}
```

Cursor pagination with typed Pydantic models:
```python
from .schemas import ReadItemSchema

paginated_items = await item_crud.get_multi_by_cursor(
    db,
    cursor=last_cursor,
    limit=10,
    schema_to_select=ReadItemSchema,
    return_as_model=True,
    sort_column='created_at',
    sort_order='desc',
)
# Returns: {"data": [ReadItemSchema, ...], "next_cursor": Any}
```

### 5. Select

```python
async def select(
    db: AsyncSession,
    schema_to_select: Optional[type[BaseModel]] = None,
    sort_columns: Optional[Union[str, list[str]]] = None,
    sort_orders: Optional[Union[str, list[str]]] = None,
    **kwargs: Any,
) -> Select
```

**Purpose**: Constructs a SQL Alchemy `Select` statement with optional column selection, filtering, and sorting.
**Usage Example**: Selects all items, filtering by `name` and sorting by `id`. Returns the `Select` statement.

```python
stmt = await item_crud.select(
    schema_to_select=ItemSchema,
    sort_columns='id',
    name='John',
)
# Note: This method returns a SQL Alchemy Select object, not the actual query result.
```

### 6. Count for Joined Models

```python
count(
    db: AsyncSession,
    joins_config: Optional[list[JoinConfig]] = None,
    **kwargs: Any,
) -> int
```

**Purpose**: To count records that match specified filters, especially useful in scenarios involving joins between models. This method supports counting unique entities across relationships, a common requirement in many-to-many or complex relationships.  
**Usage Example**: Count the number of unique projects a participant is involved in, considering a many-to-many relationship between `Project` and `Participant` models.

??? example "Models"

    ```python
    --8<--
    tests/sqlalchemy/conftest.py:model_project
    tests/sqlalchemy/conftest.py:model_participant
    tests/sqlalchemy/conftest.py:model_proj_parts_assoc
    --8<--
    ```

```python
project_crud = FastCRUD(Project)
projects_count = await project_crud.count(
    db=session,
    joins_config=[
        JoinConfig(
            model=Participant,
            join_on=ProjectsParticipantsAssociation.project_id == Project.id,
            join_type="inner",
        ),
    ],
    participant_id=specific_participant_id,
)
```

## Error Handling

FastCRUD provides mechanisms to handle common database errors, ensuring robust API behavior.
