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
# Creating a new record (v0.20.0: returns None without schema_to_select)
result = await item_crud.create(db_session, create_schema_instance)
# result is None

# To get created data back as dict:
new_record = await item_crud.create(
    db_session, 
    create_schema_instance, 
    schema_to_select=YourReadSchema
)

# To get created data back as Pydantic model:
new_record = await item_crud.create(
    db_session, 
    create_schema_instance, 
    schema_to_select=YourReadSchema,
    return_as_model=True
)
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
# v0.20.0: Returns None without schema_to_select
result = await item_crud.create(db, CreateItemSchema(name="New Item"))
# result is None

# To get the created item data back as a dict:
new_item_dict = await item_crud.create(
    db, 
    CreateItemSchema(name="New Item"), 
    schema_to_select=ReadItemSchema
)
# new_item_dict is a dict with the created item data

# To get the created item data back as a Pydantic model:
new_item_model = await item_crud.create(
    db, 
    CreateItemSchema(name="New Item"), 
    schema_to_select=ReadItemSchema,
    return_as_model=True
)
# new_item_model is a ReadItemSchema instance
```

!!! INFO "v0.20.0 Behavior"

    **Changes Completed in v0.20.0**: The `create()` method now behaves consistently with other CRUD methods like `update()`. Changes made:
    
    - **Without `schema_to_select`**: Now returns `None` (was SQLAlchemy model)
    - **With `schema_to_select`**: Returns created data immediately - dict by default, Pydantic model if `return_as_model=True`
    
    **Recommended**: Always use `schema_to_select` to get the created data back in one operation. This is more efficient than separate create + get calls.

!!! WARNING

    Note that naive `datetime` such as `datetime.utcnow` is not supported by `FastCRUD` as it was [deprecated](https://github.com/python/cpython/pull/103858).
    
    Use timezone aware `datetime`, such as `datetime.now(UTC)` instead.

#### Creating Records with Nested Relationships

FastCRUD can automatically create related objects when the Pydantic *create* schema
contains structured data (Pydantic models, dicts, or lists) for relationship
fields on the SQLAlchemy model.

For example, consider a one-to-one relationship between `NestedParent` and
`NestedChild`:

??? example "Example: One-to-one nested create"

    ```python
    # SQLAlchemy models
    class NestedParent(Base):
        __tablename__ = "nested_parent"

        id = Column(Integer, primary_key=True)
        name = Column(String(32), nullable=False)
        child = relationship("NestedChild", back_populates="parent", uselist=False)


    class NestedChild(Base):
        __tablename__ = "nested_child"

        id = Column(Integer, primary_key=True)
        provider = Column(String(32), nullable=False)
        token = Column(String(64), nullable=False)
        parent_id = Column(Integer, ForeignKey("nested_parent.id"), nullable=False)
        parent = relationship("NestedParent", back_populates="child")


    # Pydantic create schemas
    class NestedChildCreate(BaseModel):
        provider: str
        token: str


    class NestedParentCreate(BaseModel):
        name: str
        child: Optional[NestedChildCreate] = None


    crud = FastCRUD(NestedParent)

    payload = NestedParentCreate(
        name="parent-1",
        child=NestedChildCreate(provider="google", token="secret-token"),
    )

    parent = await crud.create(db, payload)
    ```

In this example:

- `NestedParent` is created from the top-level fields of `NestedParentCreate`.
- Because the `child` attribute is a relationship on the model and the payload
  contains structured data for `child`, FastCRUD builds a `NestedChild`
  instance and associates it with the parent before persisting.
- Both rows are inserted in a single `create` call, with the correct
  `parent_id` set on the child.

The same mechanism works for one-to-many relationships using `uselist=True`:

??? example "Example: One-to-many nested create"

    ```python
    class Author(Base):
        __tablename__ = "author"

        id = Column(Integer, primary_key=True)
        name = Column(String(32), nullable=False)
        books = relationship(
            "Book", back_populates="author", cascade="all, delete-orphan"
        )


    class Book(Base):
        __tablename__ = "book"

        id = Column(Integer, primary_key=True)
        title = Column(String(64), nullable=False)
        author_id = Column(Integer, ForeignKey("author.id"), nullable=False)
        author = relationship("Author", back_populates="books")


    class BookCreate(BaseModel):
        title: str


    class AuthorCreate(BaseModel):
        name: str
        books: list[BookCreate] = []


    author_crud = FastCRUD(Author)

    payload = AuthorCreate(
        name="Author 1",
        books=[BookCreate(title="Book 1"), BookCreate(title="Book 2")],
    )

    author = await author_crud.create(db, payload)
    ```

Here, two `Book` rows are created and linked to the newly created `Author` row.

**Automatic detection rules**

- Only payload keys that match **relationship attributes** on the model are
  considered for nested creation.
- A value is treated as nested if it is:
  - a Pydantic `BaseModel` instance,
  - a mapping/dict, or
  - a non-empty list/tuple of such values.
- All other values, including scalar foreign keys, are left intact and passed
  directly to the model constructor (preserving existing flat create behavior).

**Supported relationship types**

- One-to-one and many-to-one (via `uselist=False` relationships)
- One-to-many (via `uselist=True` relationships)

Many-to-many and more complex join scenarios are better handled via
specialized helpers (see the dedicated many-to-many documentation and tests).

**Async / lazy-loading note**

When using SQLAlchemy’s async API with the default lazy-loading strategy,
accessing relationship attributes on the returned ORM instance (e.g.
`parent.child` or `author.books`) may trigger a lazy load that requires a
running greenlet context. In async tests or FastAPI endpoints this can raise
`sqlalchemy.exc.MissingGreenlet`.

To avoid this:

- Prefer explicit async queries using `select(...)` and `db.execute(...)` to
  verify related rows after a create, or
- Configure eager loading on your relationships if you need relationship data
  immediately on the returned instance.

??? example "Example: Nested create with schema_to_select"

    ```python
    # SQLAlchemy models (same as above)
    class NestedParent(Base):
        __tablename__ = "nested_parent"

        id = Column(Integer, primary_key=True)
        name = Column(String(32), nullable=False)
        child = relationship("NestedChild", back_populates="parent", uselist=False)


    class NestedChild(Base):
        __tablename__ = "nested_child"

        id = Column(Integer, primary_key=True)
        provider = Column(String(32), nullable=False)
        token = Column(String(64), nullable=False)
        parent_id = Column(Integer, ForeignKey("nested_parent.id"), nullable=False)
        parent = relationship("NestedParent", back_populates="child")


    # Pydantic create schema with nested payload
    class NestedChildCreate(BaseModel):
        provider: str
        token: str


    class NestedParentCreate(BaseModel):
        name: str
        child: Optional[NestedChildCreate] = None


    # Pydantic read/select schema used with schema_to_select
    class NestedChildRead(BaseModel):
        provider: str
        token: str


    class NestedParentReadWithChild(BaseModel):
        id: int
        name: str
        child: Optional[NestedChildRead] = None


    crud = FastCRUD(NestedParent)

    payload = NestedParentCreate(
        name="parent-select",
        child=NestedChildCreate(provider="google", token="secret-token"),
    )

    # return_as_model=True -> instance of NestedParentReadWithChild
    result_model = await crud.create(
        db,
        payload,
        schema_to_select=NestedParentReadWithChild,
        return_as_model=True,
    )

    # return_as_model=False -> plain dict with the same scalar fields
    result_dict = await crud.create(
        db,
        payload,
        schema_to_select=NestedParentReadWithChild,
        return_as_model=False,
    )
    ```

In this example:

- The nested payload still causes both `NestedParent` and `NestedChild` rows
  to be created and linked in the database.
- The **shape of the returned value** is controlled by `schema_to_select`
  and `return_as_model`, not by the create schema.
  - With `return_as_model=True`, you get a `NestedParentReadWithChild`
    instance.
  - With `return_as_model=False`, you get a plain `dict` with the selected
    scalar fields.
- Relationship fields declared on the select schema (like `child`) are **not
  automatically loaded** in the result, matching the behavior of `get` and
  `get_multi`. By default you will see `child=None` or the key omitted.

If you need relationship data right after create, use:

- explicit async queries with `select(NestedChild)` / `db.execute(...)`, or
- an eager loading strategy on your relationships.

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
- When `return_as_model=True` and `schema_to_select` is provided: `GetMultiResponseModel[SelectSchemaType]` - a TypedDict with `data: list[SelectSchemaType]` and `total_count: int` (when `return_total_count=True`)
- When `return_as_model=False`: `GetMultiResponseDict` - a TypedDict with `data: list[dict[str, Any]]` and `total_count: int` (when `return_total_count=True`)

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

**Typing Your Functions**:

When wrapping `get_multi` in your own functions, use the proper return types for full type safety:

```python
from fastcrud import GetMultiResponseDict, GetMultiResponseModel
from .schemas import ReadItemSchema

# Option 1: Return as dict with proper typing
async def get_items(db: AsyncSession) -> GetMultiResponseDict:
    return await item_crud.get_multi(db)

# Option 2: Return as model with proper typing
async def get_items_typed(db: AsyncSession) -> GetMultiResponseModel[ReadItemSchema]:
    return await item_crud.get_multi(
        db,
        schema_to_select=ReadItemSchema,
        return_as_model=True
    )

# Option 3: Let the type be inferred (omit return annotation)
async def get_items_inferred(db: AsyncSession):
    return await item_crud.get_multi(db)
```

!!! warning "Avoid `-> dict[str, Any]`"
    Using `-> dict[str, Any]` as a return type annotation discards the type information that `GetMultiResponseDict` provides. If you explicitly annotate with `dict[str, Any]`, you'll need to cast the result and lose the benefit of knowing that `result["data"]` is a `list` and `result["total_count"]` is an `int`.

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
- When `return_as_model=True` and `schema_to_select` is provided: `GetMultiResponseModel[SelectSchemaType]` - a TypedDict with `data: list[SelectSchemaType]` and `total_count: int` (when `return_total_count=True`)
- When `return_as_model=False`: `GetMultiResponseDict` - a TypedDict with `data: list[dict[str, Any]]` and `total_count: int` (when `return_total_count=True`)

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
**Return Types**: A dictionary with `data: list[...]` and `next_cursor: Any`:
- When `return_as_model=True` and `schema_to_select` is provided: `{"data": list[SelectSchemaType], "next_cursor": Any}`
- When `return_as_model=False`: `{"data": list[dict[str, Any]], "next_cursor": Any}`

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

---

## Bulk Operations for High-Performance Database Operations

FastCRUD now includes powerful bulk operations that enable efficient processing of large datasets. These methods provide significant performance improvements over individual record operations while maintaining data integrity and providing comprehensive error handling.

### When to Use Bulk Operations

Bulk operations are ideal for:
- **Data Migration**: Moving large amounts of data between systems
- **Batch Processing**: Processing CSV imports, data exports, or scheduled batch jobs
- **Initial Data Setup**: Creating large datasets during application initialization
- **Analytics Operations**: Bulk updates based on complex conditions
- **Data Cleanup**: Removing or archiving large numbers of records

### Benefits of Bulk Operations

1. **Performance**: Significantly faster than individual record operations
2. **Memory Efficiency**: Automatic batching prevents memory overflow
3. **Reliability**: Comprehensive error handling with partial success support
4. **Monitoring**: Detailed operation summaries with performance metrics
5. **Flexibility**: Support for both Pydantic models and dictionaries

### 1. Bulk Insert Operations

The `insert_multi` method efficiently inserts multiple records with batch processing and error handling.

#### Basic Bulk Insert

```python
# Create FastCRUD instance
user_crud = FastCRUD(User)

# Prepare data for bulk insertion
users_data = [
    CreateUserSchema(name="Alice", email="alice@example.com"),
    CreateUserSchema(name="Bob", email="bob@example.com"),
    CreateUserSchema(name="Charlie", email="charlie@example.com"),
    CreateUserSchema(name="Diana", email="diana@example.com"),
    CreateUserSchema(name="Eve", email="eve@example.com")
]

# Simple bulk insert
inserted_users = await user_crud.insert_multi(db, users_data)
print(f"Inserted {len(inserted_users)} users")
```

#### Bulk Insert with Configuration

```python
# Bulk insert with custom batch size and options
summary = await user_crud.insert_multi(
    db,
    objects=users_data,
    batch_size=1000,  # Process 1000 records at a time
    commit=False,     # Don't commit automatically (manage transaction yourself)
    allow_partial_success=True,  # Continue even if some batches fail
    return_summary=True  # Get detailed operation results
)

print(f"Total requested: {summary.total_requested}")
print(f"Successfully inserted: {summary.successful_count}")
print(f"Failed: {summary.failed_count}")
print(f"Duplicate entries: {summary.duplicate_count}")
print(f"Constraint violations: {summary.constraint_violations}")
print(f"Operation duration: {summary.duration_ms:.2f}ms")
print(f"Average speed: {summary.items_per_second:.2f} items/second")
```

#### Bulk Insert from Dictionaries

```python
# Insert from dictionary data (useful for CSV imports)
import csv

# Simulating CSV data
csv_data = [
    {"name": "Alice", "email": "alice@example.com", "age": 30},
    {"name": "Bob", "email": "bob@example.com", "age": 25},
    {"name": "Charlie", "email": "charlie@example.com", "age": 35}
]

# Convert to model instances and insert
user_data = [CreateUserSchema(**data) for data in csv_data]
result = await user_crud.insert_multi(db, user_data)
```

#### Bulk Insert with Schema Conversion

```python
# Insert and return specific fields as Pydantic models
inserted_users = await user_crud.insert_multi(
    db,
    objects=users_data,
    schema_to_select=ReadUserSchema,  # Return only specific fields
    return_as_model=True  # Convert to Pydantic models
)

# Now you have a list of ReadUserSchema instances
for user in inserted_users:
    print(f"User: {user.name}, Email: {user.email}")
```

### 2. Bulk Update Operations

The `update_multi` method efficiently updates multiple records while handling primary key identification and partial updates.

#### Basic Bulk Update

```python
# Prepare update data (must include primary key information)
updates = [
    {"id": 1, "email": "alice.updated@example.com", "is_active": True},
    {"id": 2, "email": "bob.updated@example.com", "is_active": False},
    {"id": 3, "email": "charlie.updated@example.com", "is_active": True},
    {"id": 4, "email": "diana.updated@example.com", "is_active": False}
]

# Simple bulk update
updated_users = await user_crud.update_multi(db, updates)
print(f"Updated {len(updated_users)} users")
```

#### Bulk Update with Partial Data

```python
# Update only specific fields for different users
partial_updates = [
    {"id": 1, "last_login": datetime.now()},  # Update only last login
    {"id": 2, "is_active": False},            # Update only active status
    {"id": 3, "email": "new.email@example.com"}  # Update only email
]

# Bulk update with summary for detailed results
summary = await user_crud.update_multi(
    db,
    objects=partial_updates,
    return_summary=True,
    batch_size=500
)

print(f"Updated: {summary.successful_count}")
print(f"Not found: {summary.not_found_count}")
print(f"Unchanged: {summary.unchanged_count}")
```

#### Bulk Update with Composite Primary Keys

```python
# Model with composite primary key
class OrderItem(Base):
    order_id = Column(Integer, ForeignKey("order.id"), primary_key=True)
    product_id = Column(Integer, ForeignKey("product.id"), primary_key=True)
    quantity = Column(Integer)
    price = Column(Numeric(10, 2))

# Update multiple order items
order_updates = [
    {"order_id": 1, "product_id": 101, "quantity": 5, "price": 25.99},
    {"order_id": 1, "product_id": 102, "quantity": 3, "price": 15.50},
    {"order_id": 2, "product_id": 101, "quantity": 2, "price": 25.99}
]

order_item_crud = FastCRUD(OrderItem)
result = await order_item_crud.update_multi(db, order_updates)
```

### 3. Bulk Delete Operations

The `delete_multi` method efficiently deletes multiple records with advanced filtering support and both soft and hard delete options.

#### Basic Bulk Delete

```python
# Delete users who haven't logged in for over a year
from datetime import datetime, timedelta

one_year_ago = datetime.now() - timedelta(days=365)

# Soft delete inactive users
deleted_count = await user_crud.delete_multi(
    db,
    last_login__lt=one_year_ago,
    soft_delete=True
)

print(f"Soft deleted {deleted_count} inactive users")
```

#### Bulk Delete with Advanced Filtering

```python
# Delete users based on multiple criteria
summary = await user_crud.delete_multi(
    db,
    is_test_user=True,           # Filter 1: test users
    created_at__lt=one_year_ago, # Filter 2: created over a year ago
    last_login__lt=one_year_ago, # Filter 3: not logged in recently
    is_active=False,             # Filter 4: inactive users
    allow_multiple=True,         # Allow deleting multiple records
    return_summary=True,
    soft_delete=True
)

print(f"Total deleted: {summary.successful_count}")
print(f"Soft deleted: {summary.soft_deleted_count}")
print(f"Hard deleted: {summary.hard_deleted_count}")
print(f"Operation took: {summary.duration_ms:.2f}ms")
```

#### Hard Delete with Cleanup

```python
# Permanently delete test data older than 30 days
test_data_cutoff = datetime.now() - timedelta(days=30)

summary = await user_crud.delete_multi(
    db,
    is_test_data=True,        # Only test data
    created_at__lt=test_data_cutoff,  # Older than 30 days
    soft_delete=False,        # Hard delete (permanent)
    return_summary=True
)

print(f"Permanently deleted {summary.hard_deleted_count} test records")
```

### 4. Performance Optimization and Best Practices

#### Batch Size Configuration

```python
# Different batch sizes for different scenarios

# For memory-intensive operations (large records)
summary = await user_crud.insert_multi(
    db, large_users_data, batch_size=100  # Small batches for memory efficiency
)

# For high-throughput operations (small records)
summary = await user_crud.insert_multi(
    db, simple_records_data, batch_size=5000  # Larger batches for speed
)

# Adaptive batch sizing based on record complexity
def get_optimal_batch_size(record_size_bytes: int, available_memory_mb: int = 100) -> int:
    """Calculate optimal batch size based on record size and available memory."""
    memory_per_record = record_size_bytes * 2  # Account for overhead
    max_batch_size = (available_memory_mb * 1024 * 1024) // memory_per_record
    return min(max_batch_size, 1000)  # Cap at 1000 for stability

# Use adaptive batching
batch_size = get_optimal_batch_size(record_size=1024, available_memory=200)
summary = await user_crud.insert_multi(db, data, batch_size=batch_size)
```

#### Transaction Management

```python
# Managed transaction approach
try:
    # Bulk insert with managed transaction
    summary = await user_crud.insert_multi(
        db,
        objects=user_data,
        commit=False,  # Don't auto-commit
        return_summary=True
    )
    
    if summary.success_rate > 0.95:  # 95% success threshold
        await db.commit()
        print(f"Successfully inserted {summary.successful_count} records")
    else:
        await db.rollback()
        print(f"Operation failed with {summary.failed_count} errors")
        
except Exception as e:
    await db.rollback()
    print(f"Transaction failed: {e}")
```

#### Error Handling and Recovery

```python
# Comprehensive error handling
summary = await user_crud.insert_multi(
    db,
    objects=user_data,
    allow_partial_success=True,  # Continue on errors
    return_summary=True
)

# Analyze results and handle failures
print(f"Success rate: {summary.success_rate:.2%}")

if summary.failed_items:
    print("Failed items:")
    for failed_item in summary.failed_items[:10]:  # Show first 10 failures
        print(f"  - {failed_item.error}: {failed_item.item}")
        
    # Retry failed items with different strategy
    failed_data = [item.item for item in summary.failed_items]
    retry_summary = await user_crud.insert_multi(
        db,
        objects=failed_data,
        batch_size=50  # Smaller batches for problematic data
    )
```

#### Monitoring and Logging

```python
# Monitor bulk operation performance
import logging

# Set up performance monitoring
def monitor_bulk_operation(operation_name: str, summary):
    logger = logging.getLogger("bulk_operations")
    
    logger.info(f"{operation_name} completed:")
    logger.info(f"  - Total items: {summary.total_requested}")
    logger.info(f"  - Success rate: {summary.success_rate:.2%}")
    logger.info(f"  - Duration: {summary.duration_ms:.2f}ms")
    logger.info(f"  - Throughput: {summary.items_per_second:.2f} items/sec")
    
    if summary.failed_count > 0:
        logger.warning(f"  - Failed items: {summary.failed_count}")
        logger.warning(f"  - Error rate: {(summary.failed_count/summary.total_requested)*100:.1f}%")

# Monitor a bulk insert operation
summary = await user_crud.insert_multi(db, user_data, return_summary=True)
monitor_bulk_operation("User import", summary)
```

### 5. Real-World Examples

#### CSV Data Import

```python
import csv
from fastapi import HTTPException

async def import_users_from_csv(db: AsyncSession, csv_file_path: str):
    """Import users from CSV file with error handling and progress reporting."""
    
    users_data = []
    errors = []
    
    # Read CSV data
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_num, row in enumerate(reader, 1):
            try:
                # Validate and convert row data
                user_data = CreateUserSchema(
                    name=row['name'],
                    email=row['email'],
                    age=int(row['age']) if row['age'] else None
                )
                users_data.append(user_data)
            except Exception as e:
                errors.append(f"Row {row_num}: {e}")
    
    if errors:
        raise HTTPException(status_code=400, detail=f"CSV validation errors: {errors}")
    
    # Bulk insert with progress reporting
    summary = await user_crud.insert_multi(
        db,
        objects=users_data,
        batch_size=100,
        return_summary=True
    )
    
    return {
        "imported": summary.successful_count,
        "failed": summary.failed_count,
        "errors": [error.error_message for error in summary.failed_items[:10]]
    }
```

#### Scheduled Data Cleanup

```python
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler

async def cleanup_old_data():
    """Scheduled cleanup of old user data."""
    
    # Delete inactive users older than 2 years
    cutoff_date = datetime.now() - timedelta(days=730)
    
    summary = await user_crud.delete_multi(
        db,
        is_active=False,
        last_login__lt=cutoff_date,
        soft_delete=True,
        return_summary=True
    )
    
    # Log cleanup results
    print(f"Cleanup completed: {summary.successful_count} users archived")
    
    # Archive soft-deleted users (convert to hard delete after 30 days)
    archive_summary = await user_crud.delete_multi(
        db,
        is_deleted=True,
        deleted_at__lt=datetime.now() - timedelta(days=30),
        soft_delete=False,
        return_summary=True
    )
    
    print(f"Archive completed: {archive_summary.hard_deleted_count} users permanently removed")

# Schedule daily cleanup at 2 AM
scheduler = AsyncIOScheduler()
scheduler.add_job(cleanup_old_data, 'cron', hour=2)
scheduler.start()
```

#### Data Synchronization

```python
async def sync_external_users(db: AsyncSession, external_data: list):
    """Synchronize users from external system with conflict resolution."""
    
    # Separate new users and existing users
    new_users = []
    existing_updates = []
    
    for ext_user in external_data:
        if ext_user.get('id'):
            # Existing user - prepare update
            existing_updates.append(ext_user)
        else:
            # New user - prepare insert
            new_users.append(ext_user)
    
    # Process new users
    if new_users:
        new_summary = await user_crud.insert_multi(
            db,
            objects=new_users,
            return_summary=True
        )
        print(f"Added {new_summary.successful_count} new users")
    
    # Process existing users
    if existing_updates:
        update_summary = await user_crud.update_multi(
            db,
            objects=existing_updates,
            return_summary=True
        )
        print(f"Updated {update_summary.successful_count} existing users")
        print(f"Not found: {update_summary.not_found_count}")
    
    return {
        "new_users": len(new_users),
        "updated_users": len(existing_updates),
        "sync_errors": new_summary.failed_count + update_summary.failed_count if 'new_summary' in locals() else update_summary.failed_count
    }
```

These bulk operations significantly improve performance for large-scale database operations while maintaining data integrity and providing comprehensive error handling and monitoring capabilities.
FastCRUD provides mechanisms to handle common database errors, ensuring robust API behavior.
