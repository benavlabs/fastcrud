# FastCRUD API Reference

`FastCRUD` is a comprehensive base class for CRUD operations on a model, utilizing Pydantic schemas for data validation and serialization. It provides both individual record operations and efficient bulk operations for high-performance database interactions.

## Class Definition

::: fastcrud.FastCRUD
    rendering:
      show_if_no_docstring: true

## Bulk Operations Overview

FastCRUD now includes powerful bulk operations that enable efficient processing of large datasets:

### 1. Bulk Insert

```python
async def insert_multi(
    db: AsyncSession,
    objects: List[Union[CreateSchemaType, dict[str, Any]]],
    *,
    batch_size: int = 1000,
    commit: bool = True,
    allow_partial_success: bool = True,
    return_summary: bool = False,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    return_as_model: bool = False,
) -> Union["BulkInsertSummary", List[Union[dict[str, Any], SelectSchemaType]]]
```

**Purpose**: Efficiently insert multiple records with batch processing, error handling, and detailed reporting.

**Key Features**:
- Configurable batch sizes for memory management
- Support for both Pydantic models and dictionaries
- Comprehensive error handling with partial success support
- Detailed operation summaries with performance metrics
- Support for soft delete columns and custom schemas

**Example Usage**:
```python
# Bulk insert users
users_data = [
    CreateUserSchema(name="Alice", email="alice@example.com"),
    CreateUserSchema(name="Bob", email="bob@example.com"),
    CreateUserSchema(name="Charlie", email="charlie@example.com")
]

# Simple bulk insert
inserted_users = await user_crud.insert_multi(db, users_data)

# Bulk insert with detailed summary
summary = await user_crud.insert_multi(
    db, 
    users_data, 
    return_summary=True,
    batch_size=500
)
print(f"Inserted {summary.successful_count} users")
print(f"Failed: {summary.failed_count}")
print(f"Duplicates: {summary.duplicate_count}")
```

### 2. Bulk Update

```python
async def update_multi(
    db: AsyncSession,
    objects: List[Union[UpdateSchemaType, dict[str, Any]]],
    *,
    batch_size: int = 1000,
    commit: bool = True,
    allow_partial_success: bool = True,
    return_summary: bool = False,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    return_as_model: bool = False,
) -> Union["BulkUpdateSummary", List[Union[dict[str, Any], SelectSchemaType]]]
```

**Purpose**: Efficiently update multiple records with proper primary key identification and batch processing.

**Key Features**:
- Automatic primary key detection (including composite keys)
- Support for both Pydantic models and dictionaries
- Handles partial updates and missing records gracefully
- Detailed operation summaries with failure analysis
- Transaction management with rollback support

**Example Usage**:
```python
# Bulk update users
updates = [
    {"id": 1, "email": "alice_new@example.com", "is_active": True},
    {"id": 2, "email": "bob_new@example.com", "is_active": False},
    UpdateUserSchema(id=3, email="charlie_new@example.com", role="admin")
]

# Simple bulk update
updated_users = await user_crud.update_multi(db, updates)

# Bulk update with summary
summary = await user_crud.update_multi(
    db, 
    updates, 
    return_summary=True,
    batch_size=500
)
print(f"Updated {summary.successful_count} users")
print(f"Not found: {summary.not_found_count}")
print(f"Unchanged: {summary.unchanged_count}")
```

### 3. Bulk Delete

```python
async def delete_multi(
    db: AsyncSession,
    *,
    allow_multiple: bool = True,
    batch_size: int = 1000,
    commit: bool = True,
    allow_partial_success: bool = True,
    return_summary: bool = False,
    soft_delete: Optional[bool] = None,
    **filters: Any,
) -> Union["BulkDeleteSummary", int]
```

**Purpose**: Efficiently delete multiple records with filtering, supporting both soft and hard deletes.

**Key Features**:
- Advanced filtering support with FastCRUD's filter system
- Automatic soft delete detection and configuration
- Comprehensive error handling and reporting
- Both soft and hard delete support
- Batch processing for large datasets

**Example Usage**:
```python
# Soft delete inactive users
deleted_count = await user_crud.delete_multi(
    db,
    is_active=False,
    soft_delete=True
)

# Bulk delete with summary
summary = await user_crud.delete_multi(
    db,
    status="archived",
    return_summary=True,
    batch_size=500
)
print(f"Deleted {summary.successful_count} users")
print(f"Soft deleted: {summary.soft_deleted_count}")
print(f"Hard deleted: {summary.hard_deleted_count}")

# Hard delete with date filters
await user_crud.delete_multi(
    db,
    is_test=True,
    created_at__lt=datetime.now() - timedelta(days=30),
    soft_delete=False
)
```

## Traditional CRUD Operations

### Core Methods

#### Create
```python
async def create(
    db: AsyncSession,
    object: CreateSchemaType,
    *,
    commit: bool = True,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    return_as_model: bool = False,
) -> Union[ModelType, SelectSchemaType, dict[str, Any], None]
```

**Purpose**: Create a new record in the database from a Pydantic schema.

When the `object` schema contains structured data (Pydantic models, dicts, or
lists) for attributes that correspond to relationship fields on the SQLAlchemy
model, FastCRUD will automatically construct and persist the related
instances as part of the same `create` call. This supports simple nested
creates for:

- one-to-one / many-to-one relationships (`uselist=False`)
- one-to-many relationships (`uselist=True`)

Relationship fields declared on `schema_to_select` are treated the same way
as in `get` / `get_multi`: they are **not** automatically populated unless
you explicitly join and nest them. The nested create behavior is about what
gets written to the database, not about eager-loading relationships on the
returned instance.

#### Get
```python
async def get(
    db: AsyncSession,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    return_as_model: bool = False,
    one_or_none: bool = False,
    **kwargs: Any,
) -> Optional[Union[dict[str, Any], SelectSchemaType]]
```

**Purpose**: Fetch a single record based on specified filters.

#### Update
```python
async def update(
    db: AsyncSession,
    object: Union[UpdateSchemaType, dict[str, Any]],
    allow_multiple: bool = False,
    commit: bool = True,
    return_columns: Optional[list[str]] = None,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    return_as_model: bool = False,
    one_or_none: bool = False,
    **kwargs: Any,
) -> Optional[Union[dict[str, Any], SelectSchemaType]]
```

**Purpose**: Update an existing record or multiple records based on specified filters.

#### Delete
```python
async def delete(
    db: AsyncSession,
    db_row: Optional[Row] = None,
    allow_multiple: bool = False,
    commit: bool = True,
    filters: Optional[DeleteSchemaType] = None,
    **kwargs: Any,
) -> None

async def db_delete(
    db: AsyncSession,
    allow_multiple: bool = False,
    commit: bool = True,
    filters: Optional[DeleteSchemaType] = None,
    **kwargs: Any,
) -> None
```

**Purpose**: Soft delete (using `delete`) or hard delete (using `db_delete`) records based on filters.

### Query Methods

#### Get Multiple
```python
async def get_multi(
    db: AsyncSession,
    offset: int = 0,
    limit: Optional[int] = 100,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    sort_columns: Optional[Union[str, list[str]]] = None,
    sort_orders: Optional[Union[str, list[str]]] = None,
    return_as_model: bool = False,
    return_total_count: bool = True,
    **kwargs: Any,
) -> Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict]
```

**Purpose**: Fetch multiple records with optional sorting, pagination, and model conversion.

#### Get Joined
```python
async def get_joined(
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
) -> Optional[dict[str, Any]]
```

**Purpose**: Fetch a single record with joins on other models, supporting nested data structures.

#### Get Multiple Joined
```python
async def get_multi_joined(
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
) -> Union[GetMultiResponseModel[SelectSchemaType], GetMultiResponseDict]
```

**Purpose**: Fetch multiple records with joins, supporting pagination, sorting, and nested data structures.

#### Get Multiple by Cursor
```python
async def get_multi_by_cursor(
    db: AsyncSession,
    cursor: Any = None,
    limit: int = 100,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    sort_column: str = "id",
    sort_order: str = "asc",
    return_as_model: bool = False,
    **kwargs: Any,
) -> dict[str, Union[list[Union[dict[str, Any], SelectSchemaType]], Any]]
```

**Purpose**: Implement cursor-based pagination for efficient data retrieval in large datasets.

### Utility Methods

#### Exists
```python
async def exists(
    db: AsyncSession,
    **kwargs: Any,
) -> bool
```

**Purpose**: Check if any records exist that match the given filter conditions.

#### Count
```python
async def count(
    db: AsyncSession,
    joins_config: Optional[list[JoinConfig]] = None,
    distinct_on_primary: bool = False,
    **kwargs: Any,
) -> int
```

**Purpose**: Count records that match specified filters, with support for joins and distinct counting.

#### Select
```python
async def select(
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    sort_columns: Optional[Union[str, list[str]]] = None,
    sort_orders: Optional[Union[str, list[str]]] = None,
    **kwargs: Any,
) -> Select
```

**Purpose**: Construct a SQLAlchemy Select statement with optional column selection, filtering, and sorting.

#### Upsert Operations
```python
async def upsert(
    db: AsyncSession,
    instance: Union[UpdateSchemaType, CreateSchemaType],
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    return_as_model: bool = False,
) -> Union[SelectSchemaType, dict[str, Any], None]

async def upsert_multi(
    db: AsyncSession,
    instances: list[Union[UpdateSchemaType, CreateSchemaType]],
    commit: bool = False,
    return_columns: Optional[list[str]] = None,
    schema_to_select: Optional[type[SelectSchemaType]] = None,
    return_as_model: bool = False,
    update_override: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[Union[UpsertMultiResponseDict, UpsertMultiResponseModel[SelectSchemaType]]]
```

**Purpose**: Upsert (insert or update) single or multiple records using database-specific optimizations.

## Configuration Options

### Constructor Parameters

```python
FastCRUD(
    model: type[ModelType],
    is_deleted_column: str = "is_deleted",
    deleted_at_column: str = "deleted_at", 
    updated_at_column: str = "updated_at",
    multi_response_key: str = "data",
)
```

**Parameters**:
- `model`: The SQLAlchemy model class to operate on
- `is_deleted_column`: Column name for soft delete flag (default: "is_deleted")
- `deleted_at_column`: Column name for soft delete timestamp (default: "deleted_at")
- `updated_at_column`: Column name for update timestamp (default: "updated_at")
- `multi_response_key`: Key name for multi-record responses (default: "data")

## Error Handling

FastCRUD provides comprehensive error handling across all operations:

### Validation Errors
- Invalid schema types
- Missing required fields
- Type validation failures

### Database Errors
- Integrity constraint violations
- Foreign key violations
- Unique constraint violations
- Transaction rollbacks

### Bulk Operation Errors
- Partial success handling
- Detailed error reporting
- Batch-level error isolation
- Performance metrics and diagnostics

## Performance Considerations

### Bulk Operations
- **Batch Size**: Tune based on your database and memory constraints
- **Transaction Strategy**: Use appropriate commit strategies for your use case
- **Partial Success**: Enable for resilient bulk operations
- **Monitoring**: Use summary reports to track performance

### Traditional Operations
- **Indexing**: Ensure proper indexes for frequently filtered columns
- **Pagination**: Use cursor-based pagination for large datasets
- **Joins**: Optimize join configurations for complex queries
- **Schema Selection**: Use specific schemas to reduce data transfer

## Examples

For comprehensive examples, see the [Usage Guide](../usage/crud.md) and [Integration Guide](../usage/endpoint.md).
