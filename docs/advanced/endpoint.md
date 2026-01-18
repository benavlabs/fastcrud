# Advanced Use of EndpointCreator

## Available Automatic Endpoints
FastCRUD automates the creation of CRUD (Create, Read, Update, Delete) endpoints for your FastAPI application. Here's an overview of the available automatic endpoints and how they work, based on [the automatic endpoints we've generated before](../usage/endpoint.md#step-3-use-crud_router-to-create-endpoints):

### Create

- **Endpoint**: `/{model}`
- **Method**: `POST`
- **Description**: Creates a new item in the database.
- **Request Body**: JSON object based on the `create_schema`.
- **Example Request**: `POST /items` with JSON body.
- **Response**: 
  - When `select_schema` is provided: Returns the created item data immediately (dict by default, Pydantic model if configured)
  - When no `select_schema`: Returns `null` (v0.20.0 behavior)
  - When auto fields are used: Returns SQLAlchemy model (legacy compatibility)
- **Tip**: Use `select_schema` to get the created data back without additional API calls

### Read

- **Endpoint**: `/{model}/{id}`
- **Method**: `GET`
- **Description**: Retrieves a single item by its ID.
- **Path Parameters**: `id` - The ID of the item to retrieve.
- **Example Request**: `GET /items/1`.
- **Example Return**:
```javascript
{
    "id": 1,
    "name": "Item 1",
    "description": "Description of item 1",
    "category": "Movies",
    "price": 5.99,
    "last_sold": null,
    "created_at": "2024-01-01 12:00:00"
}
```

### Read Multiple

- **Endpoint**: `/{model}`
- **Method**: `GET`
- **Description**: Retrieves multiple items with optional pagination.
- **Query Parameters**:
    - `offset` (optional): The offset from where to start fetching items.
    - `limit` (optional): The maximum number of items to return.
    - `page` (optional): The page number, starting from 1.
    - `itemsPerPage` (optional): The number of items per page.
    - `sort` (optional): Sort results by one or more fields. Format: `field1,-field2` where `-` prefix indicates descending order.
- **Example Requests**:
    - `GET /items?offset=3&limit=4` (pagination)
    - `GET /items?sort=name` (sort by name ascending)
    - `GET /items?sort=-price,name` (sort by price descending, then name ascending)
- **Example Return**:
```javascript
{
  "data": [
    {
        "id": 4,
        "name": "Item 4",
        "description": "Description of item 4",
        "category": "Books",
        "price": 5.99,
        "last_sold": null,
        "created_at": "2024-01-01 12:01:00"
    },
    {
        "id": 5,
        "name": "Item 5",
        "description": "Description of item 5",
        "category": "Music",
        "price": 5.99,
        "last_sold": "2024-04-01 00:00:00",
        "created_at": "2024-01-01 12:10:00"
    },
    {
        "id": 6,
        "name": "Item 6",
        "description": "Description of item 6",
        "category": "TV",
        "price": 5.99,
        "last_sold": null,
        "created_at": "2024-01-01 12:15:00"
    },
    {
        "id": 7,
        "name": "Item 7",
        "description": "Description of item 7",
        "category": "Books",
        "price": 5.99,
        "last_sold": null,
        "created_at": "2024-01-01 13:00:30"
    }
  ],
  "total_count": 50
}
```
- **Example Paginated Request**: `GET /items?page=1&itemsPerPage=3`.
- **Example Paginated Return**:
```javascript
{
  "data": [
    {
        "id": 1,
        "name": "Item 1",
        "description": "Description of item 1",
        "category": "Movies",
        "price": 5.99,
        "last_sold": null,
        "created_at": "2024-01-01 12:00:01"
    },
    {
        "id": 2,
        "name": "Item 2",
        "description": "Description of item 2",
        "category": "TV",
        "price": 19.99,
        "last_sold": null,
        "created_at": "2024-01-01 12:00:15"
    },
    {
        "id": 3,
        "name": "Item 3",
        "description": "Description of item 3",
        "category": "Books",
        "price": 4.99,
        "last_sold": null,
        "created_at": "2024-01-01 12:00:16"
    }
  ],
  "total_count": 50,
  "has_more": true,
  "page": 1,
  "items_per_page": 3
}
```

!!! NOTE

    `_read_paginated` endpoint was deprecated and mixed into `_read_items` in the release `0.15.0`.
    Simple `_read_items` behaviour persists with no breaking changes.

    Read items paginated:
    ```sh
    $ curl -X 'GET' \
      'http://localhost:8000/users?page=2&itemsPerPage=10' \
      -H 'accept: application/json'
    ```

    Read items unpaginated:
    ```sh
    $ curl -X 'GET' \
      'http://localhost:8000/users?offset=0&limit=100' \
      -H 'accept: application/json'
    ```


### Update

- **Endpoint**: `/{model}/{id}`
- **Method**: `PATCH`
- **Description**: Updates an existing item by its ID.
- **Path Parameters**: `id` - The ID of the item to update.
- **Request Body**: JSON object based on the `update_schema`.
- **Example Request**: `PATCH /items/1` with JSON body.
- **Example Return**: `None`
- Note: If the target item is not found by ID, the generated endpoint returns a 404 Not Found with detail "Item not found".

### Delete

- **Endpoint**: `/{model}/{id}`
- **Method**: `DELETE`
- **Description**: Deletes (soft delete if configured) an item by its ID.
- **Path Parameters**: `id` - The ID of the item to delete.
- **Example Request**: `DELETE /items/1`.
- **Example Return**: `None`
- Note: If the target item is not found by ID, the generated endpoint returns a 404 Not Found with detail "Item not found".

### DB Delete (Hard Delete)

- **Endpoint**: `/{model}/db_delete/{id}` (Available if a `delete_schema` is provided)
- **Method**: `DELETE`
- **Description**: Permanently deletes an item by its ID, bypassing the soft delete mechanism.
- **Path Parameters**: `id` - The ID of the item to hard delete.
- **Example Request**: `DELETE /items/db_delete/1`.
- **Example Return**: `None`

## Selective CRUD Operations

You can control which CRUD operations are exposed by using `included_methods` and `deleted_methods`. These parameters allow you to specify exactly which CRUD methods should be included or excluded when setting up the router. By default, all CRUD endpoints are included.

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

### Using `included_methods`

Using `included_methods` you may define exactly the methods you want to be included.

```python hl_lines="10"
# Using crud_router with selective CRUD methods
my_router = crud_router(
    session=get_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    crud=FastCRUD(MyModel),
    path="/mymodel",
    tags=["MyModel"],
    included_methods=["create", "read", "update"],  # Only these methods will be included
)

app.include_router(my_router)
```

### Using `deleted_methods`

Using `deleted_methods` you define the methods that will not be included.

```python hl_lines="10"
# Using crud_router with selective CRUD methods
my_router = crud_router(
    session=get_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    crud=FastCRUD(MyModel),
    path="/mymodel",
    tags=["MyModel"],
    deleted_methods=["update", "delete"],  # All but these methods will be included
)

app.include_router(my_router)
```

!!! WARNING

    If `included_methods` and `deleted_methods` are both provided, a `ValueError` will be raised.

## Customizing Endpoint Names

You can customize the names of the auto generated endpoints by passing an `endpoint_names` dictionary when initializing the `EndpointCreator` or calling the `crud_router` function. This dictionary should map the CRUD operation names (`create`, `read`, `update`, `delete`, `db_delete`, `read_multi`) to your desired endpoint names.

### Example: Using `crud_router`

Here's how you can customize endpoint names using the `crud_router` function:

```python
from fastapi import FastAPI
from fastcrud import crud_router

from .database import async_session
from .mymodel.model import MyModel
from .mymodel.schemas import CreateMyModelSchema, UpdateMyModelSchema

app = FastAPI()

# Custom endpoint names
custom_endpoint_names = {
    "create": "add",
    "read": "fetch",
    "update": "modify",
    "delete": "remove",
    "read_multi": "list",
}

# Setup CRUD router with custom endpoint names
app.include_router(crud_router(
    session=async_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    path="/mymodel",
    tags=["MyModel"],
    endpoint_names=custom_endpoint_names,
))
```

In this example, the standard CRUD endpoints will be replaced with `/add`, `/fetch/{id}`, `/modify/{id}`, `/remove/{id}`, `/list`, and `/paginate`.

### Example: Using `EndpointCreator`

If you are using `EndpointCreator`, you can also pass the `endpoint_names` dictionary to customize the endpoint names similarly:

```python
# Custom endpoint names
custom_endpoint_names = {
    "create": "add_new",
    "read": "get_single",
    "update": "change",
    "delete": "erase",
    "db_delete": "hard_erase",
    "read_multi": "get_all",
    "read_paginated": "get_page",
}

# Initialize and use the custom EndpointCreator
endpoint_creator = EndpointCreator(
    session=async_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    path="/mymodel",
    tags=["MyModel"],
    endpoint_names=custom_endpoint_names,
)

endpoint_creator.add_routes_to_router()
app.include_router(endpoint_creator.router)
```

!!! TIP

    You only need to pass the names of the endpoints you want to change in the `endpoint_names` `dict`.

!!! NOTE

    `default_endpoint_names` for `EndpointCreator` were changed to empty strings in `0.15.0`.
    See [this issue](https://github.com/igorbenav/fastcrud/issues/67) for more details.

## Joined Model Filtering

FastCRUD supports filtering on related models using dot notation in filter configurations. This allows you to filter records based on attributes of joined models without manually writing complex queries.

### Basic Joined Model Filtering

You can filter records based on attributes of related models by using dot notation in your filter configuration:

```python
from fastapi import FastAPI
from fastcrud import EndpointCreator, FilterConfig

# Assuming you have User and Company models with a relationship
app = FastAPI()

endpoint_creator = EndpointCreator(
    session=async_session,
    model=User,
    create_schema=CreateUserSchema,
    update_schema=UpdateUserSchema,
    filter_config=FilterConfig(
        # Regular filters
        name=None,
        email=None,
        # Joined model filters
        **{
            "company.name": None,           # Filter by company name
            "company.industry": None,       # Filter by company industry
            "company.founded_year": None,   # Filter by company founded year
        }
    ),
)

endpoint_creator.add_routes_to_router()
app.include_router(endpoint_creator.router, prefix="/users")
```

### Using Joined Model Filters

Once configured, you can use joined model filters in your API requests:

```bash
# Filter users by company name
GET /users?company.name=TechCorp

# Filter users by company industry
GET /users?company.industry=Technology

# Combine regular and joined filters
GET /users?name=John&company.name=TechCorp

# Use filter operators with joined models
GET /users?company.founded_year__gte=2000
```

### Supported Filter Operators

Joined model filters support all the same operators as regular filters:

```python
filter_config=FilterConfig(**{
    "company.name__eq": None,           # Exact match
    "company.name__ne": None,           # Not equal
    "company.name__in": None,           # In list
    "company.founded_year__gte": None,  # Greater than or equal
    "company.founded_year__lt": None,   # Less than
    "company.revenue__between": None,   # Between values
})
```

### Multi-level Relationships

You can filter through multiple levels of relationships:

```python
# Assuming User -> Company -> Address relationship
filter_config=FilterConfig(**{
    "company.address.city": None,
    "company.address.country": None,
})

# Usage:
# GET /users?company.address.city=San Francisco
```

### How It Works

When you use joined model filters, FastCRUD automatically:

1. **Detects joined filters**: Identifies filter keys containing dot notation
2. **Validates relationships**: Ensures the relationship path exists in your models
3. **Generates joins**: Automatically creates the necessary SQL joins
4. **Applies filters**: Adds WHERE clauses for the joined model attributes

The system generates efficient SQL queries like:

```sql
SELECT user.id, user.name, user.email, user.company_id,
       company.id AS company_id_1, company.name AS company_name, company.industry
FROM user
LEFT OUTER JOIN company ON user.company_id = company.id
WHERE company.name = 'TechCorp'
```

### Limitations

- Currently supports single-relationship joins (one level of relationship at a time)
- Complex many-to-many relationships may require custom implementation
- Performance considerations apply for deeply nested relationships

### Example Models

Here's an example of models that work well with joined model filtering:

```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Company(Base):
    __tablename__ = "company"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    industry = Column(String(50))
    users = relationship("User", back_populates="company")

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    company = relationship("Company", back_populates="users")
```

## Including Relationships in Endpoints

Auto-generated endpoints can automatically include related data from SQLAlchemy relationships using the `include_relationships` parameter.

### Basic Usage

Enable relationship inclusion when creating your router:

=== "All Relationships"
    ```python
    from fastcrud import crud_router

    router = crud_router(
        session=get_session,
        model=User,
        create_schema=UserCreate,
        update_schema=UserUpdate,
        include_relationships=True,  # Include all relationships
        nest_joins=True,  # Nest related data
        path="/users",
        tags=["Users"],
    )
    ```

=== "Specific Relationships"
    ```python
    from fastcrud import crud_router

    router = crud_router(
        session=get_session,
        model=User,
        create_schema=UserCreate,
        update_schema=UserUpdate,
        include_relationships=["tier", "department"],  # Only these relationships
        nest_joins=True,
        path="/users",
        tags=["Users"],
    )
    ```

=== "Manual JoinConfig"
    ```python
    from fastcrud import crud_router, JoinConfig

    router = crud_router(
        session=get_session,
        model=User,
        create_schema=UserCreate,
        update_schema=UserUpdate,
        joins_config=[  # Full control over joins
            JoinConfig(
                model=Tier,
                join_on=User.tier_id == Tier.id,
                join_prefix="tier_",
            ),
        ],
        nest_joins=True,
        path="/users",
        tags=["Users"],
    )
    ```

### Parameter Options

The `include_relationships` parameter accepts:

| Value | Description |
|-------|-------------|
| `False` | No automatic relationships (default) |
| `True` | Include one-to-one relationships only (safe default) |
| `["rel1", "rel2"]` | Include specific relationships by name (all types) |

!!! note "Safe Defaults for One-to-Many"
    When using `include_relationships=True`, one-to-many relationships are **excluded by default** for safety. This prevents unbounded data fetching. To include one-to-many relationships:

    ```python
    # Option 1: Set include_one_to_many=True
    router = crud_router(
        ...,
        include_relationships=True,
        include_one_to_many=True,  # Include all one-to-many relationships
    )

    # Option 2: Explicitly list the relationships you want
    router = crud_router(
        ...,
        include_relationships=["tier", "posts"],  # "posts" (one-to-many) included
    )

    # Option 3: Use joins_config for full control with nested_limit
    router = crud_router(
        ...,
        joins_config=[
            JoinConfig(
                model=Post,
                relationship_type="one-to-many",
                nested_limit=10,  # SQL-level limiting
            )
        ],
    )
    ```

The `joins_config` parameter provides manual control:

| Value | Description |
|-------|-------------|
| `None` | No manual joins (default) |
| `[JoinConfig(...)]` | List of JoinConfig objects for explicit join control |

!!! warning "Mutual Exclusivity"
    You cannot use both `include_relationships` and `joins_config` together:

    ```python
    # ❌ This raises ValueError
    router = crud_router(
        ...,
        include_relationships=True,
        joins_config=[JoinConfig(...)],  # Error!
    )
    ```

!!! tip "Error Messages for Invalid Relationships"
    If you pass a relationship name that doesn't exist on the model, FastCRUD raises a helpful error:

    ```python
    router = crud_router(
        ...,
        include_relationships=["nonexistent"],  # Raises ValueError
    )
    # ValueError: Invalid relationship name(s): ['nonexistent'].
    # Available relationships on 'User': ['department', 'tasks', 'tier'].
    # Tip: Use `include_relationships=True` to include all relationships,
    # or check your model's relationship definitions.
    ```

### Response Format

When `include_relationships=True`, read endpoints return related data:

**Without relationships:**
```json
GET /users/1
{
  "id": 1,
  "name": "Alice Smith",
  "tier_id": 3,
  "department_id": 5
}
```

**With relationships (`nest_joins=True`):**
```json
GET /users/1
{
  "id": 1,
  "name": "Alice Smith",
  "tier_id": 3,
  "tier": {
    "id": 3,
    "name": "Premium"
  },
  "department_id": 5,
  "department": {
    "id": 5,
    "name": "Engineering"
  }
}
```

**With relationships (`nest_joins=False`):**
```json
GET /users/1
{
  "id": 1,
  "name": "Alice Smith",
  "tier_id": 3,
  "tier_name": "Premium",
  "department_id": 5,
  "department_name": "Engineering"
}
```

### Affected Endpoints

Relationship data is included in:

- ✅ **Read Item** (`GET /users/{id}`) - Returns single item with relationships
- ✅ **Read Items** (`GET /users`) - Returns list with relationships for each item
- ✅ **Create Item** (`POST /users`) - Returns created item with relationships
- ✅ **Update Item** (`PATCH /users/{id}`) - Returns updated item with relationships
- ✅ **Delete Item** (`DELETE /users/{id}`) - Returns deleted item with relationships
- ✅ **DB Delete Item** (`DELETE /users/{id}/db`) - Returns item before deletion with relationships

### How It Works

Under the hood, `include_relationships=True`:

1. Uses FastCRUD's `auto_detect_relationships` feature
2. Automatically discovers all SQLAlchemy relationships on the model
3. Applies left joins to include related data
4. Works seamlessly with pagination, sorting, and filtering

For more details on auto-detection, see [Auto-Detecting Relationships](joins.md#auto-detecting-relationships).

### Performance Considerations

!!! warning "Database Query Performance"
    Including relationships affects query performance:

    - **Read operations**: Each relationship adds a LEFT JOIN to your queries. More JOINs = slower queries and larger responses.
    - **Write operations (create/update/delete)**: These perform an **additional SELECT query** after the write to return the result with relationships. This means:
        - `POST /users` → 1 INSERT + 1 SELECT with JOINs
        - `PATCH /users/{id}` → 1 UPDATE + 1 SELECT with JOINs
        - `DELETE /users/{id}` → 1 SELECT with JOINs + 1 DELETE

    **Performance Tips:**

    - Use `include_relationships=["rel1"]` to include only the relationships you need
    - Use `joins_config` for full control over join conditions and schemas
    - For high-throughput write endpoints, consider disabling relationships entirely


## Extending `EndpointCreator`

You can create a subclass of `EndpointCreator` and override or add new methods to define custom routes. Here's an example:

### Creating a Custom `EndpointCreator`

```python hl_lines="3 4"
from fastcrud import EndpointCreator

# Define the custom EndpointCreator
class MyCustomEndpointCreator(EndpointCreator):
    # Add custom routes or override existing methods
    def _custom_route(self):
        async def custom_endpoint():
            # Custom endpoint logic
            return {"message": "Custom route"}

        return custom_endpoint

    # override add_routes_to_router to also add the custom routes
    def add_routes_to_router(self, ...):
        # First, add standard CRUD routes if you want them
        super().add_routes_to_router(...)

        # Now, add custom routes
        self.router.add_api_route(
            path="/custom",
            endpoint=self._custom_route(),
            methods=["GET"],
            tags=self.tags,
            # Other parameters as needed
        )
```

### Adding custom routes

```python hl_lines="5-11"
from fastcrud import EndpointCreator

# Define the custom EndpointCreator
class MyCustomEndpointCreator(EndpointCreator):
    # Add custom routes or override existing methods
    def _custom_route(self):
        async def custom_endpoint():
            # Custom endpoint logic
            return {"message": "Custom route"}

        return custom_endpoint

    # override add_routes_to_router to also add the custom routes
    def add_routes_to_router(self, ...):
        # First, add standard CRUD routes if you want them
        super().add_routes_to_router(...)

        # Now, add custom routes
        self.router.add_api_route(
            path="/custom",
            endpoint=self._custom_route(),
            methods=["GET"],
            tags=self.tags,
            # Other parameters as needed
        )
```

### Overriding `add_routes_to_router`

```python hl_lines="13-25"
from fastcrud import EndpointCreator

# Define the custom EndpointCreator
class MyCustomEndpointCreator(EndpointCreator):
    # Add custom routes or override existing methods
    def _custom_route(self):
        async def custom_endpoint():
            # Custom endpoint logic
            return {"message": "Custom route"}

        return custom_endpoint

    # override add_routes_to_router to also add the custom routes
    def add_routes_to_router(self, ...):
        # First, add standard CRUD routes if you want them
        super().add_routes_to_router(...)

        # Now, add custom routes
        self.router.add_api_route(
            path="/custom",
            endpoint=self._custom_route(),
            methods=["GET"],
            tags=self.tags,
            # Other parameters as needed
        )
```

### Using the Custom EndpointCreator

```python hl_lines="6 15 18"
# Assuming MyCustomEndpointCreator was created

...

# Use the custom EndpointCreator with crud_router
my_router = crud_router(
    session=get_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    crud=FastCRUD(MyModel),
    path="/mymodel",
    tags=["MyModel"],
    included_methods=["create", "read", "update"],  # Including selective methods
    endpoint_creator=MyCustomEndpointCreator,
)

app.include_router(my_router)
```

## Reusing Pagination Query Parameters

FastCRUD provides a `PaginatedRequestQuery` Pydantic model that encapsulates all query parameters used for pagination and sorting. This model can be reused in custom endpoints using FastAPI's `Depends()`, making it easy to maintain consistent pagination behavior across your API.

!!! warning "Import Path Changed in v0.20.0"
    Until version 0.18.x, pagination utilities were imported from `fastcrud.paginated`. Starting from version 0.19.0, this import path was deprecated and was completely removed in version 0.20.0. Please update your imports to use `from fastcrud import PaginatedRequestQuery` instead.

### Using `PaginatedRequestQuery` in Custom Endpoints

The `PaginatedRequestQuery` model includes all standard pagination parameters:

- `offset` and `limit` for offset-based pagination
- `page` and `items_per_page` (alias: `itemsPerPage`) for page-based pagination
- `sort` for sorting by one or more fields

Here's how to use it in a custom endpoint:

```python
from typing import Annotated
from fastapi import Depends, APIRouter
from fastcrud import PaginatedRequestQuery
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

@router.get("/custom/items")
async def get_custom_items(
    db: Annotated[AsyncSession, Depends(get_session)],
    query: Annotated[PaginatedRequestQuery, Depends()],
):
    """Custom endpoint using the same pagination parameters as FastCRUD."""
    # Access pagination parameters
    if query.page is not None and query.items_per_page is not None:
        # Page-based pagination
        offset = (query.page - 1) * query.items_per_page
        limit = query.items_per_page
    else:
        # Offset-based pagination
        offset = query.offset
        limit = query.limit

    # Use offset and limit in your query
    # ... your custom logic here

    return {"offset": offset, "limit": limit, "sort": query.sort}
```

#### Query Parameter Examples

Once the `PaginatedRequestQuery` dependency is added to your endpoint, it automatically accepts the following query parameters:

**Page-based pagination:**
```bash
# Basic page-based pagination
GET /custom/items?page=2&itemsPerPage=20

# With sorting
GET /custom/items?page=1&itemsPerPage=10&sort=name,-price
```

**Offset-based pagination:**
```bash
# Basic offset-based pagination
GET /custom/items?offset=10&limit=50

# With sorting
GET /custom/items?offset=0&limit=100&sort=-created_at
```

**Sorting only:**
```bash
# Single field ascending
GET /custom/items?sort=name

# Single field descending
GET /custom/items?sort=-price

# Multiple fields mixed order
GET /custom/items?sort=category,-price,name
```

**No parameters (all optional):**
```bash
# All parameters will be None
GET /custom/items
```

### Extending `PaginatedRequestQuery`

You can also subclass `PaginatedRequestQuery` to add custom query parameters while maintaining all the standard pagination fields:

```python
from typing import Optional
from pydantic import Field
from fastcrud import PaginatedRequestQuery

class CustomPaginatedQuery(PaginatedRequestQuery):
    """Extended query with custom filter."""

    status: Optional[str] = Field(None, description="Filter by status")
    category: Optional[str] = Field(None, description="Filter by category")

@router.get("/custom/filtered-items")
async def get_filtered_items(
    db: Annotated[AsyncSession, Depends(get_session)],
    query: Annotated[CustomPaginatedQuery, Depends()],
):
    """Custom endpoint with additional filter parameters."""
    # Access both standard pagination and custom parameters
    return {
        "page": query.page,
        "items_per_page": query.items_per_page,
        "status": query.status,
        "category": query.category,
    }
```

### Benefits

Using `PaginatedRequestQuery` provides several advantages:

- **Consistency**: All endpoints use the same pagination parameter names and behavior
- **Reusability**: No need to redefine pagination parameters for each custom endpoint
- **OpenAPI Documentation**: Automatic generation of proper API documentation with field descriptions
- **Type Safety**: Full Pydantic validation for all query parameters
- **Flexibility**: Easy to extend with custom parameters while maintaining standard pagination

## Reusing Cursor Pagination Query Parameters

FastCRUD also provides a `CursorPaginatedRequestQuery` Pydantic model for cursor-based pagination. This model is ideal for large datasets and infinite scrolling features, as it provides consistent results even when data is being modified.

!!! warning "Import Path Changed in v0.20.0"
    Until version 0.18.x, pagination utilities were imported from `fastcrud.paginated`. Starting from version 0.19.0, this import path was deprecated and was completely removed in version 0.20.0. Please update your imports to use `from fastcrud import CursorPaginatedRequestQuery` instead.

### Using `CursorPaginatedRequestQuery` in Custom Endpoints

The `CursorPaginatedRequestQuery` model includes cursor-based pagination parameters:

- `cursor` - Cursor value for pagination (typically the ID of the last item from previous page)
- `limit` - Maximum number of items to return per page (default: 100, max: 1000)
- `sort_column` - Column name to sort by (default: "id")
- `sort_order` - Sort order: "asc" or "desc" (default: "asc")

Here's how to use it in a custom endpoint:

```python
from typing import Annotated
from fastapi import Depends, APIRouter
from fastcrud import CursorPaginatedRequestQuery
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

@router.get("/cursor/items")
async def get_cursor_items(
    db: Annotated[AsyncSession, Depends(get_session)],
    query: Annotated[CursorPaginatedRequestQuery, Depends()],
):
    """Custom endpoint using cursor-based pagination."""
    # Use the cursor parameters in your query logic
    items = await some_crud.get_multi_by_cursor(
        db,
        cursor=query.cursor,
        limit=query.limit,
        sort_column=query.sort_column,
        sort_order=query.sort_order,
    )
    
    return items
```

#### Cursor Pagination Query Parameter Examples

Once the `CursorPaginatedRequestQuery` dependency is added to your endpoint, it automatically accepts the following query parameters:

**Basic cursor pagination:**
```bash
# First page (no cursor)
GET /cursor/items?limit=20

# Next page using cursor from previous response
GET /cursor/items?cursor=123&limit=20
```

**With custom sorting:**
```bash
# Sort by created_at in descending order
GET /cursor/items?sort_column=created_at&sort_order=desc&limit=50

# Sort by name in ascending order with cursor
GET /cursor/items?cursor=456&sort_column=name&sort_order=asc&limit=25
```

**All parameters:**
```bash
# Full cursor pagination with all parameters
GET /cursor/items?cursor=789&limit=100&sort_column=updated_at&sort_order=desc
```

**Default behavior:**
```bash
# All parameters will use defaults: limit=100, sort_column="id", sort_order="asc"
GET /cursor/items
```

### Extending `CursorPaginatedRequestQuery`

You can also subclass `CursorPaginatedRequestQuery` to add custom query parameters:

```python
from typing import Optional
from pydantic import Field
from fastcrud import CursorPaginatedRequestQuery

class CustomCursorQuery(CursorPaginatedRequestQuery):
    """Extended cursor query with custom filters."""

    status: Optional[str] = Field(None, description="Filter by status")
    category: Optional[str] = Field(None, description="Filter by category")

@router.get("/cursor/filtered-items")
async def get_filtered_cursor_items(
    db: Annotated[AsyncSession, Depends(get_session)],
    query: Annotated[CustomCursorQuery, Depends()],
):
    """Custom endpoint with additional filter parameters."""
    # Access both cursor pagination and custom parameters
    return {
        "cursor": query.cursor,
        "limit": query.limit,
        "sort_column": query.sort_column,
        "sort_order": query.sort_order,
        "status": query.status,
        "category": query.category,
    }
```

### Benefits of Cursor Pagination

- **Consistent results**: Data modifications don't affect pagination consistency
- **Performance**: Efficient for large datasets
- **Real-time data**: Perfect for infinite scrolling and real-time feeds
- **No page drift**: Unlike offset-based pagination, items won't be skipped or duplicated when data changes

## Custom Soft Delete

To implement custom soft delete columns using `EndpointCreator` and `crud_router` in FastCRUD, you need to specify the names of the columns used for indicating deletion status and the deletion timestamp in your model. FastCRUD provides flexibility in handling soft deletes by allowing you to configure these column names directly when setting up CRUD operations or API endpoints.

Here's how to specify custom soft delete columns when utilizing `EndpointCreator` and `crud_router`:

### Defining Models with Custom Soft Delete Columns

First, ensure your SQLAlchemy model is equipped with the custom soft delete columns. Here's an example model with custom columns for soft deletion:

```python
--8<--
fastcrud/examples/mymodel/model.py:imports
fastcrud/examples/mymodel/model.py:model_softdelete
--8<--
```

And a schema necessary to activate the soft delete endpoint:

```python
--8<--
fastcrud/examples/mymodel/schemas.py:deleteschema
--8<--
```

### Using `EndpointCreator` and `crud_router` with Custom Soft Delete or Update Columns

When initializing `crud_router` or creating a custom `EndpointCreator`, you can pass the names of your custom soft delete columns through the `FastCRUD` initialization. This informs FastCRUD which columns to check and update for soft deletion operations.

Here's an example of using `crud_router` with custom soft delete columns:

```python hl_lines="11-15 23"
from fastapi import FastAPI
from fastcrud import FastCRUD, crud_router
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

# Assuming async_session is your AsyncSession generator
# and MyModel is your SQLAlchemy model

# Initialize FastCRUD with custom soft delete columns
my_model_crud = FastCRUD(
    MyModel,
    is_deleted_column='archived',  # Custom 'is_deleted' column name
    deleted_at_column='archived_at',  # Custom 'deleted_at' column name
)

# Setup CRUD router with the FastCRUD instance
app.include_router(crud_router(
    session=async_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    crud=my_model_crud,
    delete_schema=DeleteMyModelSchema,
    path="/mymodel",
    tags=["MyModel"],
))
```

You may also directly pass the names of the columns to `crud_router` or `EndpointCreator`:

```python hl_lines="9 10"
app.include_router(endpoint_creator(
    session=async_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    delete_schema=DeleteMyModelSchema,
    path="/mymodel",
    tags=["MyModel"],
    is_deleted_column='archived',
    deleted_at_column='archived_at',
))
```

This setup ensures that the soft delete functionality within your application utilizes the `archived` and `archived_at` columns for marking records as deleted, rather than the default `is_deleted` and `deleted_at` fields.

By specifying custom column names for soft deletion, you can adapt FastCRUD to fit the design of your database models, providing a flexible solution for handling deleted records in a way that best suits your application's needs.

You can also customize your `updated_at` column:

```python hl_lines="20"
--8<--
fastcrud/examples/mymodel/model.py:model
--8<--
app.include_router(endpoint_creator(
    session=async_session,
    model=MyModel,
    create_schema=CreateMyModelSchema,
    update_schema=UpdateMyModelSchema,
    delete_schema=DeleteMyModelSchema,
    path="/mymodel",
    tags=["MyModel"],
    is_deleted_column='archived',
    deleted_at_column='archived_at',
    updated_at_column='date_updated',
))
```

## Using Filters in FastCRUD

FastCRUD provides filtering capabilities, allowing you to filter query results based on various conditions. Filters can be applied to `read_multi` endpoint. This section explains how to configure and use filters in FastCRUD.

### Defining Filters

Filters are either defined using the `FilterConfig` class or just passed as a dictionary. This class allows you to specify default filter values and validate filter types. Here's an example of how to define filters for a model:

```python
from fastcrud import FilterConfig

# Define filter configuration for a model
filter_config = FilterConfig(
    tier_id=None,  # Default filter value for tier_id
    name=None,  # Default filter value for name
)
```

And the same thing using a `dict`:
```python
filter_config = {
    "tier_id": None,  # Default filter value for tier_id
    "name": None,  # Default filter value for name
}
```

By using `FilterConfig` you get better error messages.

### Applying Filters to Endpoints

You can apply filters to your endpoints by passing the `filter_config` to the `crud_router` or `EndpointCreator`. Here's an example:

```python
from fastcrud import crud_router

from .database import async_session
from .yourmodel.model import YourModel
from .yourmodel.schemas import CreateYourModelSchema, UpdateYourModelSchema

# Apply filters using crud_router
app.include_router(
    crud_router(
        session=async_session,
        model=YourModel,
        create_schema=CreateYourModelSchema,
        update_schema=UpdateYourModelSchema,
        path="/yourmodel",
        tags=["YourModel"],
        filter_config=filter_config,  # Apply the filter configuration
    ),
)
```

### Dependency-Based Filtering

FastCRUD also supports dependency-based filtering, allowing you to automatically filter query results based on values from dependencies. This is particularly useful for implementing row-level access control, where users should only see data that belongs to their organization or tenant.

```python
from fastapi import Depends
from fastcrud import crud_router, FilterConfig

# Define a dependency that returns the user's organization ID
async def get_auth_user():
    # Your authentication logic here
    return UserInfo(organization_id=123)

async def get_org_id(auth: UserInfo = Depends(get_auth_user)):
    return auth.organization_id

# Create a router with dependency-based filtering
epc_router = crud_router(
    session=async_session,
    model=ExternalProviderConfig,
    create_schema=ExternalProviderConfigSchema,
    update_schema=ExternalProviderConfigSchema,
    path="/external_provider_configs",
    filter_config=FilterConfig(
        organization_id=get_org_id,  # This will be resolved at runtime
    ),
    tags=["external_provider_configs"],
)

app.include_router(epc_router)
```

In this example, the `get_org_id` dependency will be called for each request, and the returned value will be used to filter the results by `organization_id`.

For more details on dependency-based filtering, see the [Dependency-Based Filtering](dependency_filtering.md) documentation.

### Using Filters in Requests

Once filters are configured, you can use them in your API requests. Filters are passed as query parameters. Here's an example of how to use filters in a request to a paginated endpoint:

```http
GET /yourmodel?page=1&itemsPerPage=3&tier_id=1&name=Alice
```

### Custom Filter Validation

The `FilterConfig` class includes a validator to check filter types. If an invalid filter type is provided, a `ValueError` is raised. You can customize the validation logic by extending the `FilterConfig` class:

```python
from fastcrud import FilterConfig
from pydantic import ValidationError

class CustomFilterConfig(FilterConfig):
    @field_validator("filters")
    def check_filter_types(cls, filters: dict[str, Any]) -> dict[str, Any]:
        for key, value in filters.items():
            if not isinstance(value, (type(None), str, int, float, bool)):
                raise ValueError(f"Invalid default value for '{key}': {value}")
        return filters

try:
    # Example of invalid filter configuration
    invalid_filter_config = CustomFilterConfig(invalid_field=[])
except ValidationError as e:
    print(e)
```

### Handling Invalid Filter Columns

FastCRUD ensures that filters are applied only to valid columns in your model. If an invalid filter column is specified, a `ValueError` is raised:

```python
try:
    # Example of invalid filter column
    invalid_filter_config = FilterConfig(non_existent_column=None)
except ValueError as e:
    print(e)  # Output: Invalid filter column 'non_existent_column': not found in model
```

## Sorting Results

FastCRUD automatically provides sorting functionality for the "read multiple" endpoint through the `sort` query parameter. This allows clients to control the ordering of returned results.

### Basic Sorting

Sort by a single field in ascending order:
```http
GET /items?sort=name
```

Sort by a single field in descending order (use `-` prefix):
```http
GET /items?sort=-price
```

### Multi-field Sorting

Sort by multiple fields by separating them with commas:
```http
GET /items?sort=category,name
```

Mix ascending and descending orders:
```http
GET /items?sort=category,-price,name
```
This sorts by:
1. `category` (ascending)
2. `price` (descending)
3. `name` (ascending)

### Sorting Format

The sort parameter accepts the following format:
- Field names separated by commas: `field1,field2,field3`
- Prefix with `-` for descending order: `-field1,field2,-field3`
- No spaces around commas
- Field names must match your model's column names

### Error Handling

If you specify an invalid column name that doesn't exist in your model, FastCRUD will return a 400 Bad Request error with details about the invalid column.

### Combining with Other Parameters

Sorting can be combined with pagination and filtering:

```http
GET /items?sort=-created_at&page=1&itemsPerPage=10&category=Books
```

This example:
- Sorts by `created_at` in descending order (newest first)
- Returns the first page with 10 items per page
- Filters for items in the "Books" category

## Server-Side Field Injection

FastCRUD provides automatic field injection through `CreateConfig`, `UpdateConfig`, and `DeleteConfig` classes. This feature allows you to automatically inject values for fields before data is written to the database, perfect for authentication context, timestamps, and audit fields.

### When to Use Server-Side Field Injection

This feature is essential for building secure, multi-user applications where you need to:

#### 1. **Prevent Security Vulnerabilities**
When you have user-specific data that should never be set by the frontend:

```python
# SECURITY RISK: Frontend can set any user_id
class ItemCreateSchema(BaseModel):
    name: str
    user_id: int  # Frontend could submit ANY user ID!

# SECURE: user_id automatically injected from auth context
class ItemCreateSchema(BaseModel):
    name: str
    # user_id excluded - comes from authentication

create_config = CreateConfig(
    auto_fields={
        "user_id": get_current_user_id,  # From JWT/session
    },
    exclude_from_schema=["user_id"]  # Hidden from API docs
)
```

#### 2. **Automatic Audit Trails**
For compliance and debugging, automatically track who changed what and when:

```python
# Automatically adds audit fields to every record
create_config = CreateConfig(
    auto_fields={
        "created_by": get_current_user_id,
        "created_at": get_current_timestamp,
        "ip_address": get_client_ip,
        "user_agent": get_user_agent,
    },
    exclude_from_schema=["created_by", "created_at", "ip_address", "user_agent"]
)

update_config = UpdateConfig(
    auto_fields={
        "updated_by": get_current_user_id,
        "updated_at": get_current_timestamp,
    },
    exclude_from_schema=["updated_by", "updated_at"]
)
```

#### 3. **Multi-Tenant Applications**
Automatically scope data to the correct organization/tenant:

```python
async def get_current_org_id(user: User = Depends(get_current_user)):
    return user.organization_id

create_config = CreateConfig(
    auto_fields={
        "organization_id": get_current_org_id,  # Auto-scoped
        "user_id": lambda user=Depends(get_current_user): user.id,
    },
    exclude_from_schema=["organization_id", "user_id"]
)

# Combined with filtering - users only see their org's data
router = crud_router(
    # ... other params
    create_config=create_config,
    filter_config=FilterConfig(organization_id=get_current_org_id),
)
```

#### 4. **Workflow and Status Management**
Automatically set status fields based on business logic:

```python
def get_initial_status():
    return "pending_review"

async def get_approver_id(user: User = Depends(get_current_user)):
    if user.role == "admin":
        return None  # Admins auto-approve
    return user.manager_id

create_config = CreateConfig(
    auto_fields={
        "status": get_initial_status,
        "assigned_to": get_approver_id,
        "submitted_by": get_current_user_id,
    }
)
```

#### 5. **Preventing Data Tampering**
When you need to ensure certain fields can only be set server-side:

```python
# Frontend cannot manipulate pricing or financial data
create_config = CreateConfig(
    auto_fields={
        "price": calculate_dynamic_price,     # Based on business rules
        "discount": get_user_discount_rate,   # Based on user tier
        "tax_rate": get_applicable_tax_rate,  # Based on location
    },
    exclude_from_schema=["price", "discount", "tax_rate"]
)
```

## Common Use Cases for Server-Side Field Injection

Server-side field injection is particularly valuable in these common scenarios. The following table provides a quick overview of the most frequent use cases and their solutions:

| Scenario | Problem | Solution |
|----------|---------|----------|
| **User Items** | Frontend could create items for other users | Auto-inject `user_id` from authentication |
| **Audit Logs** | Manual timestamp/user tracking is error-prone | Auto-inject `created_by`, `created_at`, etc. |
| **Multi-Tenant** | Users could access other organizations' data | Auto-inject `organization_id` + filter by it |
| **Approval Workflows** | Need to track who submitted what when | Auto-inject submitter, timestamp, initial status |
| **Financial Data** | Pricing must be server-controlled | Auto-calculate and inject prices, taxes, discounts |
| **Content Moderation** | All posts need initial review status | Auto-inject `status: "pending"` |

These scenarios address security and data integrity requirements where client-provided data cannot be trusted or where server-side business logic must control field values. Automatic field injection prevents manual validation errors and ensures consistent data handling.

Below are detailed examples and explanations for each scenario:

### User-Scoped Data
**Problem**: Frontend applications can potentially create data for other users by manipulating user IDs in requests.

**Solution**: Auto-inject the authenticated user's ID from the session/JWT token.

```python
create_config = CreateConfig(
    auto_fields={"user_id": get_current_user_id},
    exclude_from_schema=["user_id"]
)
```

This ensures that when a user creates an item, it's automatically associated with their account, preventing unauthorized data creation.

### Audit Trail Requirements
**Problem**: Manual tracking of who modified what and when is error-prone and often forgotten.

**Solution**: Automatically inject audit fields for every operation.

```python
create_config = CreateConfig(
    auto_fields={
        "created_by": get_current_user_id,
        "created_at": get_current_timestamp,
        "created_ip": get_client_ip,
    }
)

update_config = UpdateConfig(
    auto_fields={
        "updated_by": get_current_user_id, 
        "updated_at": get_current_timestamp,
    }
)
```

Perfect for compliance requirements, debugging, and maintaining data lineage.

### Multi-Tenant Applications
**Problem**: In multi-tenant systems, users could potentially access or modify data from other organizations.

**Solution**: Auto-inject organization/tenant identifiers and combine with filtering.

```python
create_config = CreateConfig(
    auto_fields={"organization_id": get_current_org_id},
    exclude_from_schema=["organization_id"]
)

# Also filter reads by organization
router = crud_router(
    # ... other config
    filter_config=FilterConfig(organization_id=get_current_org_id)
)
```

This creates complete data isolation between tenants automatically.

### Approval and Workflow Systems
**Problem**: Tracking submission details, approval status, and workflow state requires consistent data entry.

**Solution**: Auto-inject workflow metadata based on business rules.

```python
async def determine_initial_status(user = Depends(get_current_user)):
    if user.role == "admin":
        return "approved"
    return "pending_review"

create_config = CreateConfig(
    auto_fields={
        "status": determine_initial_status,
        "submitted_by": get_current_user_id,
        "submitted_at": get_current_timestamp,
    }
)
```

Ensures consistent workflow state management without manual intervention.

### Financial and Pricing Data
**Problem**: Pricing, taxes, and financial calculations must be controlled server-side to prevent manipulation.

**Solution**: Auto-calculate and inject financial fields based on business logic.

```python
async def calculate_price(item_data, user = Depends(get_current_user)):
    base_price = get_base_price(item_data.product_id)
    user_discount = get_user_discount_rate(user.tier)
    return base_price * (1 - user_discount)

create_config = CreateConfig(
    auto_fields={
        "price": calculate_price,
        "tax_rate": get_applicable_tax_rate,
        "currency": lambda: "USD",
    },
    exclude_from_schema=["price", "tax_rate", "currency"]
)
```

Prevents price manipulation while ensuring accurate calculations.

### Content Moderation
**Problem**: User-generated content needs consistent initial moderation status and safety checks.

**Solution**: Auto-inject moderation fields and initial review status.

```python
async def get_initial_moderation_status(content: str):
    if contains_flagged_content(content):
        return "requires_review"
    return "approved"

create_config = CreateConfig(
    auto_fields={
        "moderation_status": get_initial_moderation_status,
        "flagged_at": lambda: None,  # Will be set later if flagged
        "reviewed_by": lambda: None,  # Will be set when reviewed
    }
)
```

Ensures all content goes through proper moderation workflows automatically.

### Real-World Example

Here's a complete example for a multi-user blog application:

```python
from datetime import datetime
from fastapi import Depends, HTTPException
from fastcrud import crud_router, CreateConfig, UpdateConfig, FilterConfig

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = await verify_jwt_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    return user

# Auto-field functions
async def get_user_id(user = Depends(get_current_user)):
    return user.id

async def get_user_org(user = Depends(get_current_user)):
    return user.organization_id

def get_timestamp():
    return datetime.utcnow()

def get_initial_post_status():
    return "draft"  # All posts start as drafts

# Schema - notice what's missing!
class PostCreateSchema(BaseModel):
    title: str
    content: str
    # NO user_id, organization_id, timestamps, etc.

# Configuration
create_config = CreateConfig(
    auto_fields={
        "author_id": get_user_id,           # Who wrote it
        "organization_id": get_user_org,     # Which org
        "status": get_initial_post_status,   # Initial state
        "created_at": get_timestamp,         # When
        "version": lambda: 1,                # Version control
    },
    exclude_from_schema=[
        "author_id", "organization_id", "status",
        "created_at", "version"
    ]
)

update_config = UpdateConfig(
    auto_fields={
        "updated_at": get_timestamp,
        "updated_by": get_user_id,
    },
    exclude_from_schema=["updated_at", "updated_by"]
)

# Router with automatic security
router = crud_router(
    session=get_db,
    model=Post,
    create_schema=PostCreateSchema,
    update_schema=PostUpdateSchema,
    create_config=create_config,
    update_config=update_config,
    # Users only see posts from their organization
    filter_config=FilterConfig(organization_id=get_user_org),
)
```

**Result**:

- Users can only create posts for themselves
- All posts are automatically scoped to their organization
- Complete audit trail is maintained
- Frontend cannot tamper with sensitive fields
- Clean API schema without clutter

### Auto Field Injection

Auto fields are automatically injected values that are added to your data before it's written to the database. These values are provided by callable functions that can use FastAPI's dependency injection system.

#### CreateConfig

Use `CreateConfig` to inject fields during create operations:

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

#### UpdateConfig

Use `UpdateConfig` to inject fields during update operations:

```python
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

#### DeleteConfig

Use `DeleteConfig` to inject fields during soft delete operations:

```python
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

### Schema Exclusion

The `exclude_from_schema` parameter removes fields from the request schema, preventing them from appearing in API documentation and ensuring clients cannot manually set these values:

```python
create_config = CreateConfig(
    auto_fields={
        "user_id": get_current_user_id,
        "created_at": get_current_timestamp,
    },
    exclude_from_schema=["user_id", "created_at"]  # Hidden from API docs
)
```

### Authorization and Validation

Auto fields can include authorization checks:

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
        "deleted_by": check_can_delete,  # Includes auth check
    }
)
```

### Multiple Auto Fields

You can inject multiple fields simultaneously:

```python
create_config = CreateConfig(
    auto_fields={
        "user_id": get_current_user_id,
        "organization_id": get_current_org_id,
        "created_at": get_current_timestamp,
        "created_by": get_current_user_id,
        "version": lambda: "1.0",
    }
)
```

### Using with EndpointCreator

All configuration classes work with both `crud_router` and `EndpointCreator`:

```python
from fastcrud import EndpointCreator

endpoint_creator = EndpointCreator(
    session=get_db,
    model=Item,
    create_schema=CreateItemSchema,
    update_schema=UpdateItemSchema,
    create_config=create_config,
    update_config=update_config,
    delete_config=delete_config,
)

endpoint_creator.add_routes_to_router()
app.include_router(endpoint_creator.router, prefix="/items")
```

### Benefits

- **Security**: Prevent clients from setting sensitive fields like user_id or audit timestamps
- **Automation**: Automatically populate fields without manual intervention
- **Consistency**: Ensure all records have proper audit trails and ownership
- **Flexibility**: Use FastAPI's dependency injection for complex value resolution
- **Clean APIs**: Keep auto-injected fields hidden from API documentation

## Conclusion

The `EndpointCreator` class in FastCRUD offers flexibility and control over CRUD operations and custom endpoint creation. By extending this class or using the `included_methods` and `deleted_methods` parameters, you can tailor your API's functionality to your specific requirements, ensuring a more customizable and streamlined experience.
