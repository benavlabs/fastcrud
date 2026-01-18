# Automatic Endpoint Creation with `crud_router`

This section of the documentation explains how to use the `crud_router` utility function from the FastCRUD package for automatic endpoint creation in a FastAPI application. The `crud_router` simplifies the process of creating standard CRUD (Create, Read, Update, Delete) endpoints for your models.

## Prerequisites

Before proceeding, ensure you have FastAPI and FastCRUD installed in your environment. FastCRUD streamlines interactions with the database using SQLAlchemy models and Pydantic schemas.

---

## Using `crud_router`

### Step 1: Define Your Model and Schema

First, define your SQLAlchemy model and corresponding Pydantic schemas for creating and updating data.

???+ example "`item/model.py`"

    ```python
    --8<--
    fastcrud/examples/item/model.py:imports
    fastcrud/examples/item/model.py:model
    --8<--
    ```

???+ example "`item/schemas.py`"

    ```python
    --8<--
    fastcrud/examples/item/schemas.py:imports
    fastcrud/examples/item/schemas.py:createschema
    fastcrud/examples/item/schemas.py:updateschema
    --8<--
    ```

### Step 2: Set Up FastAPI and FastCRUD

Next, set up your FastAPI application, you can optionally set up a custom FastCRUD instance as well. This involves configuring the database connection.

```python
from typing import AsyncGenerator

from fastapi import FastAPI
from fastcrud import FastCRUD, crud_router
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Database setup (Async SQLAlchemy)
DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Database session dependency
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

# Create tables before the app start
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

# FastAPI app
app = FastAPI(lifespan=lifespan)
```

### Step 3: Use `crud_router` to Create Endpoints

```python
# CRUD router setup
item_router = crud_router(
    session=get_session,
    model=Item,
    create_schema=CreateItemSchema,
    update_schema=UpdateItemSchema,
    include_relationships=True,  # Optional: automatically include related data
    path="/items",
    tags=["Items"],
)

app.include_router(item_router)
```

### Available Endpoints

For a comprehensive list of all available endpoints, read the [advanced section](../advanced/endpoint.md).

!!! tip "Sorting Support"

    The automatically created endpoints include built-in sorting functionality. You can sort results using the `sort` query parameter:
    
    - `GET /items?sort=name` - Sort by name (ascending)
    - `GET /items?sort=-price` - Sort by price (descending) 
    - `GET /items?sort=category,-price` - Sort by category ascending, then price descending
    
    For detailed sorting documentation, see [Sorting Results](../advanced/endpoint.md#sorting-results).

!!! tip "Including Related Data"

    Set `include_relationships=True` to automatically include related models in responses. This uses FastCRUD's auto-detection feature to discover and join all SQLAlchemy relationships defined on your model.

    For more details and configuration options, see [Including Relationships in Endpoints](../advanced/endpoint.md#including-relationships-in-endpoints).

## Usage and Testing

Once the application is running, you can test the automatically created endpoints using tools like Swagger UI, which FastAPI provides by default. The endpoints for creating, reading, updating, and deleting Item instances are now accessible at `/items`.

---

## Using `EndpointCreator` Directly

Using the `EndpointCreator` class in FastCRUD is a more flexible way to add CRUD endpoints to a FastAPI application.

### Step 1: Define Your Model and Schema

Define your SQLAlchemy models and corresponding Pydantic schemas for data validation.

???+ example "`item/model.py`"

    ```python
    --8<--
    fastcrud/examples/item/model.py:imports
    fastcrud/examples/item/model.py:model
    --8<--
    ```

???+ example "`item/schemas.py`"

    ```python
    --8<--
    fastcrud/examples/item/schemas.py:imports
    fastcrud/examples/item/schemas.py:createschema
    fastcrud/examples/item/schemas.py:updateschema
    fastcrud/examples/item/schemas.py:deleteschema
    --8<--
    ```

### Step 2: Set Up FastAPI and FastCRUD

Next, set up your FastAPI application and FastCRUD instances. This involves configuring the database connection and creating a CRUD instance for your model.

```python
from typing import AsyncGenerator

from fastapi import FastAPI
from fastcrud import FastCRUD, crud_router
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Database setup (Async SQLAlchemy)
DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Database session dependency
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

# Create tables before the app start
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

# FastAPI app
app = FastAPI(lifespan=lifespan)

# CRUD operations setup
item_crud = FastCRUD(Item)
```

### Step 3: Initialize `EndpointCreator`

Create an instance of `EndpointCreator` by passing the necessary parameters, including your model, session, CRUD instance, and schemas.

```python
from fastcrud import EndpointCreator

# Initialize EndpointCreator
endpoint_creator = EndpointCreator(
    session=get_session,
    model=Item,
    create_schema=CreateItemSchema,
    update_schema=UpdateItemSchema,
    crud=item_crud,
    delete_schema=DeleteItemSchema,
    path="/itempath",
    tags=["ItemTag"]
)
```

### Step 4: Add Custom Endpoints

Add custom endpoints using `EndpointCreator`. You can inject dependencies as needed.

```python
# Example of adding custom dependencies
endpoint_creator.add_routes_to_router(
    read_deps=[custom_dependency],
    update_deps=[another_custom_dependency],
)

```

### Step 5: Include the Router in Your Application

Finally, include the router from the `EndpointCreator` in your FastAPI application.

```python
app.include_router(endpoint_creator.router)

```

## Advanced Customization

You can override the default methods in `EndpointCreator` for more control over the CRUD operations. You can also specify the operations you want to include. Read more in the [advanced section](../advanced/endpoint.md).

## Conclusion

By following these steps, you can quickly set up CRUD endpoints for your models in a FastAPI application using `crud_router` or `EndpointCreator`. This utility function reduces boilerplate code and increases development efficiency by automating the creation of standard API endpoints.
