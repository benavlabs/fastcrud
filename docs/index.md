<style>
    .md-typeset h1,
    .md-content__button {
        display: none;
    }
</style>

<p align="center">
  <a href="https://github.com/benavlabs/fastcrud">
    <img src="assets/fastcrud.png?raw=true" alt="FastCRUD written in white with a drawing of a gear and inside this gear a bolt." width="45%" height="auto">
  </a>
</p>
<p align="center" markdown=1>
  <i>Powerful CRUD methods and automatic endpoint creation for FastAPI.</i>
</p>
<p align="center" markdown=1>
<a href="https://github.com/benavlabs/fastcrud/actions/workflows/tests.yml">
  <img src="https://github.com/benavlabs/fastcrud/actions/workflows/tests.yml/badge.svg" alt="Tests"/>
</a>
<a href="https://pypi.org/project/fastcrud/">
  <img src="https://img.shields.io/pypi/v/fastcrud?color=%2334D058&label=pypi%20package" alt="PyPi Version"/>
</a>
<a href="https://pypi.org/project/fastcrud/">
  <img src="https://img.shields.io/pypi/pyversions/fastcrud.svg?color=%2334D058" alt="Supported Python Versions"/>
</a>
</a>
<a href="https://codecov.io/gh/benavlabs/fastcrud" > 
  <img src="https://codecov.io/gh/benavlabs/fastcrud/graph/badge.svg?token=J7XUP29RKU"/> 
</a>
</p>
<hr>
<p align="justify">
<b>FastCRUD</b> is a Python package for <b>FastAPI</b>, offering robust async CRUD operations and flexible endpoint creation utilities, streamlined through advanced features like <b>auto-detected join</b> conditions, <b>dynamic sorting</b>, and offset and cursor <b>pagination</b>.
</p>
<hr>

## Features

- **Fully Async**: Leverages Python's async capabilities for non-blocking database operations.
- **SQLAlchemy 2.0**: Works with the latest SQLAlchemy version for robust database interactions.
- **SQLModel Support**: You can optionally use SQLModel 0.14 or newer instead of SQLAlchemy.
- **Powerful CRUD Functionality**: Full suite of efficient CRUD operations with support for joins.
- **Dynamic Query Building**: Supports building complex queries dynamically, including filtering, sorting, and pagination.
- **Advanced Join Operations**: Facilitates performing SQL joins with other models with automatic join condition detection.
- **Built-in Offset Pagination**: Comes with ready-to-use offset pagination.
- **Cursor-based Pagination**: Implements efficient pagination for large datasets, ideal for infinite scrolling interfaces.
- **Modular and Extensible**: Designed for easy extension and customization to fit your requirements.
- **Auto-generated Endpoints**: Streamlines the process of adding CRUD endpoints with custom dependencies and configurations.
- **Clean Architecture**: Built with a six-level dependency hierarchy that prevents circular dependencies and supports future extensibility.

## Quick Start

!!! info "Using SQLModel?"
    Since SQLModel combines SQLAlchemy and Pydantic, you can replace the separate model and schema definitions below with SQLModel classes. Wherever you see SQLAlchemy models or Pydantic schemas in the documentation, just use SQLModel instead - it works seamlessly with FastCRUD.

### 1. Define Your Model and Schema

```python title="models.py"
import datetime
from sqlalchemy import Column, DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from pydantic import BaseModel

class Base(DeclarativeBase):
    pass

# SQLAlchemy Model
--8<--
fastcrud/examples/item/model.py:model
--8<--

# Pydantic Schema
--8<--
fastcrud/examples/item/schemas.py:itemschema
--8<--

# Database connection
DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

### 2. Set Up FastAPI with FastCRUD

```python title="main.py"
from typing import AsyncGenerator
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from fastcrud import crud_router

# Database session dependency
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

# Create tables before app start
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

# FastAPI app
app = FastAPI(lifespan=lifespan)

# Create CRUD router
item_router = crud_router(
    session=get_session,
    model=Item,
    create_schema=ItemSchema,
    update_schema=ItemSchema,
    path="/items",
    tags=["Items"]
)

app.include_router(item_router)
```

### 3. That's It!

Start your FastAPI application and visit `/docs` - your CRUD endpoints are automatically created and documented!

## Requirements

Before installing FastCRUD, ensure you have the following prerequisites:

* **Python:** Version 3.10 or newer.
* **FastAPI:** FastCRUD is built to work with FastAPI, so having FastAPI in your project is essential.
* **SQLAlchemy or SQLModel:** FastCRUD uses SQLAlchemy 2.0 for database operations, so you need SQLAlchemy 2.0 or newer or SQLModel 0.14 or newer.
* **Pydantic V2 or SQLModel:** FastCRUD leverages Pydantic models for data validation and serialization, so you need Pydantic 2.0 or newer or SQLModel 0.14 or newer.

## Installing

To install, just run:

```sh
pip install fastcrud
```

Or, if using UV:

```sh
uv add fastcrud
```

## Usage

FastCRUD offers two primary ways to use its functionalities:

1. By using `crud_router` for automatic endpoint creation.
2. By integrating `FastCRUD` directly into your FastAPI endpoints for more control.

Below are examples demonstrating both approaches:

### Using `crud_router` for Automatic Endpoint Creation

Here's a quick example to get you started:

#### Define Your Model and Schemas

```python title="item/model.py"
--8<--
fastcrud/examples/item/model.py:imports
fastcrud/examples/item/model.py:model
--8<--
```

```python title="item/schemas.py"
--8<--
fastcrud/examples/item/schemas.py:imports
fastcrud/examples/item/schemas.py:createschema
fastcrud/examples/item/schemas.py:updateschema
--8<--
```

#### Set Up FastAPI and FastCRUD

```python title="main.py"
from typing import AsyncGenerator

from fastapi import FastAPI
from fastcrud import crud_router
from fastcrud import FastCRUD
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .item.model import Base, Item
from .item.schemas import CreateItemSchema, UpdateItemSchema

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

# CRUD router setup
item_router = crud_router(
    session=get_session,
    model=Item,
    create_schema=CreateItemSchema,
    update_schema=UpdateItemSchema,
    crud=item_crud,
    path="/items",
    tags=["Items"],
)

app.include_router(item_router)
```

### Using FastCRUD in User-Defined FastAPI Endpoints

For more control over your endpoints, you can use FastCRUD directly within your custom FastAPI route functions. Here's an example:

```python title="api/v1/items.py" hl_lines="10 14 18"
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastcrud import FastCRUD

from .item.model import Item
from .item.schemas import CreateItemSchema, UpdateItemSchema

# Assume async_session is already set up as per the previous example

# Instantiate FastCRUD with your model
item_crud = FastCRUD(Item)

@app.post("/custom/items/")
async def create_item(item_data: CreateItemSchema, db: AsyncSession = Depends(get_session)):
    return await item_crud.create(db, item_data)

@app.get("/custom/items/{item_id}")
async def read_item(item_id: int, db: AsyncSession = Depends(get_session)):
    item = await item_crud.get(db, id=item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

# You can add more routes for update and delete operations in a similar fashion
```

## License

[`MIT`](community/LICENSE.md)

---

<div style="text-align: center; margin-top: 50px;">
    <a href="why-fastcrud/" class="md-button md-button--primary">
        Why Choose FastCRUD? →
    </a>
</div>
