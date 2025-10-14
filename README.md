<p align="center">
  <a href="https://benavlabs.github.io/fastcrud/">
    <img src="https://github.com/igorbenav/fastcrud/blob/main/assets/fastcrud.png?raw=true" alt="FastCRUD written in white with a drawing of a gear and inside this gear a bolt." width="45%" height="auto">
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
<a href="https://codecov.io/gh/benavlabs/fastcrud" >
  <img src="https://codecov.io/gh/benavlabs/fastcrud/graph/badge.svg?token=J7XUP29RKU"/>
</a>
<a href="https://deepwiki.com/benavlabs/fastcrud">
  <img src="https://img.shields.io/badge/DeepWiki-1F2937.svg?logo=book&logoColor=white&labelColor=1F2937&color=34D058" alt="DeepWiki"/>
</a>
</p>
<hr>
<p align="justify">
<b>FastCRUD</b> is a Python package for <b>FastAPI</b>, offering robust async CRUD operations and flexible endpoint creation utilities, streamlined through advanced features like <b>auto-detected join</b> conditions, <b>dynamic sorting</b>, and offset and cursor <b>pagination</b>.
</p>
<p><b>Documentation</b>: <a href="https://benavlabs.github.io/fastcrud/">benavlabs.github.io/fastcrud</a></p>
<p><b>Discord Community</b>: <a href="https://discord.com/invite/TEmPs22gqB">Join our Discord server</a></p>

<p><i>Also take a look at <a href="https://github.com/benavlabs/crudadmin">CRUDAdmin - Modern admin interface for FastAPI</a></i></p>
<hr>
<h2>Features</h2>

- ⚡️ **Fully Async**: Leverages Python's async capabilities for non-blocking database operations.
- 📚 **SQLAlchemy 2.0**: Works with the latest SQLAlchemy version for robust database interactions.
- 🦾 **Powerful CRUD Functionality**: Full suite of efficient CRUD operations with support for joins.
- ⚙️ **Dynamic Query Building**: Supports building complex queries dynamically, including filtering, sorting, and pagination.
- 🤝 **Advanced Join Operations**: Facilitates performing SQL joins with other models with automatic join condition detection.
- 📖 **Built-in Offset Pagination**: Comes with ready-to-use offset pagination.
- ➤ **Cursor-based Pagination**: Implements efficient pagination for large datasets, ideal for infinite scrolling interfaces.
- 🤸‍♂️ **Modular and Extensible**: Designed for easy extension and customization to fit your requirements.
- 🛣️ **Auto-generated Endpoints**: Streamlines the process of adding CRUD endpoints with custom dependencies and configurations.

<h2>Requirements</h2>
<p>Before installing FastCRUD, ensure you have the following prerequisites:</p>
<ul>
  <li><b>Python:</b> Version 3.9 or newer.</li>
  <li><b>FastAPI:</b> FastCRUD is built to work with FastAPI, so having FastAPI in your project is essential.</li>
  <li><b>SQLAlchemy:</b> Version 2.0.21 or newer. FastCRUD uses SQLAlchemy for database operations.</li>
  <li><b>Pydantic:</b> Version 2.4.1 or newer. FastCRUD leverages Pydantic models for data validation and serialization.</li>
  <li><b>SQLAlchemy-Utils:</b> Optional, but recommended for additional SQLAlchemy utilities.</li>
</ul>

> **⚠️ Warning:**
> If you are using a non-native column type (such as those from `sqlalchemy-utils`) in your models, you may encounter a `NotImplementedError`. In such cases, you need to add a `python_type` attribute to your column type. For more details and a discussion on this issue, see [this pull request](https://github.com/kvesteri/sqlalchemy-utils/pull/779).


<h2>Installing</h2>

To install, just run:

```sh
pip install fastcrud
```

Or, if using UV:

```sh
uv add fastcrud
```

<h2>Usage</h2>

FastCRUD offers two primary ways to use its functionalities:

1. By using `crud_router` for automatic endpoint creation.
2. By integrating `FastCRUD` directly into your FastAPI endpoints for more control.

Below are examples demonstrating both approaches:

<h3>Using crud_router for Automatic Endpoint Creation</h3>

Here's a quick example to get you started:

<h4>Define Your Model and Schema</h4>

**models.py**

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
```

**schemas.py**

```python
from pydantic import BaseModel

class ItemCreateSchema(BaseModel):
    name: str
    description: str

class ItemUpdateSchema(BaseModel):
    name: str
    description: str
```

<h4>Set Up FastAPI and FastCRUD</h4>

**main.py**

```python
from typing import AsyncGenerator

from fastapi import FastAPI
from fastcrud import FastCRUD, crud_router
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from yourapp.models import Base, Item
from yourapp.schemas import ItemCreateSchema, ItemUpdateSchema

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

# CRUD router setup
item_router = crud_router(
    session=get_session,
    model=Item,
    create_schema=ItemCreateSchema,
    update_schema=ItemUpdateSchema,
    path="/items",
    tags=["Items"],
)

app.include_router(item_router)

```

<h3>Using FastCRUD in User-Defined FastAPI Endpoints</h3>

For more control over your endpoints, you can use FastCRUD directly within your custom FastAPI route functions. Here's an example:

**main.py**

```python
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastcrud import FastCRUD

from models import Base, Item
from schemas import ItemCreateSchema, ItemUpdateSchema

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

# Instantiate FastCRUD with your model
item_crud = FastCRUD(Item)

@app.post("/custom/items/")
async def create_item(
    item_data: ItemCreateSchema, db: AsyncSession = Depends(get_session)
):
    return await item_crud.create(db, item_data)

@app.get("/custom/items/{item_id}")
async def read_item(item_id: int, db: AsyncSession = Depends(get_session)):
    item = await item_crud.get(db, id=item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

# You can add more routes for update and delete operations in a similar fashion
```

In this example, we define custom endpoints for creating and reading items using FastCRUD directly, providing more flexibility in how the endpoints are structured and how the responses are handled.

To read more detailed descriptions, go to the <a href="https://benavlabs.github.io/fastcrud/">documentation</a>.

🧠 **DeepWiki Docs: [deepwiki.com/benavlabs/FastAPI-boilerplate](https://deepwiki.com/benavlabs/FastAPI-boilerplate)**

<h2>Showcase</h2>

Browse our [showcase](https://benavlabs.github.io/fastcrud/showcase/) to see projects and tutorials built with FastCRUD:

- 🚀 **Applications**: Web apps and services powered by FastCRUD
- 📖 **Open Source**: Libraries and tools built with FastCRUD
- 📝 **Tutorials**: Learn how to build with FastCRUD

<h3>Featured Projects</h3>

- **[FastAPI Boilerplate](https://github.com/benavlabs/FastAPI-boilerplate)**: Extendable async API using FastAPI, Pydantic V2, SQLAlchemy 2.0 and PostgreSQL
- **[Email Assistant API](https://github.com/igorbenav/email-assistant-api)**: Personalized email writing assistant using OpenAI
- **[SQLModel Boilerplate](https://github.com/benavlabs/SQLModel-boilerplate)**: Async API boilerplate using FastAPI, SQLModel and PostgreSQL

<h3>Share Your Project</h3>

Built something with FastCRUD? We'd love to feature it! Submit your project through our [showcase submission process](https://benavlabs.github.io/fastcrud/community/showcase_submission/).

## References

- This project was heavily inspired by CRUDBase in [`FastAPI Microservices`](https://github.com/Kludex/fastapi-microservices) by [@kludex](https://github.com/kludex).
- Thanks [@ada0l](https://github.com/ada0l) for the PyPI package name!

## Similar Projects

- **[flask-muck](https://github.com/dtiesling/flask-muck)** - _"I'd love something like this for flask"_ There you have it
- **[FastAPI CRUD Router](https://github.com/awtkns/fastapi-crudrouter)** - Supports multiple ORMs, but currently unmantained
- **[FastAPI Quick CRUD](https://github.com/LuisLuii/FastAPIQuickCRUD)** - Same purpose, but only for SQLAlchemy 1.4

## License

[`MIT`](LICENSE.md)

## Contact

Benav Labs – [benav.io](https://benav.io)
[github.com/benavlabs](https://github.com/benavlabs/)

<hr>
<a href="https://benav.io">
  <img src="docs/assets/benav_labs_banner.png" alt="Powered by Benav Labs - benav.io"/>
</a>
