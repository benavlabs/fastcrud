from collections.abc import AsyncGenerator
from typing import Optional
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import make_url, Column, String
from pydantic import ConfigDict
from sqlmodel import SQLModel, Field, Relationship
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.sql import func
from testcontainers.postgres import PostgresContainer  # type: ignore
from testcontainers.mysql import MySqlContainer  # type: ignore
from testcontainers.core.docker_client import DockerClient  # type: ignore

from fastcrud.crud.fast_crud import FastCRUD
from fastcrud.endpoint.crud_router import crud_router
from fastcrud import EndpointCreator, FilterConfig


class MultiPKModel(SQLModel, table=True):
    __tablename__ = "multi_pk"
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: Optional[str] = Field(default=None, primary_key=True, max_length=25)
    name: str = Field(index=True)
    test_id: Optional[int] = Field(default=None, foreign_key="test.id")
    test: "ModelTest" = Relationship(back_populates="multi_pk")


class CategoryModel(SQLModel, table=True):
    __tablename__ = "category"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    tests: list["ModelTest"] = Relationship(back_populates="category")


class ModelTest(SQLModel, table=True):
    __tablename__ = "test"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    tier_id: int = Field(default=None, foreign_key="tier.id")
    category_id: Optional[int] = Field(default=None, foreign_key="category.id")
    tier: "TierModel" = Relationship(back_populates="tests")
    multi_pk: "MultiPKModel" = Relationship(back_populates="test")
    category: "CategoryModel" = Relationship(back_populates="tests")
    is_deleted: bool = Field(default=False)
    deleted_at: Optional[datetime] = Field(default=None)


class ModelTestWithTimestamp(SQLModel, table=True):
    __tablename__ = "model_test_with_timestamp"
    id: int = Field(primary_key=True)
    name: str
    tier_id: Optional[int] = Field(foreign_key="tier.id")
    category_id: Optional[int] = Field(default=None, foreign_key="category.id")
    is_deleted: bool = Field(default=False)
    deleted_at: Optional[datetime] = Field(default=None)
    updated_at: datetime = Field(default=func.now(), nullable=False)


class TierModel(SQLModel, table=True):
    __tablename__ = "tier"
    id: int = Field(primary_key=True)
    name: str = Field(unique=True)
    tests: list["ModelTest"] = Relationship(back_populates="tier")


class ProjectsParticipantsAssociation(SQLModel, table=True):
    __tablename__ = "projects_participants_association"
    project_id: int = Field(foreign_key="projects.id", primary_key=True)
    participant_id: int = Field(foreign_key="participants.id", primary_key=True)


class Project(SQLModel, table=True):
    __tablename__ = "projects"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    participants: list["Participant"] = Relationship(
        back_populates="projects", link_model=ProjectsParticipantsAssociation
    )


class Participant(SQLModel, table=True):
    __tablename__ = "participants"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    role: Optional[str] = None
    projects: list["Project"] = Relationship(
        back_populates="participants", link_model=ProjectsParticipantsAssociation
    )


class Card(SQLModel, table=True):
    __tablename__ = "cards"
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    articles: list["Article"] = Relationship(back_populates="card")


class Article(SQLModel, table=True):
    __tablename__ = "articles"
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    card_id: Optional[int] = Field(foreign_key="cards.id")
    card: Optional[Card] = Relationship(back_populates="articles")


class Client(SQLModel, table=True):
    __tablename__ = "clients"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    contact: str = Field(nullable=False)
    phone: str = Field(nullable=False)
    email: str = Field(nullable=False)
    tasks: list["Task"] = Relationship(back_populates="client")
    users: list["User"] = Relationship(back_populates="company")


class Department(SQLModel, table=True):
    __tablename__ = "departments"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    tasks: list["Task"] = Relationship(back_populates="department")
    users: list["User"] = Relationship(back_populates="department")


class User(SQLModel, table=True):
    __tablename__ = "users"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    username: str = Field(nullable=False, unique=True)
    email: str = Field(nullable=False, unique=True)
    phone: Optional[str] = Field(default=None)
    profile_image_url: Optional[str] = Field(default=None)
    department_id: Optional[int] = Field(default=None, foreign_key="departments.id")
    company_id: Optional[int] = Field(default=None, foreign_key="clients.id")
    department: Optional[Department] = Relationship(back_populates="users")
    company: Optional[Client] = Relationship(back_populates="users")
    tasks: list["Task"] = Relationship(back_populates="assignee")


class Task(SQLModel, table=True):
    __tablename__ = "tasks"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    description: Optional[str] = Field(default=None)
    client_id: Optional[int] = Field(default=None, foreign_key="clients.id")
    department_id: Optional[int] = Field(default=None, foreign_key="departments.id")
    assignee_id: Optional[int] = Field(default=None, foreign_key="users.id")
    client: Optional[Client] = Relationship(back_populates="tasks")
    department: Optional[Department] = Relationship(back_populates="tasks")
    assignee: Optional[User] = Relationship(back_populates="tasks")


class CreateSchemaTest(SQLModel):
    model_config = ConfigDict(extra="forbid")  # type: ignore[assignment]
    name: str
    tier_id: int


class BookingModel(SQLModel, table=True):
    __tablename__ = "booking"
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(default=None, foreign_key="test.id")
    user_id: int = Field(default=None, foreign_key="test.id")
    booking_date: datetime
    owner: "ModelTest" = Relationship(
        sa_relationship_kwargs={"foreign_keys": "BookingModel.owner_id"}
    )
    user: "ModelTest" = Relationship(
        sa_relationship_kwargs={"foreign_keys": "BookingModel.user_id"}
    )


class ModelWithCustomColumns(SQLModel, table=True):
    __tablename__ = "test_custom"

    id: Optional[int] = Field(default=None, primary_key=True)
    meta: str = Field(sa_column=Column("metadata", String(32), nullable=False))
    name: str = Field(sa_column=Column("display_name", String(32), nullable=False))


class ReadSchemaTest(SQLModel):
    id: int
    name: str
    tier_id: int


class UpdateSchemaTest(SQLModel):
    name: str


class DeleteSchemaTest(SQLModel):
    pass


class TierSchemaTest(SQLModel):
    name: str


class TierDeleteSchemaTest(SQLModel):
    pass


class CategorySchemaTest(SQLModel):
    id: Optional[int] = None
    name: str


class BookingSchema(SQLModel):
    id: Optional[int] = None
    owner_id: int
    user_id: int
    booking_date: datetime


class MultiPkCreate(SQLModel):
    id: int
    uuid: str
    name: str
    test_id: Optional[int] = None


class MultiPkSchema(SQLModel):
    name: str
    test_id: Optional[int] = None


class ArticleSchema(SQLModel):
    id: int
    title: str
    card_id: int


class CardSchema(SQLModel):
    id: int
    title: str
    articles: Optional[list[ArticleSchema]] = []


class DepartmentRead(SQLModel):
    id: int
    name: str


class UserReadSub(SQLModel):
    id: int
    name: str
    username: str
    email: str
    phone: Optional[str]
    profile_image_url: str
    department_id: Optional[int]
    company_id: Optional[int]


class ClientRead(SQLModel):
    id: int
    name: str
    contact: str
    phone: str
    email: str


class TaskReadSub(SQLModel):
    id: int
    name: str
    description: Optional[str]


class TaskRead(TaskReadSub):
    department: Optional[DepartmentRead]
    assignee: Optional[UserReadSub]
    client: Optional[ClientRead]


def is_docker_running() -> bool:  # pragma: no cover
    try:
        DockerClient()
        return True
    except Exception:
        return False


async_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:", echo=True, future=True
)


@asynccontextmanager
async def _setup_database(url: str) -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine(url, echo=True, future=True)
    session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_maker() as session:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        try:
            yield session
        finally:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(
    request: pytest.FixtureRequest,
) -> AsyncGenerator[AsyncSession]:  # pragma: no cover
    dialect_marker = request.node.get_closest_marker("dialect")
    dialect = dialect_marker.args[0] if dialect_marker else "sqlite"

    if dialect == "postgresql" or dialect == "mysql":
        if not is_docker_running():
            pytest.skip("Docker is required, but not running")

        if dialect == "postgresql":
            with PostgresContainer() as postgres:
                url = postgres.get_connection_url()
                async with _setup_database(url) as session:
                    yield session
        elif dialect == "mysql":
            with MySqlContainer() as mysql:
                url = make_url(mysql.get_connection_url())._replace(
                    drivername="mysql+aiomysql"
                )
                async with _setup_database(url) as session:
                    yield session
    else:
        async with _setup_database("sqlite+aiosqlite:///:memory:") as session:
            yield session


@pytest.fixture(scope="function")
def test_data() -> list[dict]:
    return [
        {"id": 1, "name": "Charlie", "tier_id": 1, "category_id": 1},
        {"id": 2, "name": "Alice", "tier_id": 2, "category_id": 1},
        {"id": 3, "name": "Bob", "tier_id": 1, "category_id": 2},
        {"id": 4, "name": "David", "tier_id": 2, "category_id": 1},
        {"id": 5, "name": "Eve", "tier_id": 1, "category_id": 1},
        {"id": 6, "name": "Frank", "tier_id": 2, "category_id": 2},
        {"id": 7, "name": "Grace", "tier_id": 1, "category_id": 2},
        {"id": 8, "name": "Hannah", "tier_id": 2, "category_id": 1},
        {"id": 9, "name": "Ivan", "tier_id": 1, "category_id": 1},
        {"id": 10, "name": "Judy", "tier_id": 2, "category_id": 2},
        {"id": 11, "name": "Alice", "tier_id": 1, "category_id": 1},
    ]


@pytest.fixture(scope="function")
def test_data_tier() -> list[dict]:
    return [{"id": 1, "name": "Premium"}, {"id": 2, "name": "Basic"}]


@pytest.fixture(scope="function")
def test_data_category() -> list[dict]:
    return [{"id": 1, "name": "Tech"}, {"id": 2, "name": "Health"}]


@pytest.fixture(
    scope="function",
    params=[
        {"id": 1, "uuid": "a", "name": "Tech"},
        {"id": 1, "uuid": "b", "name": "Health"},
    ],
)
def test_data_multipk(request) -> list[dict]:
    return request.param


@pytest.fixture(scope="function")
def test_data_booking() -> list[dict]:
    return [
        {
            "id": 1,
            "owner_id": 1,
            "user_id": 2,
            "booking_date": datetime(2024, 3, 10, 15, 30),
        },
        {
            "id": 2,
            "owner_id": 1,
            "user_id": 3,
            "booking_date": datetime(2024, 3, 11, 10, 0),
        },
    ]


@pytest.fixture
def test_model():
    return ModelTest


@pytest.fixture
def tier_model():
    return TierModel


@pytest.fixture
def create_schema():
    return CreateSchemaTest


@pytest.fixture
def read_schema():
    return ReadSchemaTest


@pytest.fixture
def tier_schema():
    return TierSchemaTest


@pytest.fixture
def update_schema():
    return UpdateSchemaTest


@pytest.fixture
def delete_schema():
    return DeleteSchemaTest


@pytest.fixture
def tier_delete_schema():
    return TierDeleteSchemaTest


@pytest.fixture
def multi_pk_model():
    return MultiPKModel


@pytest.fixture
def multi_pk_test_schema():
    return MultiPkSchema


@pytest.fixture
def multi_pk_test_create_schema():
    return MultiPkCreate


@pytest.fixture
def test_model_custom_columns():
    return ModelWithCustomColumns


async def test_read_dep():
    pass


@pytest.fixture
def client(
    test_model,
    tier_model,
    multi_pk_model,
    create_schema,
    update_schema,
    delete_schema,
    tier_schema,
    tier_delete_schema,
    multi_pk_test_schema,
    multi_pk_test_create_schema,
    async_session,
) -> TestClient:
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            path="/test",
            tags=["test"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )
    )

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=tier_model,
            crud=FastCRUD(tier_model),
            create_schema=tier_schema,
            update_schema=tier_schema,
            delete_schema=tier_delete_schema,
            path="/tier",
            tags=["tier"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )
    )

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=multi_pk_model,
            crud=FastCRUD(multi_pk_model),
            create_schema=multi_pk_test_create_schema,
            update_schema=multi_pk_test_schema,
            read_deps=[test_read_dep],
            delete_schema=multi_pk_test_schema,
            path="/multi_pk",
            tags=["multi_pk"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )
    )

    return TestClient(app)


@pytest.fixture
def filtered_client(
    test_model,
    create_schema,
    update_schema,
    delete_schema,
    async_session,
):
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            filter_config=FilterConfig(tier_id=None, name=None, name__startswith=None),
            path="/test",
            tags=["test"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )
    )

    return TestClient(app)


@pytest.fixture
def dict_filtered_client(
    test_model,
    create_schema,
    update_schema,
    delete_schema,
    async_session,
):
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=FastCRUD(test_model),
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            filter_config={"tier_id": None, "name": None},
            path="/test",
            tags=["test"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )
    )

    return TestClient(app)


@pytest.fixture
def invalid_filtered_client(
    test_model,
    create_schema,
    update_schema,
    delete_schema,
    async_session,
):
    filter_config = {"invalid_column": None}

    with pytest.raises(
        ValueError, match="Invalid filter column 'invalid_column': not found in model"
    ):
        EndpointCreator(
            session=lambda: async_session,
            model=test_model,
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            filter_config=filter_config,
            path="/test",
            tags=["test"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )


@pytest.fixture
def endpoint_creator(test_model, async_session) -> EndpointCreator:
    """Fixture to create an instance of EndpointCreator."""
    return EndpointCreator(
        session=lambda: async_session,
        model=ModelTest,
        crud=FastCRUD(test_model),
        create_schema=CreateSchemaTest,
        update_schema=UpdateSchemaTest,
        delete_schema=DeleteSchemaTest,
        path="/custom_test",
        tags=["custom_test"],
        endpoint_names={
            "create": "create",
            "read": "get",
            "update": "update",
            "delete": "delete",
            "db_delete": "db_delete",
            "read_multi": "get_multi",
        },
    )


@pytest.fixture
def client_with_select_schema(
    test_model,
    create_schema,
    update_schema,
    read_schema,
    async_session,
) -> TestClient:
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            select_schema=read_schema,
            create_schema=create_schema,
            update_schema=update_schema,
            path="/test",
            tags=["test"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )
    )

    return TestClient(app)
