from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    Boolean,
    DateTime,
    make_url,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, relationship
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.sql import func
from testcontainers.postgres import PostgresContainer  # type: ignore
from testcontainers.mysql import MySqlContainer  # type: ignore
from testcontainers.core.docker_client import DockerClient  # type: ignore

from fastcrud.crud.fast_crud import FastCRUD
from fastcrud.endpoint.crud_router import crud_router
from fastcrud import EndpointCreator, FilterConfig


class Base(DeclarativeBase):
    pass


class MultiPkModel(Base):
    __tablename__ = "multi_pk"
    id = Column(Integer, primary_key=True)
    uuid = Column(String(32), primary_key=True)
    name = Column(String(32), unique=True)
    test_id = Column(Integer, ForeignKey("test.id"))
    test = relationship("ModelTest", back_populates="multi_pk")


class CategoryModel(Base):
    __tablename__ = "category"
    tests = relationship("ModelTest", back_populates="category")
    id = Column(Integer, primary_key=True)
    name = Column(String(32), unique=True)


class ModelTest(Base):
    __tablename__ = "test"
    id = Column(Integer, primary_key=True)
    name = Column(String(32))
    tier_id = Column(Integer, ForeignKey("tier.id"))
    category_id = Column(
        Integer, ForeignKey("category.id"), nullable=True, default=None
    )
    tier = relationship("TierModel", back_populates="tests")
    category = relationship("CategoryModel", back_populates="tests")
    multi_pk = relationship("MultiPkModel", back_populates="test")
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True, default=None)


class ModelTestWithTimestamp(Base):
    __tablename__ = "model_test_with_timestamp"
    id = Column(Integer, primary_key=True)
    name = Column(String(32))
    tier_id = Column(Integer, ForeignKey("tier.id"))
    category_id = Column(
        Integer, ForeignKey("category.id"), nullable=True, default=None
    )
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True, default=None)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )


class TierModel(Base):
    __tablename__ = "tier"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), unique=True)
    tests = relationship("ModelTest", back_populates="tier")


class BookingModel(Base):
    __tablename__ = "booking"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("test.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("test.id"), nullable=False)
    booking_date = Column(DateTime, nullable=False)
    owner = relationship("ModelTest", foreign_keys=[owner_id], backref="owned_bookings")
    user = relationship("ModelTest", foreign_keys=[user_id], backref="user_bookings")


# --8<-- [start:model_project]
class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False)
    description = Column(String(32))
    participants = relationship(
        "Participant",
        secondary="projects_participants_association",
        back_populates="projects",
    )


# --8<-- [end:model_project]
# --8<-- [start:model_participant]
class Participant(Base):
    __tablename__ = "participants"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False)
    role = Column(String(32))
    projects = relationship(
        "Project",
        secondary="projects_participants_association",
        back_populates="participants",
    )


# --8<-- [end:model_participant]
# --8<-- [start:model_proj_parts_assoc]
class ProjectsParticipantsAssociation(Base):
    __tablename__ = "projects_participants_association"
    project_id = Column(Integer, ForeignKey("projects.id"), primary_key=True)
    participant_id = Column(Integer, ForeignKey("participants.id"), primary_key=True)


# --8<-- [end:model_proj_parts_assoc]
class Card(Base):
    __tablename__ = "cards"
    id = Column(Integer, primary_key=True)
    title = Column(String(32))


class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True)
    name = Column(String(32))
    articles = relationship("Article", back_populates="author")


class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True)
    title = Column(String(32))
    content = Column(String(100))
    published_date = Column(String(32))
    author_id = Column(Integer, ForeignKey("authors.id"))
    author = relationship("Author", back_populates="articles")
    card_id = Column(Integer, ForeignKey("cards.id"), nullable=True)
    card = relationship("Card", back_populates="articles")


Card.articles = relationship("Article", order_by=Article.id, back_populates="card")


class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False)
    contact = Column(String(32), nullable=False)
    phone = Column(String(32), nullable=False)
    email = Column(String(32), nullable=False)


class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False)
    username = Column(String(32), nullable=False, unique=True)
    email = Column(String(32), nullable=False, unique=True)
    phone = Column(String(32), nullable=True)
    profile_image_url = Column(String(32), nullable=True)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    company_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    department = relationship("Department", backref="users")
    company = relationship("Client", backref="users")


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False)
    description = Column(String(32), nullable=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    assignee_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    client = relationship("Client", backref="tasks")
    department = relationship("Department", backref="tasks")
    assignee = relationship("User", backref="tasks")


class ModelWithOrgTest(Base):
    __tablename__ = "test_with_org"
    id = Column(Integer, primary_key=True)
    name = Column(String(32))
    tier_id = Column(Integer)
    category_id = Column(Integer, nullable=True, default=None)
    organization_id = Column(Integer, nullable=True, default=None)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True, default=None)


# Models for testing joined model filtering
class Company(Base):
    __tablename__ = "company"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    industry = Column(String(50))
    users = relationship("UserModel", back_populates="company")


class UserModel(Base):
    __tablename__ = "user_model"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    company = relationship("Company", back_populates="users")


class ModelWithCustomColumns(Base):
    __tablename__ = "test_custom"

    id = Column(Integer, primary_key=True)
    meta = Column("metadata", String(32), nullable=False)
    name = Column("display_name", String(32), nullable=False)


class CreateSchemaTest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    tier_id: int
    category_id: int | None = None


class ReadSchemaTest(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: int | None


class UpdateSchemaTest(BaseModel):
    name: str


class DeleteSchemaTest(BaseModel):
    pass


class TierSchemaTest(BaseModel):
    name: str


class TierDeleteSchemaTest(BaseModel):
    pass


class CategorySchemaTest(BaseModel):
    id: int | None = None
    name: str


class MultiPkSchema(BaseModel):
    id: int
    uuid: str
    name: str


class MultiPkCreate(MultiPkSchema):
    pass


class BookingSchema(BaseModel):
    id: int | None = None
    owner_id: int
    user_id: int
    booking_date: datetime


class ArticleSchema(BaseModel):
    id: int
    title: str
    content: str | None = None
    published_date: str | None = None
    author_id: int | None = None
    card_id: int | None = None


class AuthorSchema(BaseModel):
    id: int
    name: str
    articles: list[ArticleSchema] | None = []


class CardSchema(BaseModel):
    id: int
    title: str
    articles: list[ArticleSchema] | None = []


class DepartmentRead(BaseModel):
    id: int
    name: str


class UserReadSub(BaseModel):
    id: int
    name: str
    username: str
    email: str
    phone: str | None
    profile_image_url: str
    department_id: int | None
    company_id: int | None


class ClientRead(BaseModel):
    id: int
    name: str
    contact: str
    phone: str
    email: str


class TaskReadSub(BaseModel):
    id: int
    name: str
    description: str | None


class TaskRead(TaskReadSub):
    department: DepartmentRead | None
    assignee: UserReadSub | None
    client: ClientRead | None


# Schemas for joined model filtering tests
class CompanySchema(BaseModel):
    id: int | None = None
    name: str
    industry: str | None = None


class CompanyRead(BaseModel):
    id: int
    name: str
    industry: str | None = None


class UserModelSchema(BaseModel):
    id: int | None = None
    name: str
    email: str
    company_id: int | None = None


class UserModelRead(BaseModel):
    id: int
    name: str
    email: str
    company_id: int | None = None


def is_docker_running() -> bool:  # pragma: no cover
    try:
        DockerClient()
        return True
    except Exception:
        return False


@asynccontextmanager
async def _async_session(url: str) -> AsyncGenerator[AsyncSession]:
    async_engine = create_async_engine(url, echo=True, future=True)

    session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore

    async with session() as s:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield s

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await async_engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(request: pytest.FixtureRequest) -> AsyncGenerator[AsyncSession]:
    dialect_marker = request.node.get_closest_marker("dialect")
    dialect = dialect_marker.args[0] if dialect_marker else "sqlite"
    if dialect == "postgresql":
        if not is_docker_running():  # pragma: no cover
            pytest.skip("Docker is required, but not running")
        with PostgresContainer(driver="psycopg") as pg:
            async with _async_session(
                url=pg.get_connection_url(host=pg.get_container_host_ip())
            ) as session:
                yield session
    elif dialect == "sqlite":
        async with _async_session(url="sqlite+aiosqlite:///:memory:") as session:
            yield session
    elif dialect == "mysql":
        if not is_docker_running():  # pragma: no cover
            pytest.skip("Docker is required, but not running")
        with MySqlContainer() as mysql:
            async with _async_session(
                url=make_url(name_or_url=mysql.get_connection_url())._replace(  # type: ignore
                    drivername="mysql+aiomysql"
                )
            ) as session:
                yield session
    else:  # pragma: no cover
        raise NotImplementedError(f"Unsupported dialect: {dialect}")


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
    return request.param  # type: ignore


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
    return MultiPkModel


@pytest.fixture
def multi_pk_test_schema():
    return MultiPkSchema


@pytest.fixture
def multi_pk_test_create_schema():
    return MultiPkCreate


@pytest.fixture
def test_model_custom_columns():
    return ModelWithCustomColumns


@pytest.fixture
def test_model_with_org():
    return ModelWithOrgTest


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
            delete_schema=multi_pk_test_schema,
            read_deps=[test_read_dep],
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
    test_model, create_schema, update_schema, delete_schema, async_session
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
    test_model, create_schema, update_schema, delete_schema, async_session
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
    test_model, create_schema, update_schema, delete_schema, async_session
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
