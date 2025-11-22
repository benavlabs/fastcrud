import pytest
from typing import Optional, List

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import Column, Integer, String, ForeignKey, select
from sqlalchemy.orm import relationship
from sqlalchemy.exc import IntegrityError

from fastcrud.crud.fast_crud import FastCRUD
from ...sqlalchemy.conftest import Base


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


class NestedChildCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    token: str


class NestedParentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    child: Optional[NestedChildCreate] = None


class NestedParentRead(BaseModel):
    id: int
    name: str


class NestedChildRead(BaseModel):
    provider: str
    token: str


class NestedParentReadWithChild(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    child: Optional[NestedChildRead] = None


class OneToManyParent(Base):
    __tablename__ = "one_to_many_parent"

    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False)
    children = relationship(
        "OneToManyChild", back_populates="parent", cascade="all, delete-orphan"
    )


class OneToManyChild(Base):
    __tablename__ = "one_to_many_child"

    id = Column(Integer, primary_key=True)
    title = Column(String(64), nullable=False)
    parent_id = Column(Integer, ForeignKey("one_to_many_parent.id"), nullable=False)
    parent = relationship("OneToManyParent", back_populates="children")


class OneToManyChildCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str


class OneToManyParentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    children: List[OneToManyChildCreate] = []


class ConstrainedParent(Base):
    __tablename__ = "constrained_parent"

    id = Column(Integer, primary_key=True)
    name = Column(String(32), nullable=False, unique=True)
    children = relationship(
        "ConstrainedChild", back_populates="parent", cascade="all, delete-orphan"
    )


class ConstrainedChild(Base):
    __tablename__ = "constrained_child"

    id = Column(Integer, primary_key=True)
    code = Column(String(32), nullable=False)
    parent_id = Column(Integer, ForeignKey("constrained_parent.id"), nullable=False)
    parent = relationship("ConstrainedParent", back_populates="children")


class ConstrainedChildCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Optional at the schema level, but NOT NULL at the database level
    code: Optional[str] = None


class ConstrainedParentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    children: List[ConstrainedChildCreate]


@pytest.mark.asyncio
async def test_nested_create_parent_and_child(async_session):
    crud = FastCRUD(NestedParent)

    payload = NestedParentCreate(
        name="parent-1",
        child=NestedChildCreate(provider="google", token="secret-token"),
    )

    # Use schema_to_select to get the created instance back
    created = await crud.create(
        async_session,
        payload,
        schema_to_select=NestedParentRead,
        return_as_model=True
    )

    # The parent should be persisted with an id and name
    assert created.id is not None
    assert created.name == "parent-1"

    # Verify the child was created and linked via an explicit async query
    stmt = select(NestedChild).where(NestedChild.parent_id == created.id)
    result = await async_session.execute(stmt)
    child_row = result.scalar_one_or_none()

    assert child_row is not None
    assert child_row.provider == "google"
    assert child_row.token == "secret-token"
    assert child_row.parent_id == created.id


@pytest.mark.asyncio
async def test_nested_create_parent_only(async_session):
    crud = FastCRUD(NestedParent)

    payload = NestedParentCreate(name="parent-no-child")
    created = await crud.create(
        async_session,
        payload,
        schema_to_select=NestedParentRead,
        return_as_model=True
    )

    assert created.id is not None
    assert created.name == "parent-no-child"

    # There should be no child rows linked to this parent in the database
    stmt = select(NestedChild).where(NestedChild.parent_id == created.id)
    result = await async_session.execute(stmt)
    child_row = result.scalar_one_or_none()

    assert child_row is None


@pytest.mark.asyncio
async def test_nested_create_no_commit(async_session):
    crud = FastCRUD(NestedParent)

    payload = NestedParentCreate(
        name="parent-tx",
        child=NestedChildCreate(provider="github", token="tx-token"),
    )
    created = await crud.create(
        async_session,
        payload,
        commit=False,
        schema_to_select=NestedParentRead,
        return_as_model=True
    )

    assert created.id is not None

    # Within the same transaction (before rollback), the parent and child
    # should be visible to queries
    stmt_parent = select(NestedParent).where(NestedParent.id == created.id)
    result_parent = await async_session.execute(stmt_parent)
    parent_in_tx = result_parent.scalar_one_or_none()
    assert parent_in_tx is not None

    stmt_child = select(NestedChild).where(NestedChild.parent_id == created.id)
    result_child = await async_session.execute(stmt_child)
    child_in_tx = result_child.scalar_one_or_none()
    assert child_in_tx is not None

    await async_session.rollback()

    # After rollback, neither parent nor child should be persisted
    stmt = select(NestedParent).where(NestedParent.name == "parent-tx")
    result = await async_session.execute(stmt)
    parent_row = result.scalar_one_or_none()

    assert parent_row is None


@pytest.mark.asyncio
async def test_nested_create_invalid_child_raises(async_session):
    _ = FastCRUD(NestedParent)

    # Missing required field "token" in NestedChildCreate should raise ValidationError
    with pytest.raises(ValidationError):
        NestedParentCreate(
            name="parent-invalid",
            child=NestedChildCreate(provider="x", token=None),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_nested_create_with_schema_to_select_return_as_model(async_session):
    crud = FastCRUD(NestedParent)

    payload = NestedParentCreate(
        name="parent-select-model",
        child=NestedChildCreate(provider="google", token="secret-token"),
    )

    created = await crud.create(
        async_session,
        payload,
        schema_to_select=NestedParentReadWithChild,
        return_as_model=True,
    )

    assert isinstance(created, NestedParentReadWithChild)
    assert created.id is not None
    assert created.name == "parent-select-model"
    # Relationship field is defined on the schema but is not populated without joins.
    assert created.child is None

    # The child row should still be created and linked to the parent.
    stmt = select(NestedChild).where(NestedChild.parent_id == created.id)
    result = await async_session.execute(stmt)
    child_row = result.scalar_one_or_none()

    assert child_row is not None
    assert child_row.provider == "google"
    assert child_row.token == "secret-token"
    assert child_row.parent_id == created.id


@pytest.mark.asyncio
async def test_nested_create_with_schema_to_select_return_as_dict(async_session):
    crud = FastCRUD(NestedParent)

    payload = NestedParentCreate(
        name="parent-select-dict",
        child=NestedChildCreate(provider="google", token="secret-token"),
    )

    created = await crud.create(
        async_session,
        payload,
        schema_to_select=NestedParentReadWithChild,
        return_as_model=False,
    )

    assert isinstance(created, dict)
    assert created["id"] is not None
    assert created["name"] == "parent-select-dict"
    # Relationship fields in the select schema are excluded from the SELECT columns.
    assert "child" not in created

    stmt = select(NestedChild).where(NestedChild.parent_id == created["id"])
    result = await async_session.execute(stmt)
    child_row = result.scalar_one_or_none()

    assert child_row is not None
    assert child_row.provider == "google"
    assert child_row.token == "secret-token"
    assert child_row.parent_id == created["id"]


@pytest.mark.asyncio
async def test_nested_create_one_to_many_children(async_session):
    crud = FastCRUD(OneToManyParent)

    payload = OneToManyParentCreate(
        name="parent-with-children",
        children=[
            OneToManyChildCreate(title="Book 1"),
            OneToManyChildCreate(title="Book 2"),
        ],
    )

    # Create without schema_to_select returns None in v0.20.0
    result = await crud.create(async_session, payload)
    assert result is None  # This is the expected v0.20.0 behavior

    # Query to verify the parent was created
    stmt_parent = select(OneToManyParent).where(OneToManyParent.name == "parent-with-children")
    parent_result = await async_session.execute(stmt_parent)
    created = parent_result.scalar_one_or_none()

    assert created is not None
    assert created.id is not None
    assert created.name == "parent-with-children"

    # Verify all child rows were created and linked
    stmt_children = select(OneToManyChild).where(OneToManyChild.parent_id == created.id)
    children = (await async_session.execute(stmt_children)).scalars().all()

    assert len(children) == 2
    titles = {c.title for c in children}
    assert titles == {"Book 1", "Book 2"}


@pytest.mark.asyncio
async def test_nested_create_child_db_constraint_violation_rollback(async_session):
    crud = FastCRUD(ConstrainedParent)

    # Pydantic validation passes (code is Optional[str]),
    # but the database enforces NOT NULL on ConstrainedChild.code.
    payload = ConstrainedParentCreate(
        name="parent-db-error",
        children=[ConstrainedChildCreate(code=None)],
    )

    with pytest.raises(IntegrityError):
        await crud.create(async_session, payload)

    # The transaction should be rolled back so that no partial rows remain.
    await async_session.rollback()

    stmt_parent = select(ConstrainedParent).where(
        ConstrainedParent.name == "parent-db-error"
    )
    parent_row = (await async_session.execute(stmt_parent)).scalar_one_or_none()
    assert parent_row is None

    stmt_child = select(ConstrainedChild)
    children = (await async_session.execute(stmt_child)).scalars().all()
    assert children == []

