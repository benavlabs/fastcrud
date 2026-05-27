import pytest
from sqlalchemy import select
from fastcrud.crud.fast_crud import FastCRUD
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_create_successful(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    result = await crud.create(async_session, new_data)

    # v0.20.0 behavior: create() without schema_to_select returns None
    assert result is None

    stmt = select(test_model).where(test_model.name == "New Record")
    db_result = await async_session.execute(stmt)
    fetched_record = db_result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "New Record"
    assert fetched_record.tier_id == 1


@pytest.mark.asyncio
async def test_create_and_read_successful(
    async_session, test_model, create_schema, read_schema
):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    created_record = await crud.create(
        async_session, new_data, schema_to_select=read_schema
    )

    assert created_record is not None
    assert created_record["name"] == "New Record"
    assert created_record["tier_id"] == 1


@pytest.mark.asyncio
async def test_create_and_read_missing_schema(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    with pytest.raises(ValueError):
        await crud.create(async_session, new_data, return_as_model=True)


@pytest.mark.asyncio
async def test_create_and_read_successful_return_as_model(
    async_session, test_model, create_schema, read_schema
):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    created_record = await crud.create(
        async_session,
        new_data,
        schema_to_select=read_schema,
        return_as_model=True,
    )

    assert created_record is not None
    assert created_record.name == "New Record"
    assert created_record.tier_id == 1


@pytest.mark.asyncio
async def test_create_no_commit(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="No Commit Record", tier_id=1)
    result = await crud.create(async_session, new_data, commit=False)

    # v0.20.0 behavior: create() without schema_to_select returns None
    assert result is None

    await async_session.rollback()

    stmt = select(test_model).where(test_model.name == "No Commit Record")
    db_result = await async_session.execute(stmt)
    fetched_record = db_result.scalar_one_or_none()

    assert fetched_record is None


@pytest.mark.asyncio
async def test_create_no_commit_read(
    async_session, test_model, create_schema, read_schema
):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="No Commit Read", tier_id=2)
    created_record = await crud.create(
        async_session, new_data, commit=False, schema_to_select=read_schema
    )

    assert created_record is not None
    assert created_record["name"] == "No Commit Read"
    assert created_record["tier_id"] == 2

    await async_session.rollback()

    stmt = select(test_model).where(test_model.name == "No Commit Read")
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is None


@pytest.mark.asyncio
async def test_create_with_various_valid_data(async_session, test_model, create_schema):
    valid_data_samples = [
        {"name": "Example 1", "tier_id": 1},
        {"name": "Example 2", "tier_id": 2},
    ]

    for data in valid_data_samples:
        crud = FastCRUD(test_model)
        new_data = create_schema(**data)
        result = await crud.create(async_session, new_data)

        # v0.20.0 behavior: create() without schema_to_select returns None
        assert result is None

        stmt = select(test_model).where(test_model.name == data["name"])
        db_result = await async_session.execute(stmt)
        fetched_record = db_result.scalar_one_or_none()

        assert fetched_record is not None
        assert fetched_record.name == data["name"]
        assert fetched_record.tier_id == data["tier_id"]


@pytest.mark.asyncio
async def test_create_with_missing_fields(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    incomplete_data = {"name": "Missing Tier"}
    with pytest.raises(ValidationError):
        await crud.create(async_session, create_schema(**incomplete_data))


@pytest.mark.asyncio
async def test_create_with_extra_fields(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    extra_data = {"name": "Extra", "tier_id": 1, "extra_field": "value"}
    with pytest.raises(ValidationError):
        await crud.create(async_session, create_schema(**extra_data))


@pytest.mark.asyncio
async def test_create_with_invalid_data_types(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    invalid_data = {"name": 123, "tier_id": "invalid"}
    with pytest.raises(ValidationError):
        await crud.create(async_session, create_schema(**invalid_data))


@pytest.mark.asyncio
async def test_create_successful_multi_pk(
    async_session, multi_pk_model, multi_pk_test_create_schema
):
    crud = FastCRUD(multi_pk_model)
    new_data = multi_pk_test_create_schema(name="New Record", id=1, uuid="a")
    result = await crud.create(async_session, new_data)

    # v0.20.0 behavior: create() without schema_to_select returns None
    assert result is None

    stmt = select(multi_pk_model).where(multi_pk_model.name == "New Record")
    db_result = await async_session.execute(stmt)
    fetched_record = db_result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "New Record"
    assert fetched_record.id == 1
    assert fetched_record.uuid == "a"

@pytest.mark.asyncio
async def test_create_returns_inherited_columns(async_session):
    """Test that create() returns columns from parent tables in joined table inheritance."""
    from sqlalchemy import Integer, String, ForeignKey
    from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped

    class InheritBase(DeclarativeBase):
        pass

    class Animal(InheritBase):
        __tablename__ = "animal"
        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        name: Mapped[str] = mapped_column(String(50))
        __mapper_args__ = {"polymorphic_on": "type", "polymorphic_identity": "animal"}
        type: Mapped[str] = mapped_column(String(20))

    class Dog(Animal):
        __tablename__ = "dog"
        id: Mapped[int] = mapped_column(Integer, ForeignKey("animal.id"), primary_key=True)
        breed: Mapped[str] = mapped_column(String(50))
        __mapper_args__ = {"polymorphic_identity": "dog"}

    from pydantic import BaseModel

    class DogCreate(BaseModel):
        name: str
        breed: str
        type: str = "dog"

    class DogRead(BaseModel):
        model_config = {"from_attributes": True}
        id: int
        name: str
        breed: str

    # Create tables
    async with async_session.bind.begin() as conn:
        await conn.run_sync(InheritBase.metadata.create_all)

    crud = FastCRUD(Dog)
    result = await crud.create(async_session, DogCreate(name="Rex", breed="Labrador"), schema_to_select=DogRead)

    # This should include 'name' from parent Animal table
    assert result is not None
    assert result["name"] == "Rex"
    assert result["breed"] == "Labrador"

    # Cleanup
    async with async_session.bind.begin() as conn:
        await conn.run_sync(InheritBase.metadata.drop_all)