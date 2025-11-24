import pytest
from pydantic import BaseModel

from fastcrud.crud.fast_crud import FastCRUD
from ...sqlalchemy.conftest import ModelTest, CreateSchemaTest
from sqlalchemy.exc import MultipleResultsFound


@pytest.mark.asyncio
async def test_get_existing_record(async_session, test_data):
    test_record = ModelTest(**test_data[0])
    async_session.add(test_record)
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    fetched_record = await crud.get(async_session, **test_data[0])

    assert fetched_record is not None
    assert fetched_record["name"] == test_data[0]["name"]


@pytest.mark.asyncio
async def test_get_with_filters(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    for item in test_data:
        fetched_record = await crud.get(async_session, **item)
        assert fetched_record is not None
        assert fetched_record["name"] == item["name"]


@pytest.mark.asyncio
async def test_get_non_existent_record(async_session):
    crud = FastCRUD(ModelTest)
    non_existent_filter = {"name": "NonExistentName"}
    fetched_record = await crud.get(async_session, **non_existent_filter)

    assert fetched_record is None


@pytest.mark.asyncio
async def test_get_selecting_columns(async_session, test_data):
    test_record = ModelTest(**test_data[0])
    async_session.add(test_record)
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    fetched_record = await crud.get(
        async_session, schema_to_select=CreateSchemaTest, **test_data[0]
    )

    assert fetched_record is not None
    assert "name" in fetched_record


@pytest.mark.asyncio
async def test_get_with_advanced_filters(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    advanced_filter = {"id__gt": 1}
    fetched_record_gt = await crud.get(async_session, **advanced_filter)

    assert fetched_record_gt is not None
    assert fetched_record_gt["id"] > 1, "Should fetch a record with ID greater than 1"

    ne_filter = {"name__ne": test_data[0]["name"]}
    fetched_record_ne = await crud.get(async_session, **ne_filter)

    assert fetched_record_ne is not None
    assert (
        fetched_record_ne["name"] != test_data[0]["name"]
    ), "Should fetch a record with a different name"


@pytest.mark.asyncio
async def test_get_with_schema_selecting_specific_columns(
    async_session, test_data
) -> None:
    async_session.add(ModelTest(**test_data[0]))
    await async_session.commit()

    class PartialSchema(BaseModel):
        name: str

    crud: FastCRUD = FastCRUD(ModelTest)
    fetched_record = await crud.get(
        async_session, schema_to_select=PartialSchema, id=test_data[0]["id"]
    )

    assert fetched_record is not None
    assert (
        "name" in fetched_record and "tier_id" not in fetched_record
    ), "Should only fetch the 'name' column based on the PartialSchema"


@pytest.mark.asyncio
async def test_get_return_as_model_instance(async_session, test_data, read_schema):
    async_session.add(ModelTest(**test_data[0]))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    fetched_record = await crud.get(
        async_session,
        return_as_model=True,
        schema_to_select=read_schema,
        id=test_data[0]["id"],
    )

    assert isinstance(
        fetched_record, read_schema
    ), "The fetched record should be an instance of the ReadSchemaTest Pydantic model"


@pytest.mark.asyncio
async def test_get_return_as_model_without_schema(async_session, test_data):
    async_session.add(ModelTest(**test_data[0]))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    with pytest.raises(ValueError) as exc_info:
        await crud.get(
            async_session,
            return_as_model=True,
            id=test_data[0]["id"],
        )

    assert (
        str(exc_info.value)
        == "schema_to_select must be provided when return_as_model is True."
    )


@pytest.mark.asyncio
async def test_get_strict_existing_record(async_session, test_data):
    test_record = ModelTest(**test_data[0])
    async_session.add(test_record)
    async_session.add(ModelTest(**test_data[1]))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    with pytest.raises(MultipleResultsFound):
        fetched_record = await crud.get(async_session, one_or_none=True, category_id=1)

    fetched_record = await crud.get(async_session, **test_data[0])

    assert fetched_record is not None
    assert fetched_record["name"] == test_data[0]["name"]
