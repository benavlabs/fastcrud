import pytest

from fastcrud.crud.fast_crud import FastCRUD


@pytest.mark.asyncio
async def test_upsert_successful(async_session, test_model, read_schema):
    crud = FastCRUD(test_model)
    new_data = read_schema(id=1, name="New Record", tier_id=1, category_id=1)
    fetched_record = await crud.upsert(async_session, new_data, return_as_model=True)
    assert read_schema.model_validate(fetched_record) == new_data

    fetched_record.name == "New name"

    updated_fetched_record = await crud.upsert(async_session, fetched_record)
    assert read_schema.model_validate(updated_fetched_record) == fetched_record


@pytest.mark.asyncio
async def test_upsert_updates_only_its_own_record(
    async_session, test_model, read_schema
):
    crud = FastCRUD(test_model)
    await crud.upsert(
        async_session, read_schema(id=1, name="One", tier_id=1, category_id=None)
    )
    await crud.upsert(
        async_session, read_schema(id=2, name="Two", tier_id=1, category_id=None)
    )

    await crud.upsert(
        async_session, read_schema(id=1, name="Renamed", tier_id=1, category_id=None)
    )

    assert (await crud.get(async_session, id=1))["name"] == "Renamed"
    assert (await crud.get(async_session, id=2))["name"] == "Two"
