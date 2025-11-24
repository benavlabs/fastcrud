from datetime import datetime, timezone
import pytest

from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound, NoResultFound

from fastcrud.crud.fast_crud import FastCRUD
from ...sqlalchemy.conftest import ModelTest, UpdateSchemaTest, ModelTestWithTimestamp


@pytest.mark.asyncio
async def test_update_successful(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    some_existing_id = test_data[0]["id"]
    updated_data = {"name": "Updated Name"}
    await crud.update(db=async_session, object=updated_data, id=some_existing_id)

    updated_record = await async_session.execute(
        select(ModelTest).where(ModelTest.id == some_existing_id)
    )
    assert updated_record.scalar_one().name == "Updated Name"


@pytest.mark.asyncio
async def test_update_various_data(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    some_existing_id = test_data[0]["id"]
    updated_data = {"name": "Different Name"}
    await crud.update(db=async_session, object=updated_data, id=some_existing_id)

    updated_record = await async_session.execute(
        select(ModelTest).where(ModelTest.id == some_existing_id)
    )
    assert updated_record.scalar_one().name == "Different Name"


@pytest.mark.asyncio
async def test_update_non_existent_record(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    non_existent_id = 99999
    updated_data = {"name": "New Name"}

    with pytest.raises(NoResultFound) as exc_info:
        await crud.update(db=async_session, object=updated_data, id=non_existent_id)

    assert "No record found to update" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_invalid_filters(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    updated_data = {"name": "New Name"}

    non_matching_filter = {"name": "NonExistingName"}
    with pytest.raises(NoResultFound) as exc_info:
        await crud.update(db=async_session, object=updated_data, **non_matching_filter)

    assert "No record found to update" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_additional_fields(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    some_existing_id = test_data[0]["id"]
    updated_data = {"name": "Updated Name", "extra_field": "Extra"}

    with pytest.raises(ValueError) as exc_info:
        await crud.update(db=async_session, object=updated_data, id=some_existing_id)

    assert "Extra fields provided" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_with_advanced_filters(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    advanced_filter = {"id__gt": 5}
    updated_data = {"name": "Updated for Advanced Filter"}

    crud = FastCRUD(ModelTest)
    await crud.update(
        db=async_session, object=updated_data, allow_multiple=True, **advanced_filter
    )

    updated_records = await async_session.execute(
        select(ModelTest).where(ModelTest.id > 5)
    )
    assert all(
        record.name == "Updated for Advanced Filter"
        for record in updated_records.scalars()
    )


@pytest.mark.asyncio
async def test_update_multiple_records(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    updated_data = {"name": "Updated Multiple"}
    await crud.update(
        db=async_session, object=updated_data, allow_multiple=True, tier_id=2
    )

    updated_records = await async_session.execute(
        select(ModelTest).where(ModelTest.tier_id == 2)
    )
    assert all(
        record.name == "Updated Multiple" for record in updated_records.scalars()
    )


@pytest.mark.asyncio
async def test_update_multiple_records_restriction(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    updated_data = {"name": "Should Fail"}

    with pytest.raises(MultipleResultsFound) as exc_info:
        await crud.update(db=async_session, object=updated_data, id__lt=10)

    assert "Expected exactly one record to update" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_multiple_records_allow_multiple(
    async_session, test_model, test_data
):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    await crud.update(
        async_session, {"name": "UpdatedName"}, allow_multiple=True, tier_id=1
    )
    updated_count = await crud.count(async_session, name="UpdatedName")
    expected_count = len([item for item in test_data if item["tier_id"] == 1])

    assert updated_count == expected_count


@pytest.mark.asyncio
async def test_update_with_schema_object(async_session, test_data):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    target_id = test_data[0]["id"]
    update_schema = UpdateSchemaTest(name="Updated Via Schema Object")

    await crud.update(db=async_session, object=update_schema, id=target_id)

    updated_record = await async_session.execute(
        select(ModelTest).where(ModelTest.id == target_id)
    )
    updated = updated_record.scalar_one()
    assert (
        updated.name == "Updated Via Schema Object"
    ), "Record should be updated with the name from the schema object."


@pytest.mark.asyncio
async def test_update_auto_updates_updated_at(async_session, test_data):
    initial_time = datetime.now(timezone.utc)
    test_record = ModelTestWithTimestamp(name="InitialName", updated_at=initial_time)
    async_session.add(test_record)
    await async_session.commit()

    crud = FastCRUD(ModelTestWithTimestamp, updated_at_column="updated_at")
    await crud.update(
        db=async_session, object={"name": "UpdatedName"}, id=test_record.id
    )

    updated_record = await async_session.execute(
        select(ModelTestWithTimestamp).where(
            ModelTestWithTimestamp.id == test_record.id
        )
    )
    updated = updated_record.scalar_one()
    assert updated.name == "UpdatedName", "Record should be updated with the new name."
    assert (
        updated.updated_at > initial_time
    ), "updated_at should be later than the initial timestamp."


@pytest.mark.parametrize(
    ["update_kwargs", "expected_result"],
    [
        pytest.param(
            {"return_columns": ["id", "name"]},
            {
                "id": 1,
                "name": "Updated Name",
            },
            id="dict",
        ),
        pytest.param(
            {"schema_to_select": UpdateSchemaTest, "return_as_model": True},
            UpdateSchemaTest(name="Updated Name"),
            id="model",
        ),
        pytest.param(
            {"allow_multiple": True, "return_columns": ["id", "name"]},
            {
                "data": [
                    {
                        "id": 1,
                        "name": "Updated Name",
                    }
                ]
            },
            id="multiple_dict",
        ),
        pytest.param(
            {
                "allow_multiple": True,
                "schema_to_select": UpdateSchemaTest,
                "return_as_model": True,
            },
            {"data": [UpdateSchemaTest(name="Updated Name")]},
            id="multiple_model",
        ),
    ],
)
@pytest.mark.asyncio
async def test_update_with_returning(
    async_session, test_data, update_kwargs, expected_result
):
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    target_id = test_data[0]["id"]
    updated_data = {"name": "Updated Name"}

    updated_record = await crud.update(
        db=async_session,
        object=updated_data,
        id=target_id,
        **update_kwargs,
    )

    assert updated_record == expected_result

    # Rollback the current transaction to see if the record was actually committed
    await async_session.rollback()
    assert await crud.count(async_session, name="Updated Name") == 1
