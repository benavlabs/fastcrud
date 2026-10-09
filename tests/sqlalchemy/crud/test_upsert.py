import pytest

from fastcrud.crud.fast_crud import FastCRUD
from tests.sqlalchemy.conftest import CategoryModel, ReadSchemaTest, TierModel


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


@pytest.mark.parametrize(
    ["insert", "update"],
    [
        pytest.param(
            {
                "kwargs": {},
                "expected_result": None,
            },
            {
                "kwargs": {},
                "expected_result": None,
            },
            marks=pytest.mark.dialect("postgresql"),
            id="postgresql-none",
        ),
        pytest.param(
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New Record",
                        }
                    ]
                },
            },
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New name",
                        }
                    ]
                },
            },
            marks=pytest.mark.dialect("postgresql"),
            id="postgresql-dict",
        ),
        pytest.param(
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New Record",
                        }
                    ]
                },
            },
            {
                "kwargs": {
                    "return_columns": ["id", "name"],
                    "update_override": {"name": "New"},
                },
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New",
                        }
                    ]
                },
            },
            marks=pytest.mark.dialect("postgresql"),
            id="postgresql-dict-update-override",
        ),
        pytest.param(
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New Record",
                        }
                    ]
                },
            },
            {
                "kwargs": {
                    "return_columns": ["id", "name"],
                    "name__match": "NewRecord",
                },
                "expected_result": {"data": []},
            },
            marks=pytest.mark.dialect("postgresql"),
            id="postgresql-dict-filtered",
        ),
        pytest.param(
            {
                "kwargs": {
                    "schema_to_select": ReadSchemaTest,
                    "return_as_model": True,
                },
                "expected_result": {
                    "data": [
                        ReadSchemaTest(
                            id=1, name="New Record", tier_id=1, category_id=1
                        )
                    ]
                },
            },
            {
                "kwargs": {
                    "schema_to_select": ReadSchemaTest,
                    "return_as_model": True,
                },
                "expected_result": {
                    "data": [
                        ReadSchemaTest(id=1, name="New name", tier_id=1, category_id=1)
                    ]
                },
            },
            marks=pytest.mark.dialect("postgresql"),
            id="postgresql-model",
        ),
        pytest.param(
            {
                "kwargs": {},
                "expected_result": None,
            },
            {
                "kwargs": {},
                "expected_result": None,
            },
            marks=pytest.mark.dialect("sqlite"),
            id="sqlite-none",
        ),
        pytest.param(
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New Record",
                        }
                    ]
                },
            },
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New name",
                        }
                    ]
                },
            },
            marks=pytest.mark.dialect("sqlite"),
            id="sqlite-dict",
        ),
        pytest.param(
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New Record",
                        }
                    ]
                },
            },
            {
                "kwargs": {
                    "return_columns": ["id", "name"],
                    "update_override": {"name": "New"},
                },
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New",
                        }
                    ]
                },
            },
            marks=pytest.mark.dialect("sqlite"),
            id="sqlite-dict-update-override",
        ),
        pytest.param(
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_result": {
                    "data": [
                        {
                            "id": 1,
                            "name": "New Record",
                        }
                    ]
                },
            },
            {
                "kwargs": {
                    "return_columns": ["id", "name"],
                    "name__like": "NewRecord",
                },
                "expected_result": {"data": []},
            },
            marks=pytest.mark.dialect("sqlite"),
            id="sqlite-dict-filtered",
        ),
        pytest.param(
            {
                "kwargs": {
                    "schema_to_select": ReadSchemaTest,
                    "return_as_model": True,
                },
                "expected_result": {
                    "data": [
                        ReadSchemaTest(
                            id=1, name="New Record", tier_id=1, category_id=1
                        )
                    ]
                },
            },
            {
                "kwargs": {
                    "schema_to_select": ReadSchemaTest,
                    "return_as_model": True,
                },
                "expected_result": {
                    "data": [
                        ReadSchemaTest(id=1, name="New name", tier_id=1, category_id=1)
                    ]
                },
            },
            marks=pytest.mark.dialect("sqlite"),
            id="sqlite-model",
        ),
        pytest.param(
            {
                "kwargs": {},
                "expected_result": None,
            },
            {
                "kwargs": {},
                "expected_result": None,
            },
            marks=pytest.mark.dialect("mysql"),
            id="mysql-none",
        ),
        pytest.param(
            {
                "kwargs": {},
                "expected_result": None,
            },
            {
                "kwargs": {"update_override": {"name": "New"}},
                "expected_result": None,
            },
            marks=pytest.mark.dialect("mysql"),
            id="mysql-dict-update-override",
        ),
    ],
)
@pytest.mark.asyncio
async def test_upsert_multi_successful(
    async_session,
    test_model,
    read_schema,
    test_data_tier,
    test_data_category,
    insert,
    update,
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    new_data = read_schema(id=1, name="New Record", tier_id=1, category_id=1)
    fetched_records = await crud.upsert_multi(
        async_session, [new_data], commit=True, **insert["kwargs"]
    )
    assert not async_session.in_transaction()
    assert fetched_records == insert["expected_result"]

    updated_new_data = new_data.model_copy(update={"name": "New name"})
    updated_fetched_records = await crud.upsert_multi(
        async_session, [updated_new_data], commit=True, **update["kwargs"]
    )
    assert not async_session.in_transaction()
    assert updated_fetched_records == update["expected_result"]


@pytest.mark.parametrize(
    ["insert"],
    [
        pytest.param(
            {
                "kwargs": {"return_columns": ["id", "name"]},
                "expected_exception": {
                    "type": ValueError,
                    "match": r"MySQL does not support the returning clause for insert operations.",
                },
            },
            marks=pytest.mark.dialect("mysql"),
            id="mysql-dict",
        ),
        pytest.param(
            {
                "kwargs": {
                    "name__like": "NewRecord",
                },
                "expected_exception": {
                    "type": ValueError,
                    "match": r"MySQL does not support filtering on insert operations.",
                },
            },
            marks=pytest.mark.dialect("mysql"),
            id="mysql-dict-filtered",
        ),
        pytest.param(
            {
                "kwargs": {
                    "schema_to_select": ReadSchemaTest,
                    "return_as_model": True,
                },
                "expected_exception": {
                    "type": ValueError,
                    "match": r"MySQL does not support the returning clause for insert operations.",
                },
            },
            marks=pytest.mark.dialect("mysql"),
            id="mysql-model",
        ),
    ],
)
@pytest.mark.asyncio
async def test_upsert_multi_unsupported(
    async_session,
    test_model,
    read_schema,
    test_data_tier,
    test_data_category,
    insert,
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    new_data = read_schema(id=1, name="New Record", tier_id=1, category_id=1)
    with pytest.raises(
        insert["expected_exception"]["type"],
        match=insert["expected_exception"]["match"],
    ):
        await crud.upsert_multi(async_session, [new_data], **insert["kwargs"])
