"""``where=``: SQLAlchemy conditions on every read and write, AND-ed with keyword filters."""

from collections import UserList

import pytest
import pytest_asyncio
from sqlalchemy import select, text, true
from sqlalchemy.exc import NoResultFound

from fastcrud import FastCRUD
from tests.sqlmodel.conftest import CategoryModel

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def crud(
    async_session, test_model, tier_model, test_data, test_data_tier, test_data_category
):
    for tier in test_data_tier:
        async_session.add(tier_model(**tier))
    for category in test_data_category:
        async_session.add(CategoryModel(**category))
    await async_session.flush()
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()
    return FastCRUD(test_model)


def _ids(result) -> list[int]:
    return sorted(row["id"] for row in result["data"])


async def test_an_or_of_and_groups_with_its_count(
    async_session, crud, test_model, tier_model
):
    result = await crud.get_multi(
        async_session,
        where=((test_model.name == "Alice") & (test_model.tier_id == 1))
        | (test_model.name == "Bob"),
    )

    assert _ids(result) == [3, 11]
    assert result["total_count"] == 2


async def test_a_subquery_and_a_column_comparison(
    async_session, crud, test_model, tier_model
):
    premium = select(tier_model.id).where(tier_model.name == "Premium")

    result = await crud.get_multi(
        async_session,
        where=[
            test_model.tier_id.in_(premium),
            test_model.tier_id == test_model.category_id,
        ],
    )

    assert _ids(result) == [1, 5, 9, 11]


async def test_where_is_and_ed_with_keyword_filters(async_session, crud, test_model):
    result = await crud.get_multi(
        async_session, where=test_model.tier_id == 2, category_id=1
    )

    assert _ids(result) == [2, 4, 8]


async def test_get_exists_count_and_select(async_session, crud, test_model):
    alices = test_model.name == "Alice"

    first = await crud.get(async_session, where=alices, tier_id=1)
    statement = await crud.select(where=alices, sort_columns="id")
    selected = (await async_session.execute(statement)).mappings().all()

    assert first is not None and first["id"] == 11
    assert await crud.exists(async_session, where=alices)
    assert not await crud.exists(async_session, where=alices, tier_id=3)
    assert await crud.count(async_session, where=alices) == 2
    assert [row["id"] for row in selected] == [2, 11]


async def test_cursor_pagination(async_session, crud, test_model):
    first = await crud.get_multi_by_cursor(
        async_session, limit=2, where=test_model.tier_id == 1
    )
    second = await crud.get_multi_by_cursor(
        async_session,
        cursor=first["next_cursor"],
        limit=2,
        where=test_model.tier_id == 1,
    )

    assert [row["id"] for row in first["data"]] == [1, 3]
    assert [row["id"] for row in second["data"]] == [5, 7]


async def test_joined_reads_filter_the_base_model(
    async_session, crud, test_model, tier_model
):
    many = await crud.get_multi_joined(
        async_session,
        join_model=tier_model,
        join_prefix="tier_",
        where=test_model.name.in_(["Alice", "Bob"]),
    )
    one = await crud.get_joined(
        async_session,
        join_model=tier_model,
        join_prefix="tier_",
        where=test_model.name == "Bob",
    )

    assert sorted(row["id"] for row in many["data"]) == [2, 3, 11]
    assert many["total_count"] == 3
    assert one is not None and one["tier_name"] == "Premium"


async def test_a_joined_keyword_filter_keeps_where(async_session, crud, test_model):
    result = await crud.get_multi(
        async_session,
        where=test_model.category_id == 2,
        **{"tier.name": "Premium"},
    )

    assert _ids(result) == [3, 7]


async def test_update_reaches_only_the_rows_where_admits(
    async_session, crud, test_model, tier_model
):
    await crud.update(
        async_session,
        {"name": "Renamed"},
        allow_multiple=True,
        where=test_model.tier_id == 2,
        category_id=2,
    )
    with pytest.raises(NoResultFound):
        await crud.update(
            async_session, {"name": "Nope"}, where=test_model.tier_id == 2, id=1
        )

    renamed = await crud.get_multi(async_session, name="Renamed")
    assert _ids(renamed) == [6, 10]


async def test_delete_and_db_delete_take_where_alone(async_session, crud, test_model):
    await crud.delete(async_session, allow_multiple=True, where=test_model.id <= 2)
    await crud.db_delete(async_session, allow_multiple=True, where=test_model.id >= 10)

    deleted = await crud.get_multi(async_session, is_deleted=True)
    remaining = await crud.count(async_session)
    assert _ids(deleted) == [1, 2]
    assert remaining == 9


async def test_a_value_python_evaluated_is_refused(async_session, crud):
    with pytest.raises(TypeError, match="where= takes SQLAlchemy conditions"):
        await crud.get_multi(async_session, where="Alice" == "Alice")


async def test_a_boolean_column_on_its_own(async_session, crud, test_model):
    await crud.delete(async_session, allow_multiple=True, where=test_model.id <= 2)

    deleted = await crud.get_multi(async_session, where=test_model.is_deleted)

    assert _ids(deleted) == [1, 2]
    assert deleted["total_count"] == 2
    assert await crud.exists(async_session, where=[test_model.is_deleted])


@pytest.mark.parametrize("where", [[], ()], ids=["list", "tuple"])
@pytest.mark.parametrize("method", ["delete", "db_delete"])
async def test_an_empty_where_does_not_delete_everything(
    async_session, crud, method, where
):
    with pytest.raises(ValueError, match="No filters provided"):
        await getattr(crud, method)(async_session, allow_multiple=True, where=where)

    assert await crud.count(async_session) == 11


@pytest.mark.parametrize("where", [None, []], ids=["none", "empty"])
async def test_update_without_filters_is_refused(async_session, crud, where):
    with pytest.raises(ValueError, match="No filters provided"):
        await crud.update(
            async_session, {"name": "Everyone"}, allow_multiple=True, where=where
        )

    assert await crud.count(async_session, name="Everyone") == 0


async def test_where_true_reaches_every_record_on_purpose(async_session, crud):
    await crud.update(
        async_session, {"name": "Everyone"}, allow_multiple=True, where=true()
    )

    assert await crud.count(async_session, name="Everyone") == 11


async def test_a_text_clause(async_session, crud):
    result = await crud.get_multi(
        async_session, where=text("tier_id = :tier").bindparams(tier=2), category_id=2
    )

    assert _ids(result) == [6, 10]


async def test_any_sequence_of_conditions(async_session, crud, test_model):
    result = await crud.get_multi(
        async_session,
        where=UserList([test_model.tier_id == 2, test_model.category_id == 2]),
    )

    assert _ids(result) == [6, 10]
