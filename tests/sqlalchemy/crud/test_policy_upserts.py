"""``upsert`` and ``upsert_multi`` under a ``PolicyConfig``."""

import pytest
import pytest_asyncio
from pydantic import BaseModel

import fastcrud.crud.fast_crud as fast_crud_module
from fastcrud import AccessDenied, FastCRUD, PolicyConfig
from ..conftest import CategoryModel

pytestmark = pytest.mark.asyncio

ON_EACH_DATABASE = [
    pytest.param("sqlite", marks=pytest.mark.dialect("sqlite"), id="sqlite"),
    pytest.param(
        "postgresql", marks=pytest.mark.dialect("postgresql"), id="postgresql"
    ),
    pytest.param("mysql", marks=pytest.mark.dialect("mysql"), id="mysql"),
]
WITH_RETURNING = [
    pytest.param("sqlite", marks=pytest.mark.dialect("sqlite"), id="sqlite"),
    pytest.param(
        "postgresql", marks=pytest.mark.dialect("postgresql"), id="postgresql"
    ),
]


class ItemUpsert(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: int


class ItemUpsertWithCategoryDefault(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: int | None = 2


class ItemUpsertWithTier(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: int
    tier: dict | None = None


class ItemWithoutKey(BaseModel):
    name: str
    tier_id: int
    category_id: int


class ItemNamed(BaseModel):
    id: int
    name: str


@pytest_asyncio.fixture
async def unrestricted(
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


def _policy(test_model, **overrides):
    return PolicyConfig(
        **{
            "read": test_model.category_id == 1,
            "create": True,
            "update": test_model.tier_id == 1,
            **overrides,
        }
    )


@pytest.fixture
def crud(unrestricted, test_model):
    return unrestricted.with_policy(_policy(test_model))


def _item(id, name="Upserted", tier_id=1, category_id=1):
    return ItemUpsert(id=id, name=name, tier_id=tier_id, category_id=category_id)


async def _names(async_session, unrestricted) -> dict[int, str]:
    rows = await unrestricted.get_multi(async_session, limit=None)
    return {row["id"]: row["name"] for row in rows["data"]}


async def test_upsert_creates_a_new_record(async_session, crud, unrestricted):
    created = await crud.upsert(async_session, _item(20))

    assert created is not None and created["name"] == "Upserted"
    assert (await _names(async_session, unrestricted))[20] == "Upserted"


async def test_upsert_updates_a_record_in_reach(async_session, crud, unrestricted):
    updated = await crud.upsert(async_session, _item(1))

    assert updated is not None and updated["name"] == "Upserted"
    assert (await _names(async_session, unrestricted))[1] == "Upserted"


async def test_upsert_returns_a_row_that_leaves_read(async_session, crud):
    updated = await crud.upsert(async_session, _item(1, category_id=2))

    assert updated is not None and updated["category_id"] == 2
    assert await crud.get(async_session, id=1) is None


@pytest.mark.parametrize("id", [2, 3], ids=["outside-update", "outside-read"])
async def test_upsert_refuses_a_conflicting_row_out_of_reach(
    async_session, crud, unrestricted, id
):
    with pytest.raises(AccessDenied) as denied:
        await crud.upsert(async_session, _item(id))

    assert denied.value.operation == "update"
    assert denied.value.reason == "row"
    assert str(denied.value) == "the existing ModelTest is outside update's reach"
    assert (await _names(async_session, unrestricted))[id] != "Upserted"


@pytest.mark.parametrize(
    ("overrides", "id", "operation"),
    [({"create": False}, 20, "create"), ({"update": False}, 1, "update")],
    ids=["create-denied", "update-denied"],
)
async def test_upsert_needs_the_operation_it_performs(
    async_session, unrestricted, test_model, overrides, id, operation
):
    crud = unrestricted.with_policy(_policy(test_model, **overrides))

    with pytest.raises(AccessDenied) as denied:
        await crud.upsert(async_session, _item(id))

    assert denied.value.operation == operation


async def test_upsert_leaves_hidden_columns_out_of_what_it_returns(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(_policy(test_model, hidden_columns={"category_id"}))

    with pytest.raises(AccessDenied, match="may not write: category_id"):
        await crud.upsert(async_session, _item(20))
    created = await crud.upsert(
        async_session,
        ItemUpsertWithCategoryDefault(id=20, name="Upserted", tier_id=1),
        schema_to_select=ItemUpsertWithCategoryDefault,
    )

    assert created is not None and "category_id" not in created


async def test_upsert_refuses_a_response_model_requiring_a_hidden_column_up_front(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(_policy(test_model, hidden_columns={"name"}))

    with pytest.raises(AccessDenied, match="may not return: name"):
        await crud.upsert(
            async_session,
            ItemUpsertWithCategoryDefault(id=20, name="x", tier_id=1),
            schema_to_select=ItemNamed,
            return_as_model=True,
        )

    assert not await unrestricted.exists(async_session, id=20)


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
async def test_upsert_multi_writes_new_and_conflicting_rows_in_reach(
    async_session, crud, unrestricted, dialect
):
    await crud.upsert_multi(async_session, [_item(1), _item(20)], commit=True)

    names = await _names(async_session, unrestricted)
    assert names[1] == "Upserted" and names[20] == "Upserted"


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
@pytest.mark.parametrize("id", [2, 3], ids=["outside-update", "outside-read"])
async def test_upsert_multi_refuses_a_conflicting_row_out_of_reach(
    async_session, crud, unrestricted, test_model, dialect, id
):
    earlier = await async_session.get(test_model, 9)
    earlier.name = "Changed by the caller"

    with pytest.raises(AccessDenied) as denied:
        await crud.upsert_multi(async_session, [_item(20), _item(1), _item(id)])
    await async_session.commit()

    async_session.expire_all()
    names = await _names(async_session, unrestricted)
    assert denied.value.operation == "update"
    assert denied.value.reason == "row"
    assert 20 not in names
    assert names[1] == "Charlie" and names[id] != "Upserted"
    assert names[9] == "Changed by the caller"


@pytest.mark.parametrize("dialect", WITH_RETURNING)
async def test_upsert_multi_keeps_the_callers_where_skipping_rows_silently(
    async_session, crud, unrestricted, test_model, dialect
):
    result = await crud.upsert_multi(
        async_session,
        [_item(1), _item(5), _item(20)],
        commit=True,
        return_columns=["id", "name"],
        where=test_model.name != "Charlie",
    )

    names = await _names(async_session, unrestricted)
    assert result is not None
    assert sorted(row["id"] for row in result["data"]) == [5, 20]
    assert names[1] == "Charlie" and names[5] == "Upserted"


@pytest.mark.parametrize(
    ("overrides", "items", "operation"),
    [
        ({"create": False}, [_item(1), _item(20)], "create"),
        ({"update": False}, [_item(20), _item(1)], "update"),
    ],
    ids=["create-denied", "update-denied"],
)
async def test_upsert_multi_refuses_a_denied_operation_before_writing(
    async_session, unrestricted, test_model, overrides, items, operation
):
    crud = unrestricted.with_policy(_policy(test_model, **overrides))

    with pytest.raises(AccessDenied) as denied:
        await crud.upsert_multi(async_session, items, commit=True)

    names = await _names(async_session, unrestricted)
    assert denied.value.operation == operation
    assert 20 not in names and names[1] == "Charlie"


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
@pytest.mark.parametrize(
    ("overrides", "items", "operation"),
    [
        ({"create_check": "tier_two"}, [_item(20, tier_id=1)], "create"),
        ({"update_check": "tier_two"}, [_item(1, tier_id=1)], "update"),
    ],
    ids=["create-check", "update-check"],
)
async def test_upsert_multi_checks_each_row_by_how_it_was_written(
    async_session, unrestricted, test_model, dialect, overrides, items, operation
):
    tier_two = test_model.tier_id == 2
    crud = unrestricted.with_policy(
        _policy(
            test_model,
            update=True,
            **{name: tier_two for name in overrides},
        )
    )

    with pytest.raises(AccessDenied) as denied:
        await crud.upsert_multi(async_session, items, commit=True)

    names = await _names(async_session, unrestricted)
    assert denied.value.operation == operation
    assert denied.value.reason == "check"
    assert 20 not in names and names[1] == "Charlie"


async def test_upsert_multi_applies_the_column_rules_to_update_override(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(
        _policy(test_model, read_only_columns={"category_id"})
    )

    with pytest.raises(AccessDenied, match="may not write: category_id"):
        await crud.upsert_multi(
            async_session,
            [ItemUpsertWithCategoryDefault(id=1, name="Upserted", tier_id=1)],
            update_override={"category_id": 2},
        )


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
async def test_upsert_multi_leaves_unset_read_only_columns_alone(
    async_session, unrestricted, test_model, dialect
):
    crud = unrestricted.with_policy(
        _policy(test_model, read_only_columns={"category_id"})
    )

    await crud.upsert_multi(
        async_session,
        [ItemUpsertWithCategoryDefault(id=1, name="Upserted", tier_id=1)],
        commit=True,
    )

    row = await unrestricted.get(async_session, id=1)
    assert row is not None and row["name"] == "Upserted" and row["category_id"] == 1


async def test_upsert_multi_refuses_setting_an_unwritable_column(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(
        _policy(test_model, read_only_columns={"category_id"})
    )

    with pytest.raises(AccessDenied, match="update on ModelTest may not write"):
        await crud.upsert_multi(async_session, [_item(1)])


async def test_upsert_multi_refuses_relationship_data(async_session, crud):
    item = ItemUpsertWithTier(
        id=20, name="New", tier_id=1, category_id=1, tier={"id": 9}
    )

    with pytest.raises(AccessDenied, match="create on ModelTest may not write: tier"):
        await crud.upsert_multi(async_session, [item])


@pytest.mark.parametrize("dialect", WITH_RETURNING)
async def test_upsert_multi_leaves_hidden_columns_out_and_refuses_using_them(
    async_session, unrestricted, test_model, dialect
):
    crud = unrestricted.with_policy(_policy(test_model, hidden_columns={"category_id"}))
    item = ItemUpsertWithCategoryDefault(id=20, name="New", tier_id=1)

    with pytest.raises(AccessDenied, match="may not filter or sort by: category_id"):
        await crud.upsert_multi(
            async_session, [item], where=test_model.category_id == 1
        )
    with pytest.raises(AccessDenied, match="may not return: category_id"):
        await crud.upsert_multi(
            async_session,
            [item],
            schema_to_select=ItemUpsert,
            return_as_model=True,
        )
    result = await crud.upsert_multi(
        async_session, [item], return_columns=["id", "category_id"]
    )

    assert result is not None and result["data"] == [{"id": 20}]


@pytest.mark.dialect("mysql")
async def test_upsert_multi_on_mysql_needs_primary_keys_to_check_created_rows(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(
        _policy(test_model, create_check=test_model.tier_id == 1)
    )

    with pytest.raises(ValueError, match="needs every instance's primary key"):
        await crud.upsert_multi(
            async_session, [ItemWithoutKey(name="New", tier_id=1, category_id=1)]
        )


@pytest.fixture
def row_inserted_after_the_lookup(monkeypatch):
    async def lookup_finds_nothing(db, model, keys):
        return set()

    monkeypatch.setattr(fast_crud_module, "existing_keys", lookup_finds_nothing)


@pytest.mark.parametrize("dialect", WITH_RETURNING)
@pytest.mark.parametrize(
    ("overrides", "id", "reason"),
    [({}, 2, "row"), ({"update": False}, 1, "operation")],
    ids=["outside-reach", "update-denied"],
)
async def test_upsert_multi_refuses_a_row_inserted_after_the_lookup(
    async_session,
    unrestricted,
    test_model,
    row_inserted_after_the_lookup,
    dialect,
    overrides,
    id,
    reason,
):
    crud = unrestricted.with_policy(_policy(test_model, **overrides))

    with pytest.raises(AccessDenied) as denied:
        await crud.upsert_multi(async_session, [_item(20), _item(id)], commit=True)

    names = await _names(async_session, unrestricted)
    assert denied.value.operation == "update"
    assert denied.value.reason == reason
    assert 20 not in names and names[id] != "Upserted"
