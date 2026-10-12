"""``PolicyConfig``: which operations a caller may perform, and which rows each one reaches."""

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import or_, true
from sqlalchemy.exc import MultipleResultsFound, NoResultFound

from fastcrud import AccessDenied, FastCRUD, JoinConfig, PolicyConfig
from ..conftest import CategoryModel, MultiPkModel

pytestmark = pytest.mark.asyncio

ON_EACH_DATABASE = [
    pytest.param("sqlite", marks=pytest.mark.dialect("sqlite"), id="sqlite"),
    pytest.param(
        "postgresql", marks=pytest.mark.dialect("postgresql"), id="postgresql"
    ),
    pytest.param("mysql", marks=pytest.mark.dialect("mysql"), id="mysql"),
]
READABLE = [1, 2, 4, 5, 8, 9, 11]
UPDATABLE = [1, 5, 9, 11]
DELETABLE = [2, 4, 8]


class ItemCreate(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: int


class ItemRead(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: int


class ItemCreateWithTier(BaseModel):
    id: int
    name: str
    tier_id: int
    tier: dict | None = None


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


@pytest.fixture
def policy(test_model):
    return PolicyConfig(
        read=test_model.category_id == 1,
        create=True,
        update=test_model.tier_id == 1,
        delete=test_model.tier_id == 2,
    )


@pytest.fixture
def crud(unrestricted, policy):
    return unrestricted.with_policy(policy)


def _ids(rows) -> list[int]:
    return sorted(row["id"] for row in rows)


async def _names(async_session, unrestricted) -> dict[int, str]:
    rows = await unrestricted.get_multi(async_session, limit=None)
    return {row["id"]: row["name"] for row in rows["data"]}


def _tier_join(test_model, tier_model):
    return JoinConfig(
        model=tier_model,
        join_on=test_model.tier_id == tier_model.id,
        join_prefix="tier_",
    )


DENIED_OPERATIONS = [
    pytest.param("read", lambda crud, db, m, t: crud.get(db, id=1), id="get"),
    pytest.param("read", lambda crud, db, m, t: crud.get_multi(db), id="get_multi"),
    pytest.param("read", lambda crud, db, m, t: crud.count(db), id="count"),
    pytest.param("read", lambda crud, db, m, t: crud.exists(db, id=1), id="exists"),
    pytest.param("read", lambda crud, db, m, t: crud.select(), id="select"),
    pytest.param(
        "read", lambda crud, db, m, t: crud.get_multi_by_cursor(db), id="cursor"
    ),
    pytest.param(
        "read",
        lambda crud, db, m, t: crud.get_joined(
            db, joins_config=[_tier_join(m, t)], id=1
        ),
        id="get_joined",
    ),
    pytest.param(
        "read",
        lambda crud, db, m, t: crud.get_multi_joined(
            db, joins_config=[_tier_join(m, t)]
        ),
        id="get_multi_joined",
    ),
    pytest.param(
        "create",
        lambda crud, db, m, t: crud.create(
            db, ItemCreate(id=20, name="New", tier_id=1, category_id=1)
        ),
        id="create",
    ),
    pytest.param(
        "update",
        lambda crud, db, m, t: crud.update(db, {"name": "Renamed"}, id=1),
        id="update",
    ),
    pytest.param("delete", lambda crud, db, m, t: crud.delete(db, id=1), id="delete"),
    pytest.param(
        "delete", lambda crud, db, m, t: crud.db_delete(db, id=1), id="db_delete"
    ),
]


@pytest.mark.parametrize(("operation", "call"), DENIED_OPERATIONS)
async def test_an_empty_policy_denies_every_operation(
    async_session, unrestricted, test_model, tier_model, operation, call
):
    crud = unrestricted.with_policy(PolicyConfig())

    with pytest.raises(AccessDenied) as denied:
        await call(crud, async_session, test_model, tier_model)

    assert denied.value.operation == operation
    assert denied.value.reason == "operation"
    assert denied.value.model is test_model
    assert str(denied.value) == f"{operation} on ModelTest is not allowed by the policy"


@pytest.mark.parametrize("rule", ["read", "update", "delete"])
@pytest.mark.parametrize("value", ["everyone", None, []], ids=["str", "none", "empty"])
async def test_a_rule_takes_true_false_or_conditions(rule, value):
    with pytest.raises(TypeError, match=f"{rule} takes True, False or SQLAlchemy"):
        PolicyConfig(**{rule: value})


async def test_create_takes_only_true_or_false(test_model):
    with pytest.raises(TypeError, match="put the conditions a created row must meet"):
        PolicyConfig(create=test_model.tier_id == 1)


async def test_a_check_takes_only_sqlalchemy_conditions():
    with pytest.raises(TypeError, match="where= takes SQLAlchemy conditions"):
        PolicyConfig(update_check="owner")


async def test_access_denied_tells_its_reasons_apart(test_model):
    check = AccessDenied("update", test_model, reason="check")
    columns = AccessDenied("create", test_model, columns=["tier"], reason="columns")

    assert str(check) == "the updated ModelTest doesn't pass update_check"
    assert str(columns) == "create on ModelTest may not write: tier"
    assert columns.columns == ("tier",)


async def test_with_policy_returns_a_copy_and_leaves_the_original(
    async_session, unrestricted, policy
):
    custom = FastCRUD(unrestricted.model, custom_filters={"double": lambda c: c})

    restricted = custom.with_policy(policy)

    assert restricted is not custom
    assert custom.policy is None
    assert restricted.policy is policy
    assert restricted.custom_filters is custom.custom_filters
    assert await custom.count(async_session) == 11
    assert await restricted.count(async_session) == len(READABLE)


async def test_reads_reach_only_rows_matching_read(
    async_session, crud, test_model, tier_model
):
    result = await crud.get_multi(async_session)
    joined = await crud.get_multi_joined(
        async_session, joins_config=[_tier_join(test_model, tier_model)]
    )
    cursor = await crud.get_multi_by_cursor(async_session, limit=20)
    selected = await async_session.execute(await crud.select())

    assert _ids(result["data"]) == READABLE
    assert result["total_count"] == len(READABLE)
    assert _ids(joined["data"]) == READABLE
    assert joined["total_count"] == len(READABLE)
    assert _ids(cursor["data"]) == READABLE
    assert sorted(row.id for row in selected) == READABLE
    assert await crud.count(async_session) == len(READABLE)


async def test_a_row_outside_read_behaves_as_absent(
    async_session, crud, test_model, tier_model
):
    joins_config = [_tier_join(test_model, tier_model)]

    assert await crud.get(async_session, id=3) is None
    assert await crud.get_joined(async_session, joins_config=joins_config, id=3) is None
    assert not await crud.exists(async_session, id=3)
    assert await crud.get(async_session, id=1) is not None
    assert await crud.exists(async_session, id=1)


async def test_the_callers_where_cannot_widen_the_policy(
    async_session, crud, test_model
):
    result = await crud.get_multi(
        async_session, where=or_(test_model.id == 3, test_model.id == 1)
    )

    assert _ids(result["data"]) == [1]
    assert await crud.count(async_session, where=true()) == len(READABLE)


async def test_read_true_reaches_every_row(async_session, unrestricted):
    crud = unrestricted.with_policy(PolicyConfig(read=True))

    assert await crud.count(async_session) == 11


async def test_update_reaches_rows_matching_read_and_update(
    async_session, crud, unrestricted
):
    await crud.update(async_session, {"name": "Renamed"}, id=1)

    with pytest.raises(NoResultFound):
        await crud.update(async_session, {"name": "Renamed"}, id=3)
    with pytest.raises(NoResultFound):
        await crud.update(async_session, {"name": "Renamed"}, id=2)

    names = await _names(async_session, unrestricted)
    assert names[1] == "Renamed"
    assert names[2] == "Alice" and names[3] == "Bob"


async def test_update_counts_only_rows_in_reach(async_session, crud, unrestricted):
    with pytest.raises(MultipleResultsFound, match="found 4"):
        await crud.update(async_session, {"name": "Renamed"}, where=true())

    await crud.update(
        async_session, {"name": "Renamed"}, allow_multiple=True, where=true()
    )

    names = await _names(async_session, unrestricted)
    assert sorted(id for id, name in names.items() if name == "Renamed") == UPDATABLE


async def test_update_without_filters_is_still_refused(async_session, crud):
    with pytest.raises(ValueError, match="No filters provided"):
        await crud.update(async_session, {"name": "Renamed"}, allow_multiple=True)


async def test_update_cannot_reach_a_row_outside_read(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(
        PolicyConfig(read=test_model.category_id == 1, update=True)
    )

    with pytest.raises(NoResultFound):
        await crud.update(async_session, {"name": "Renamed"}, id=3)


async def test_update_check_defaults_to_the_update_condition(
    async_session, crud, unrestricted
):
    with pytest.raises(AccessDenied) as denied:
        await crud.update(async_session, {"tier_id": 2}, id=1)

    assert denied.value.reason == "check"
    assert str(denied.value) == "the updated ModelTest doesn't pass update_check"
    row = await unrestricted.get(async_session, id=1)
    assert row is not None and row["tier_id"] == 1


async def test_an_explicit_update_check(async_session, unrestricted, test_model):
    crud = unrestricted.with_policy(
        PolicyConfig(read=True, update=True, update_check=test_model.name != "Taken")
    )

    await crud.update(async_session, {"name": "Fine"}, id=1)
    with pytest.raises(AccessDenied):
        await crud.update(async_session, {"name": "Taken"}, id=2)

    names = await _names(async_session, unrestricted)
    assert names[1] == "Fine" and names[2] == "Alice"


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
async def test_a_failed_update_check_keeps_the_callers_earlier_work(
    async_session, crud, unrestricted, test_model, dialect
):
    earlier = await async_session.get(test_model, 9)
    earlier.name = "Changed by the caller"

    with pytest.raises(AccessDenied):
        await crud.update(async_session, {"tier_id": 2}, id=1, commit=False)
    await async_session.commit()

    async_session.expire_all()
    names = await _names(async_session, unrestricted)
    row = await unrestricted.get(async_session, id=1)
    assert names[9] == "Changed by the caller"
    assert row is not None and row["tier_id"] == 1


async def test_an_updated_row_that_leaves_read_is_returned(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(
        PolicyConfig(read=test_model.category_id == 1, update=True)
    )

    updated = await crud.update(
        async_session,
        {"category_id": 2},
        id=1,
        return_columns=["id", "category_id"],
    )

    assert updated == {"id": 1, "category_id": 2}
    assert await crud.get(async_session, id=1) is None


async def test_update_refuses_relationship_data(async_session, crud):
    with pytest.raises(AccessDenied) as denied:
        await crud.update(async_session, {"tier": {"id": 1}}, id=1)

    assert denied.value.reason == "columns"
    assert denied.value.columns == ("tier",)


async def test_delete_reaches_rows_matching_read_and_delete(
    async_session, crud, unrestricted
):
    await crud.delete(async_session, id=2)

    with pytest.raises(NoResultFound):
        await crud.delete(async_session, id=1)
    with pytest.raises(NoResultFound):
        await crud.delete(async_session, id=6)

    deleted = await unrestricted.get_multi(async_session, is_deleted=True)
    assert _ids(deleted["data"]) == [2]


async def test_a_soft_delete_is_governed_by_delete_not_update(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(PolicyConfig(read=True, delete=True))

    await crud.delete(async_session, id=1)

    deleted = await unrestricted.get_multi(async_session, is_deleted=True)
    assert _ids(deleted["data"]) == [1]


async def test_delete_of_a_loaded_row_checks_it_against_delete(
    async_session, crud, unrestricted, test_model
):
    out_of_reach = await async_session.get(test_model, 1)
    in_reach = await async_session.get(test_model, 2)

    with pytest.raises(NoResultFound):
        await crud.delete(async_session, db_row=out_of_reach)
    await crud.delete(async_session, db_row=in_reach)

    deleted = await unrestricted.get_multi(async_session, is_deleted=True)
    assert _ids(deleted["data"]) == [2]


async def test_db_delete_reaches_only_rows_in_reach(async_session, crud, unrestricted):
    with pytest.raises(NoResultFound):
        await crud.db_delete(async_session, id=1)
    with pytest.raises(MultipleResultsFound, match="found 3"):
        await crud.db_delete(async_session, where=true())
    await crud.db_delete(async_session, id=2)

    remaining = await unrestricted.get_multi(async_session, limit=None)
    assert 1 in _ids(remaining["data"])
    assert 2 not in _ids(remaining["data"])


async def test_create_check_lets_a_matching_row_in(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(
        PolicyConfig(read=True, create=True, create_check=test_model.tier_id == 1)
    )

    await crud.create(
        async_session, ItemCreate(id=20, name="New", tier_id=1, category_id=1)
    )

    assert await unrestricted.exists(async_session, id=20)


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
@pytest.mark.parametrize("commit", [True, False], ids=["commit", "no-commit"])
async def test_a_failed_create_check_keeps_the_callers_earlier_work(
    async_session, unrestricted, test_model, dialect, commit
):
    crud = unrestricted.with_policy(
        PolicyConfig(read=True, create=True, create_check=test_model.tier_id == 1)
    )
    earlier = await async_session.get(test_model, 9)
    earlier.name = "Changed by the caller"

    with pytest.raises(AccessDenied) as denied:
        await crud.create(
            async_session,
            ItemCreate(id=20, name="New", tier_id=2, category_id=1),
            commit=commit,
        )
    await async_session.commit()

    async_session.expire_all()
    names = await _names(async_session, unrestricted)
    assert denied.value.reason == "check"
    assert str(denied.value) == "the created ModelTest doesn't pass create_check"
    assert names[9] == "Changed by the caller"
    assert 20 not in names


async def test_a_created_row_outside_read_is_returned(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(
        PolicyConfig(read=test_model.category_id == 1, create=True)
    )

    created = await crud.create(
        async_session,
        ItemCreate(id=20, name="New", tier_id=1, category_id=2),
        schema_to_select=ItemRead,
    )

    assert created is not None
    assert (created["id"], created["category_id"]) == (20, 2)
    assert await crud.get(async_session, id=20) is None


async def test_create_refuses_relationship_data(async_session, crud):
    with pytest.raises(AccessDenied) as denied:
        await crud.create(
            async_session,
            ItemCreateWithTier(id=20, name="New", tier_id=1, tier={"id": 9}),
        )

    assert denied.value.reason == "columns"
    assert denied.value.columns == ("tier",)


async def test_create_takes_a_relationship_left_empty(
    async_session, crud, unrestricted
):
    await crud.create(async_session, ItemCreateWithTier(id=20, name="New", tier_id=1))

    assert await unrestricted.exists(async_session, id=20)


@pytest.mark.parametrize(
    "upsert",
    [
        lambda crud, db, item: crud.upsert(db, item),
        lambda crud, db, item: crud.upsert_multi(db, [item]),
    ],
    ids=["upsert", "upsert_multi"],
)
async def test_upserts_refuse_a_conflicting_row_outside_read(
    async_session, crud, upsert
):
    outside_read = ItemCreate(id=3, name="Taken", tier_id=1, category_id=2)

    with pytest.raises(AccessDenied) as denied:
        await upsert(crud, async_session, outside_read)

    assert denied.value.operation == "update"
    assert denied.value.reason == "row"


async def test_select_carries_the_read_condition(async_session, crud, test_model):
    statement = await crud.select()

    rows = await async_session.execute(statement.where(test_model.id.in_([1, 3])))

    assert [row.id for row in rows] == [1]


async def test_checks_match_rows_by_a_composite_primary_key(async_session):
    async_session.add(MultiPkModel(id=1, uuid="a", name="First"))
    await async_session.commit()
    crud = FastCRUD(MultiPkModel).with_policy(
        PolicyConfig(read=True, update=True, update_check=MultiPkModel.name != "Taken")
    )

    await crud.update(async_session, {"name": "Kept"}, id=1, uuid="a")
    with pytest.raises(AccessDenied) as denied:
        await crud.update(async_session, {"name": "Taken"}, id=1, uuid="a")

    row = await FastCRUD(MultiPkModel).get(async_session, id=1, uuid="a")
    assert denied.value.reason == "check"
    assert row is not None and row["name"] == "Kept"
