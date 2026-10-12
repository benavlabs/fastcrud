"""``crud_router(policy=...)``: a per-request ``PolicyConfig``, with 403s and 404s."""

import pytest
import pytest_asyncio
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from pydantic import BaseModel

from fastcrud import (
    CreateConfig,
    DeleteConfig,
    FastCRUD,
    FilterConfig,
    JoinConfig,
    PolicyConfig,
    UpdateConfig,
    crud_router,
)
from ..conftest import CategoryModel

pytestmark = pytest.mark.asyncio


class ItemCreate(BaseModel):
    name: str
    tier_id: int | None = None
    category_id: int | None = None


class ItemUpdate(BaseModel):
    name: str | None = None
    tier_id: int | None = None
    category_id: int | None = None


class ItemDelete(BaseModel):
    pass


class ItemRead(BaseModel):
    id: int
    tier_id: int
    category_id: int | None
    name: str | None = None


class TierCreate(BaseModel):
    name: str


@pytest_asyncio.fixture
async def seeded(
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
    return async_session


def _viewer_policy(test_model, **overrides):
    return PolicyConfig(
        **{
            "read": test_model.category_id == 1,
            "create": True,
            "update": test_model.tier_id == 1,
            "delete": test_model.tier_id == 1,
            **overrides,
        }
    )


def _client(async_session, test_model, policy=None, crud=None, **router_options):
    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model,
            crud=crud or FastCRUD(test_model),
            create_schema=ItemCreate,
            update_schema=ItemUpdate,
            delete_schema=ItemDelete,
            path="/items",
            tags=["items"],
            policy=policy,
            **router_options,
        )
    )
    return TestClient(app)


def _role_policy(test_model, **overrides):
    def policy_for_role(x_role: str = Header("viewer")):
        if x_role == "admin":
            return None
        return _viewer_policy(test_model, **overrides)

    return policy_for_role


async def test_reads_reach_only_rows_in_the_requests_policy(seeded, test_model):
    client = _client(seeded, test_model, policy=_role_policy(test_model))

    listed = client.get("/items", params={"limit": 20}).json()

    assert client.get("/items/1").status_code == 200
    assert client.get("/items/3").status_code == 404
    assert sorted(item["id"] for item in listed["data"]) == [1, 2, 4, 5, 8, 9, 11]
    assert client.get("/items/3", headers={"x-role": "admin"}).status_code == 200


async def test_a_denied_operation_is_a_403(seeded, test_model):
    client = _client(seeded, test_model, policy=_role_policy(test_model, create=False))

    response = client.post("/items", json={"name": "New", "tier_id": 1})

    assert response.status_code == 403
    assert (
        response.json()["detail"] == "create on ModelTest is not allowed by the policy"
    )


async def test_writes_reach_only_rows_in_the_requests_policy(seeded, test_model):
    client = _client(seeded, test_model, policy=_role_policy(test_model))

    assert client.patch("/items/1", json={"name": "Renamed"}).status_code == 200
    assert client.patch("/items/2", json={"name": "Renamed"}).status_code == 404
    assert client.patch("/items/3", json={"name": "Renamed"}).status_code == 404
    assert client.delete("/items/2").status_code == 404
    assert client.delete("/items/db_delete/3").status_code == 404
    assert client.delete("/items/db_delete/5").status_code == 200


async def test_a_failed_check_is_a_403(seeded, test_model):
    client = _client(seeded, test_model, policy=_role_policy(test_model))

    response = client.patch("/items/1", json={"tier_id": 2})

    assert response.status_code == 403
    assert (
        response.json()["detail"] == "the updated ModelTest doesn't pass update_check"
    )


async def test_a_hidden_column_in_a_filter_or_sort_is_a_403(seeded, test_model):
    client = _client(
        seeded,
        test_model,
        policy=_role_policy(test_model, hidden_columns={"name"}),
        filter_config=FilterConfig(name=None),
    )

    filtered = client.get("/items", params={"name": "Alice"})
    sorted_by = client.get("/items", params={"sort": "name"})
    listed = client.get("/items").json()

    assert filtered.status_code == 403
    assert (
        filtered.json()["detail"] == "read on ModelTest may not filter or sort by: name"
    )
    assert sorted_by.status_code == 403
    assert all("name" not in item for item in listed["data"])


async def test_db_delete_of_a_missing_row_is_a_404_without_a_policy(seeded, test_model):
    client = _client(seeded, test_model)

    assert client.delete("/items/db_delete/99").status_code == 404


async def test_a_crud_with_its_own_policy_keeps_it_for_every_request(
    seeded, test_model
):
    crud = FastCRUD(test_model, policy=_viewer_policy(test_model))
    client = _client(seeded, test_model, crud=crud)

    assert client.get("/items/1").status_code == 200
    assert client.get("/items/3", headers={"x-role": "admin"}).status_code == 404


async def test_a_crud_policy_and_a_request_policy_together_are_refused(
    seeded, test_model
):
    crud = FastCRUD(test_model, policy=_viewer_policy(test_model))

    with pytest.raises(ValueError, match="already has a policy"):
        _client(seeded, test_model, policy=_role_policy(test_model), crud=crud)


def _server_tier():
    return 1


async def test_auto_fields_are_written_by_the_server_past_read_only_columns(
    seeded, test_model, unrestricted_reads
):
    client = _client(
        seeded,
        test_model,
        policy=_role_policy(test_model, read_only_columns={"tier_id"}),
        create_config=CreateConfig(auto_fields={"tier_id": _server_tier}),
        update_config=UpdateConfig(auto_fields={"tier_id": _server_tier}),
    )

    refused = client.post("/items", json={"name": "New", "tier_id": 2})
    created = client.post("/items", json={"name": "New", "category_id": 1})
    updated = client.patch("/items/5", json={"name": "Renamed"})

    rows = await unrestricted_reads.get_multi(seeded, name="New")
    assert refused.status_code == 403
    assert refused.json()["detail"] == "create on ModelTest may not write: tier_id"
    assert created.status_code == 200
    assert [row["tier_id"] for row in rows["data"]] == [1]
    assert updated.status_code == 200


def _deleted_by_server():
    return "Deleted by the server"


async def test_the_delete_endpoints_auto_fields_are_written_without_the_policy(
    seeded, test_model, unrestricted_reads
):
    client = _client(
        seeded,
        test_model,
        policy=_role_policy(test_model, update=False),
        delete_config=DeleteConfig(auto_fields={"name": _deleted_by_server}),
    )

    response = client.delete("/items/1")

    row = await unrestricted_reads.get(seeded, id=1)
    assert response.status_code == 200
    assert row is not None and row["is_deleted"] is True
    assert row["name"] == "Deleted by the server"


async def test_the_unique_value_check_runs_without_the_policy(
    seeded, test_model, tier_model
):
    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: seeded,
            model=tier_model,
            crud=FastCRUD(tier_model),
            create_schema=TierCreate,
            update_schema=TierCreate,
            path="/tiers",
            tags=["tiers"],
            policy=lambda: PolicyConfig(read=tier_model.name != "Premium", create=True),
        )
    )
    client = TestClient(app)

    duplicate = client.post("/tiers", json={"name": "Premium"})
    created = client.post("/tiers", json={"name": "Gold"})

    assert duplicate.status_code == 422
    assert created.status_code == 200


async def test_a_written_row_outside_read_is_read_back(seeded, test_model, tier_model):
    client = _client(
        seeded,
        test_model,
        policy=_role_policy(test_model, update=True),
        select_schema=ItemRead,
        joins_config=[
            JoinConfig(
                model=tier_model,
                join_on=test_model.tier_id == tier_model.id,
                join_prefix="tier_",
            )
        ],
    )

    response = client.patch("/items/1", json={"category_id": 2})

    assert response.status_code == 200
    assert response.json()["category_id"] == 2
    assert client.get("/items/1").status_code == 404


@pytest.fixture
def unrestricted_reads(test_model):
    return FastCRUD(test_model)


def _tier_client(async_session, tier_model, policy):
    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=tier_model,
            crud=FastCRUD(tier_model),
            create_schema=TierCreate,
            update_schema=TierCreate,
            path="/tiers",
            tags=["tiers"],
            policy=lambda: policy,
        )
    )
    return TestClient(app)


@pytest.mark.parametrize(
    ("policy_options", "detail"),
    [
        (
            {"create": True, "hidden_columns": {"name"}},
            "create on TierModel may not write: name",
        ),
        ({"create": False}, "create on TierModel is not allowed by the policy"),
    ],
    ids=["hidden-unique-column", "create-denied"],
)
async def test_the_unique_value_check_never_reveals_a_value_the_caller_cant_write(
    seeded, tier_model, policy_options, detail
):
    client = _tier_client(seeded, tier_model, PolicyConfig(read=True, **policy_options))

    taken = client.post("/tiers", json={"name": "Premium"})
    free = client.post("/tiers", json={"name": "Gold"})

    assert (taken.status_code, taken.json()) == (free.status_code, free.json())
    assert taken.status_code == 403
    assert taken.json()["detail"] == detail
