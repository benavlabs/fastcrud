import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastcrud import EndpointCreator
from fastcrud.core import FilterConfig
from tests.sqlalchemy.conftest import ModelTest, CreateSchemaTest, UpdateSchemaTest


@pytest.fixture
def app(async_session) -> FastAPI:
    app = FastAPI()
    endpoint_creator = EndpointCreator(
        session=lambda: async_session,
        model=ModelTest,
        create_schema=CreateSchemaTest,
        update_schema=UpdateSchemaTest,
        path="/test",
        filter_config=FilterConfig(id=None, name=None, tier_id=None),
    )
    endpoint_creator.add_routes_to_router()
    app.include_router(endpoint_creator.router)
    return app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.mark.asyncio
async def test_get_multi_with_sort_ascending(client, async_session, test_data):
    # Add test data
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    # Test ascending sort by name
    response = client.get("/test/?sort=name")
    print("Response:", response.status_code, response.json())
    assert response.status_code == 200

    data = response.json()["data"]
    sorted_data = sorted(test_data, key=lambda x: x["name"])

    assert len(data) == len(sorted_data)
    assert [item["name"] for item in data] == [item["name"] for item in sorted_data]


@pytest.mark.asyncio
async def test_get_multi_with_sort_descending(client, async_session, test_data):
    # Add test data
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    # Test descending sort by name
    response = client.get("/test/?sort=-name")
    assert response.status_code == 200

    data = response.json()["data"]
    sorted_data = sorted(test_data, key=lambda x: x["name"], reverse=True)

    assert len(data) == len(sorted_data)
    assert [item["name"] for item in data] == [item["name"] for item in sorted_data]


@pytest.mark.asyncio
async def test_get_multi_with_multiple_sort_fields(client, async_session, test_data):
    # Add test data
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    # Test multiple sort fields (tier_id ascending, name descending)
    response = client.get("/test/?sort=tier_id,-name")
    assert response.status_code == 200

    data = response.json()["data"]

    # Sort first by tier_id (ascending) then by name (descending)
    sorted_data = sorted(test_data, key=lambda x: (x["tier_id"], -ord(x["name"][0])))

    assert len(data) == len(sorted_data)

    # Group by tier_id and check that names are in descending order within each group
    tier_groups = {}
    for item in data:
        tier_id = item["tier_id"]
        if tier_id not in tier_groups:
            tier_groups[tier_id] = []
        tier_groups[tier_id].append(item["name"])

    for tier_id, names in tier_groups.items():
        if len(names) > 1:
            for i in range(len(names) - 1):
                assert (
                    names[i] >= names[i + 1]
                ), f"Names in tier {tier_id} are not in descending order"


@pytest.mark.asyncio
async def test_get_multi_with_sort_and_pagination(client, async_session, test_data):
    # Add test data
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    # Test sorting with pagination
    response = client.get("/test/?sort=name&page=1&itemsPerPage=5")
    assert response.status_code == 200

    data = response.json()["data"]
    sorted_data = sorted(test_data, key=lambda x: x["name"])[:5]

    assert len(data) <= 5
    assert [item["name"] for item in data] == [item["name"] for item in sorted_data]


@pytest.mark.asyncio
async def test_get_multi_with_sort_and_filtering(client, async_session, test_data):
    # Add test data
    for item in test_data:
        async_session.add(ModelTest(**item))
    await async_session.commit()

    # Test sorting with filtering
    tier_id_to_filter = 1
    response = client.get(f"/test/?sort=name&tier_id={tier_id_to_filter}")
    print("Response:", response.status_code, response.json())
    assert response.status_code == 200

    data = response.json()["data"]
    filtered_data = [item for item in test_data if item["tier_id"] == tier_id_to_filter]
    sorted_filtered_data = sorted(filtered_data, key=lambda x: x["name"])

    assert len(data) == len(sorted_filtered_data)
    assert all(item["tier_id"] == tier_id_to_filter for item in data)
    assert [item["name"] for item in data] == [
        item["name"] for item in sorted_filtered_data
    ]
