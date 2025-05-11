import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from fastcrud import crud_router, FilterConfig
from tests.sqlalchemy.conftest import test_model_with_org, ModelWithOrgTest



class UserInfo:
    def __init__(self, organization_id: int):
        self.organization_id = organization_id


async def get_auth_user():
    return UserInfo(organization_id=123)


async def get_org_id(auth: UserInfo = Depends(get_auth_user)):
    return auth.organization_id

# Mock the get_org_id function to return a specific organization ID
async def mock_get_org_id(*args, **kwargs):
    return 42  # This should match the organization_id we set for some test items


@pytest.fixture
def dependency_filtered_client(
        test_model_with_org, create_schema, update_schema, delete_schema, async_session, monkeypatch
):
    monkeypatch.setattr("tests.sqlalchemy.endpoint.test_dependency_filter.get_org_id", mock_get_org_id)

    app = FastAPI()

    #  Include the router.  Crucially, pass the session *function*, not the session itself.
    app.include_router(
        crud_router(
            session=lambda: async_session,  # Pass a *callable* that returns a session
            model=test_model_with_org,
            create_schema=create_schema,
            update_schema=update_schema,
            delete_schema=delete_schema,
            filter_config=FilterConfig(organization_id=get_org_id),
            path="/test",
            tags=["test"],
        )
    )

    return TestClient(app)


@pytest.mark.asyncio
async def test_dependency_filtered_endpoint(dependency_filtered_client, test_data, monkeypatch, async_session):
    # Create test data with different organization IDs
    for i, item in enumerate(test_data):
        item["organization_id"] = 42 if i % 2 == 0 else 99

    tier_id = 1

    # Create test items directly in the database
    for i in range(10):
        test_item = ModelWithOrgTest(
            name=f"Test Item {i}",
            tier_id=tier_id,
            organization_id=42 if i < 5 else 99  # First 5 items have org_id=42, rest have org_id=99
        )
        async_session.add(test_item)
    await async_session.commit()

    # Get all items - should only return items with organization_id=42
    # Add the required query parameters
    response = dependency_filtered_client.get("/test", params={"args": "", "kwargs": ""})

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) > 0
    for item in data:
        assert item['organization_id'] == 42