import pytest

from fastapi.testclient import TestClient
from fastcrud import crud_router


@pytest.mark.asyncio
async def test_endpoint_custom_names(
    client: TestClient,
    test_data,
    async_session,
    test_model,
    create_schema,
    update_schema,
    read_schema,
):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    custom_endpoint_names = {
        "create": "add",
        "read": "fetch",
    }

    custom_router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        select_schema=read_schema,  # v0.20.0: provide select_schema to get data back from create
        endpoint_names=custom_endpoint_names,
        path="/test_custom_names",
        tags=["TestCustomNames"],
    )

    client.app.include_router(custom_router)  # type: ignore

    create_response = client.post(
        "/test_custom_names/add", json={"name": "Custom Endpoint Item", "tier_id": 1}
    )
    assert (
        create_response.status_code == 200
    ), "Failed to create item with custom endpoint name"

    item_id = create_response.json()["id"]

    fetch_response = client.get(f"/test_custom_names/fetch/{item_id}")
    assert (
        fetch_response.status_code == 200
    ), "Failed to fetch item with custom endpoint name"
    assert (
        fetch_response.json()["id"] == item_id
    ), "Fetched item ID does not match created item ID"
