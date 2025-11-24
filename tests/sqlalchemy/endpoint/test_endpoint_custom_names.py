import pytest

from fastapi.testclient import TestClient
from fastcrud import crud_router


@pytest.mark.parametrize(
    "custom_endpoint_names, endpoint_paths",
    [
        (
            {"create": "", "read": "", "read_multi": ""},
            ["/test_custom_names", "/test_custom_names", "/test_custom_names"],
        ),
        (
            {"create": "add", "read": "fetch", "read_multi": "fetch_multi"},
            [
                "/test_custom_names/add",
                "/test_custom_names/fetch",
                "/test_custom_names/fetch_multi",
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_endpoint_custom_names(
    client: TestClient,
    test_data,
    async_session,
    test_model,
    create_schema,
    update_schema,
    read_schema,
    custom_endpoint_names,
    endpoint_paths,
):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

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

    create_path, read_path, read_multi_path = endpoint_paths

    create_response = client.post(
        create_path, json={"name": "Custom Endpoint Item", "tier_id": 1}
    )

    assert (
        create_response.status_code == 200
    ), "Failed to create item with custom endpoint name"

    item_id = create_response.json()["id"]

    fetch_response = client.get(f"{read_path}/{item_id}")
    assert (
        fetch_response.status_code == 200
    ), "Failed to fetch item with custom endpoint name"
    assert fetch_response.json()["id"] == item_id, (
        f"Fetched item ID does not match created item ID:"
        f" {fetch_response.json()['id']} != {item_id}"
    )

    fetch_multi_response = client.get(read_multi_path)
    assert (
        fetch_multi_response.status_code == 200
    ), "Failed to fetch multi items with custom endpoint name"
    assert (
        len(fetch_multi_response.json()["data"]) == 12
    ), "Fetched item list has incorrect length"
