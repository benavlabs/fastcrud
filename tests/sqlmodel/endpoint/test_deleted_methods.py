import pytest
from fastapi.testclient import TestClient
from fastcrud import FastCRUD, crud_router


@pytest.mark.asyncio
async def test_deleted_methods(
    client: TestClient,
    async_session,
    test_model,
    create_schema,
    update_schema,
    read_schema,
):
    custom_router = crud_router(
        session=lambda: async_session,
        model=test_model,
        crud=FastCRUD(test_model),
        create_schema=create_schema,
        update_schema=update_schema,
        select_schema=read_schema,  # v0.20.0: provide select_schema to get data back from create
        deleted_methods=["delete"],
        path="/test_custom",
        tags=["Test"],
        endpoint_names={
            "create": "create",
            "read": "get",
            "update": "update",
            "delete": "delete",
            "db_delete": "db_delete",
            "read_multi": "get_multi",
        },
    )

    client.app.include_router(custom_router)  # type: ignore

    response = client.post(
        "/test_custom/create", json={"name": "Test Item", "tier_id": 1}
    )
    assert response.status_code == 200

    item_id = response.json()["id"]
    response = client.delete(f"/test_custom/delete/{item_id}")
    assert response.status_code == 404


def test_invalid_deleted_methods(endpoint_creator):
    with pytest.raises(ValueError) as exc_info:
        endpoint_creator.add_routes_to_router(deleted_methods=["swim", "jump"])
    assert "Invalid CRUD methods in deleted_methods" in str(exc_info.value)
