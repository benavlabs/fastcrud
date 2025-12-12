"""Test that generated endpoints have proper operationId values in OpenAPI schema."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastcrud import crud_router


@pytest.fixture
def app_with_crud_router(async_session, test_model, create_schema, update_schema):
    """Create a FastAPI app with crud_router endpoints."""
    app = FastAPI()

    router = crud_router(
        session=lambda: async_session,
        model=test_model,
        create_schema=create_schema,
        update_schema=update_schema,
        path="/items",
        tags=["Items"],
    )
    app.include_router(router)

    return app


@pytest.fixture
def client_for_openapi(app_with_crud_router):
    """Create a test client for OpenAPI schema testing."""
    return TestClient(app_with_crud_router)


def test_endpoint_operation_ids_use_model_name(client_for_openapi, test_model):
    """Test that operationId values include the model name, not just 'endpoint'."""
    response = client_for_openapi.get("/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()
    paths = openapi_schema["paths"]

    model_name = test_model.__name__.lower()

    # Check POST /items (create)
    assert "/items" in paths
    create_operation = paths["/items"]["post"]
    assert "operationId" in create_operation
    assert model_name in create_operation["operationId"].lower()
    assert "create" in create_operation["operationId"].lower()

    # Check GET /items (read_multi)
    read_multi_operation = paths["/items"]["get"]
    assert "operationId" in read_multi_operation
    assert model_name in read_multi_operation["operationId"].lower()
    assert "read" in read_multi_operation["operationId"].lower()

    # Check GET /items/{id} (read)
    item_path = "/items/{id}"
    assert item_path in paths
    read_operation = paths[item_path]["get"]
    assert "operationId" in read_operation
    assert model_name in read_operation["operationId"].lower()
    assert "read" in read_operation["operationId"].lower()

    # Check PATCH /items/{id} (update)
    update_operation = paths[item_path]["patch"]
    assert "operationId" in update_operation
    assert model_name in update_operation["operationId"].lower()
    assert "update" in update_operation["operationId"].lower()

    # Check DELETE /items/{id} (delete)
    delete_operation = paths[item_path]["delete"]
    assert "operationId" in delete_operation
    assert model_name in delete_operation["operationId"].lower()
    assert "delete" in delete_operation["operationId"].lower()


def test_endpoint_operation_ids_not_generic_endpoint(client_for_openapi):
    """Test that operationId values are NOT just 'endpoint' (the bug we're fixing)."""
    response = client_for_openapi.get("/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()
    paths = openapi_schema["paths"]

    for path, methods in paths.items():
        for method, operation in methods.items():
            if isinstance(operation, dict) and "operationId" in operation:
                operation_id = operation["operationId"]
                # The operationId should NOT be just "endpoint" or "endpoint_*"
                assert not operation_id.startswith("endpoint_"), (
                    f"Route {method.upper()} {path} has generic operationId: {operation_id}"
                )
                # Also check it's not exactly "endpoint"
                assert operation_id != "endpoint", (
                    f"Route {method.upper()} {path} has generic operationId: {operation_id}"
                )
