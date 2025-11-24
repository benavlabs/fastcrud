from fastapi.testclient import TestClient
from fastcrud import EndpointCreator


async def custom_endpoint(foo: str):
    return {"foo": f"{foo}"}


def test_add_custom_route(client: TestClient, endpoint_creator: EndpointCreator):
    endpoint_creator.add_custom_route(
        endpoint=custom_endpoint,
        path="/test-custom-route",
        methods=["GET"],
        tags=["test"],
        summary="Test Custom Route",
        description="This is a test for the custom route.",
    )

    client.app.include_router(endpoint_creator.router)  # type: ignore

    response = client.get("/custom_test/test-custom-route?foo=bar")
    assert response.status_code == 200
    assert response.json() == {"foo": "bar"}

    response = client.get("/custom_test/test-custom-route")
    assert response.status_code == 422


def test_add_custom_route_include_in_schema_false(
    client: TestClient, endpoint_creator: EndpointCreator
):
    endpoint_creator.add_custom_route(
        endpoint=custom_endpoint,
        path="/hidden-custom-route",
        methods=["GET"],
        include_in_schema=False,
        tags=["hidden"],
        summary="Hidden Custom Route",
        description="This is a hidden test for the custom route.",
    )

    client.app.include_router(endpoint_creator.router)  # type: ignore

    response = client.get("/custom_test/hidden-custom-route?foo=hidden")
    assert response.status_code == 200
    assert response.json() == {"foo": "hidden"}

    openapi_schema = client.app.openapi()  # type: ignore
    assert "/custom_test/hidden-custom-route" not in str(openapi_schema)
