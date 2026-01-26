from typing import Optional, Callable, Sequence

import pytest
from fastapi.testclient import TestClient
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastcrud import FastCRUD, crud_router, EndpointCreator


class CustomEndpointCreator(EndpointCreator):
    def _custom_route(self):
        async def custom_endpoint(db: AsyncSession = Depends(self.session)):
            return {"message": "Custom route"}

        return custom_endpoint

    def add_routes_to_router(
        self,
        create_deps: Sequence[Callable] = [],
        read_deps: Sequence[Callable] = [],
        read_multi_deps: Sequence[Callable] = [],
        update_deps: Sequence[Callable] = [],
        delete_deps: Sequence[Callable] = [],
        db_delete_deps: Sequence[Callable] = [],
        included_methods: Sequence[str] | None = None,
        deleted_methods: Sequence[str] | None = None,
    ):
        super().add_routes_to_router(
            create_deps,
            read_deps,
            read_multi_deps,
            update_deps,
            delete_deps,
            db_delete_deps,
            included_methods,
            deleted_methods,
        )

        self.router.add_api_route(
            path="/custom",
            endpoint=self._custom_route(),
            methods=["GET"],
            tags=self.tags,
        )


@pytest.mark.asyncio
async def test_custom_endpoint_creator(
    client: TestClient, async_session, test_model, create_schema, update_schema
):
    custom_router = crud_router(
        session=lambda: async_session,
        model=test_model,
        crud=FastCRUD(test_model),
        create_schema=create_schema,
        update_schema=update_schema,
        endpoint_creator=CustomEndpointCreator,
        path="/custom",
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

    response = client.get("/custom")
    assert response.status_code == 200
    assert response.json() == {"message": "Custom route"}
