import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from fastcrud import FilterConfig
from fastcrud.crud.fast_crud import FastCRUD
from fastcrud.endpoint.crud_router import crud_router
from tests.sqlalchemy.conftest import ProjectPoly, ContractPoly


class ProjectPolyCreate(BaseModel):
    name: str
    contract_id: int | None = None


class ProjectPolyUpdate(BaseModel):
    name: str | None = None
    contract_id: int | None = None


class ProjectPolyDelete(BaseModel):
    pass


@pytest.mark.asyncio
async def test_read_multi_joined_filter_with_joined_inheritance(async_session):
    contract_a = ContractPoly(operator_id=10)
    contract_b = ContractPoly(operator_id=20)

    project_a = ProjectPoly(name="Project A", contract=contract_a)
    project_b = ProjectPoly(name="Project B", contract=contract_b)

    async_session.add_all([contract_a, contract_b, project_a, project_b])
    await async_session.commit()

    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=ProjectPoly,
            crud=FastCRUD(ProjectPoly),
            create_schema=ProjectPolyCreate,
            update_schema=ProjectPolyUpdate,
            delete_schema=ProjectPolyDelete,
            filter_config=FilterConfig(**{"contract.operator_id__eq": None}),
            path="/projects_poly",
            tags=["projects_poly"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "db_delete": "db_delete",
                "read_multi": "get_multi",
            },
        )
    )

    client = TestClient(app)
    response = client.get(
        "/projects_poly/get_multi", params={"contract.operator_id__eq": 10}
    )

    assert response.status_code == 200
    payload = response.json()
    assert "data" in payload
    assert len(payload["data"]) == 1
    assert payload["data"][0]["name"] == "Project A"
