import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from fastcrud import FastCRUD, EndpointCreator
from tests.sqlalchemy.conftest import Author


class ArticleSchemaLocal(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    content: str | None = None


class AuthorSchemaLocal(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    articles: list[ArticleSchemaLocal] | None = []


@pytest.mark.asyncio
async def test_create_endpoint_with_relationships_coverage(async_session: AsyncSession):
    # Set up EndpointCreator for Author which has articles relationship
    author_crud: FastCRUD = FastCRUD(Author)
    endpoint_creator = EndpointCreator(
        session=lambda: async_session,
        model=Author,
        create_schema=AuthorSchemaLocal,
        update_schema=AuthorSchemaLocal,
        select_schema=AuthorSchemaLocal,
        crud=author_crud,
        path="/authors",
        tags=["authors"],
        include_relationships=True,
        include_one_to_many=True,
    )

    app = FastAPI()
    endpoint_creator.add_routes_to_router()
    app.include_router(endpoint_creator.router)

    # We need to mock the dependency for db session since EndpointCreator uses session_generator
    # which in many cases is overridden or used as a dependency.
    # But here we can just use the app directly if we can manage the session.

    # Actually, it's easier to just call the endpoint function directly if we can access it.
    # Or use AsyncClient with the app.

    @app.on_event("startup")
    async def startup():
        app.state.db = async_session

    # Override the dependency if necessary.
    # EndpointCreator uses self.session_generator as a dependency.

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        # Test create with nested articles
        response = await ac.post(
            "/authors",
            json={
                "id": 1,
                "name": "Author 1",
                "articles": [{"id": 1, "title": "Article 1", "content": "Content 1"}],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Check DB
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        stmt = (
            select(Author).where(Author.id == 1).options(selectinload(Author.articles))
        )
        res = await async_session.execute(stmt)
        _ = res.scalar_one()

        assert data["name"] == "Author 1"
        assert len(data["articles"]) == 1
        assert data["articles"][0]["title"] == "Article 1"

        # Test create with auto_fields=True (this is where the missing lines are)
        # We need a model with auto fields. ModelTest has id as auto-increment.
        from tests.sqlalchemy.conftest import (
            ModelTest,
            ReadSchemaTest,
            UpdateSchemaTest,
        )
        from fastcrud import CreateConfig

        class CreateSchemaLocal(BaseModel):
            tier_id: int

        model_crud: FastCRUD = FastCRUD(ModelTest)
        model_endpoint = EndpointCreator(
            session=lambda: async_session,
            model=ModelTest,
            create_schema=CreateSchemaLocal,
            update_schema=UpdateSchemaTest,
            select_schema=ReadSchemaTest,
            crud=model_crud,
            path="/models",
            create_config=CreateConfig(
                auto_fields={"name": lambda: "Injected Name"}
            ),
        )
        model_endpoint.add_routes_to_router()
        app.include_router(model_endpoint.router)

        response = await ac.post("/models", json={"tier_id": 1})
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["name"] == "Injected Name"
        assert "id" in res_data
