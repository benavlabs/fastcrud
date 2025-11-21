from pydantic import BaseModel
import pytest
from fastapi import FastAPI

from fastcrud import crud_router, FilterConfig


class NullSchema(BaseModel):
    pass


@pytest.fixture
def app_with_filter_params(test_model_with_types, async_session):
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=test_model_with_types,
            create_schema=NullSchema,
            update_schema=NullSchema,
            filter_config=FilterConfig(
                int_param=None, float_param=None, str_param=None, bool_param=None
            ),
            path="/test",
            tags=["test"],
        )
    )

    return app


@pytest.mark.asyncio
async def test_dependency_filtered_endpoint(app_with_filter_params):
    """Test that filter query parameters are correctly typed in the OpenAPI schema."""

    schema = app_with_filter_params.openapi()

    def get_type(param_name: str):
        params = schema["paths"]["/test"]["get"]["parameters"]
        param = next(item for item in params if item["name"] == param_name)
        return param["schema"]["type"]

    assert get_type("int_param") == "integer"
    assert get_type("float_param") == "number"
    assert get_type("str_param") == "string"
    assert get_type("bool_param") == "boolean"
