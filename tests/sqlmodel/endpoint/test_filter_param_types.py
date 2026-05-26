"""
SQLModel parity for :mod:`tests.sqlalchemy.endpoint.test_filter_param_types`.

Verifies that ``crud_router`` exposes correct column types for filter query
parameters in the generated OpenAPI schema when models are declared with
SQLModel. See issue #291.
"""

from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from pydantic import BaseModel
from sqlmodel import Field, SQLModel

from fastcrud import FastCRUD, FilterConfig, crud_router

from ..conftest import ModelTest


class FilterParamTypesModel(SQLModel, table=True):
    __tablename__ = "filter_param_types_test"
    id: int = Field(primary_key=True)
    int_param: int | None = Field(default=None)
    float_param: float | None = Field(default=None)
    str_param: str | None = Field(default=None)
    bool_param: bool | None = Field(default=None)
    uuid_param: UUID | None = Field(default_factory=uuid4)


class NullSchema(BaseModel):
    pass


def _openapi_param_schema(app: FastAPI, path: str, name: str) -> dict[str, Any]:
    op = app.openapi()["paths"][path]["get"]
    param = next(p for p in op["parameters"] if p["name"] == name)
    return cast(dict[str, Any], param["schema"])


def _router_for(model: Any, filter_config: FilterConfig, **kwargs: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(
        crud_router(
            session=lambda: None,  # type: ignore[arg-type, return-value]
            model=model,
            create_schema=NullSchema,
            update_schema=NullSchema,
            filter_config=filter_config,
            path="/items",
            tags=["items"],
            **kwargs,
        )
    )
    return app


@pytest.mark.asyncio
async def test_simple_column_types_exposed_in_openapi():
    app = _router_for(
        FilterParamTypesModel,
        FilterConfig(int_param=None, float_param=None, str_param=None, bool_param=None),
    )

    assert _openapi_param_schema(app, "/items", "int_param")["type"] == "integer"
    assert _openapi_param_schema(app, "/items", "float_param")["type"] == "number"
    assert _openapi_param_schema(app, "/items", "str_param")["type"] == "string"
    assert _openapi_param_schema(app, "/items", "bool_param")["type"] == "boolean"


@pytest.mark.asyncio
async def test_sqlmodel_guid_type_resolves_to_string():
    """SQLModel's GUID column type must resolve to a string-format OpenAPI param.

    Regression for the ``is_uuid_type`` branch that recognises SQLModel's GUID
    by class name.
    """
    app = _router_for(FilterParamTypesModel, FilterConfig(uuid_param=None))

    schema = _openapi_param_schema(app, "/items", "uuid_param")
    # FastAPI emits UUID as string with a uuid format.
    assert schema["type"] == "string"
    assert schema.get("format") == "uuid"


@pytest.mark.asyncio
async def test_operator_filters_keep_column_type():
    app = _router_for(
        FilterParamTypesModel,
        FilterConfig(int_param__gte=None, str_param__ilike=None),
    )

    assert _openapi_param_schema(app, "/items", "int_param__gte")["type"] == "integer"
    assert _openapi_param_schema(app, "/items", "str_param__ilike")["type"] == "string"


@pytest.mark.asyncio
async def test_collection_operators_render_as_array():
    app = _router_for(
        FilterParamTypesModel,
        FilterConfig(int_param__in=None, str_param__between=None),
    )

    in_schema = _openapi_param_schema(app, "/items", "int_param__in")
    assert in_schema["type"] == "array"
    assert in_schema["items"]["type"] == "integer"

    between_schema = _openapi_param_schema(app, "/items", "str_param__between")
    assert between_schema["type"] == "array"
    assert between_schema["items"]["type"] == "string"


@pytest.mark.asyncio
async def test_joined_filter_uses_related_column_type():
    app = _router_for(ModelTest, FilterConfig(**{"tier.name__eq": None}))

    schema = _openapi_param_schema(app, "/items", "tier.name__eq")
    assert schema["type"] == "string"


@pytest.mark.asyncio
async def test_custom_filters_still_work_with_typed_params():
    """Regression: v0.20.0 custom_filters should not be broken by the type wiring."""
    from sqlalchemy import func

    custom_filters = {
        "year": lambda col: lambda val: func.strftime("%Y", col) == str(val),
    }

    app = _router_for(
        FilterParamTypesModel,
        FilterConfig(int_param__year=None),
        crud=FastCRUD(FilterParamTypesModel, custom_filters=custom_filters),
        custom_filters=custom_filters,
    )

    schema = _openapi_param_schema(app, "/items", "int_param__year")
    assert schema["type"] == "integer"
