"""
Verify that ``crud_router`` exposes correct column types for filter query
parameters in the generated OpenAPI schema.

Before this was wired up, every filter parameter rendered as ``any``,
regardless of the underlying column type. See issue #291.
"""

from typing import Any, cast

import pytest
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, Float, Integer, String

from fastcrud import FastCRUD, FilterConfig, crud_router
from fastcrud.core import FilterProcessor, create_dynamic_filters

from ..conftest import Base, ModelTest


class FilterParamTypesModel(Base):
    __tablename__ = "filter_param_types_test"
    id = Column(Integer, primary_key=True)
    int_param = Column(Integer)
    float_param = Column(Float)
    str_param = Column(String(32))
    bool_param = Column(Boolean)


class NullSchema(BaseModel):
    pass


def _openapi_param_schema(app: FastAPI, path: str, name: str) -> dict[str, Any]:
    """Return the ``schema`` block for a query parameter in the GET op of *path*."""
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
    """Each scalar column kind should surface its proper OpenAPI type."""
    app = _router_for(
        FilterParamTypesModel,
        FilterConfig(int_param=None, float_param=None, str_param=None, bool_param=None),
    )

    assert _openapi_param_schema(app, "/items", "int_param")["type"] == "integer"
    assert _openapi_param_schema(app, "/items", "float_param")["type"] == "number"
    assert _openapi_param_schema(app, "/items", "str_param")["type"] == "string"
    assert _openapi_param_schema(app, "/items", "bool_param")["type"] == "boolean"


@pytest.mark.asyncio
async def test_operator_filters_keep_column_type():
    """``field__gte`` / ``field__ilike`` should inherit the column's type."""
    app = _router_for(
        FilterParamTypesModel,
        FilterConfig(int_param__gte=None, str_param__ilike=None),
    )

    assert _openapi_param_schema(app, "/items", "int_param__gte")["type"] == "integer"
    assert _openapi_param_schema(app, "/items", "str_param__ilike")["type"] == "string"


@pytest.mark.asyncio
async def test_collection_operators_render_as_array():
    """``__in`` / ``__not_in`` / ``__between`` should be typed as arrays."""
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
    """``related.field__op`` should pick up the related model's column type."""
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

    # The operator is the suffix, the value type is still the column's (integer).
    schema = _openapi_param_schema(app, "/items", "int_param__year")
    assert schema["type"] == "integer"


def test_filter_func_passes_collection_values_through_unchanged():
    """Collection operators (``__in`` / ``__between`` / ``__not_in``) skip the
    per-value coercion path — the typed annotation lets FastAPI parse the list."""
    filter_config = FilterConfig(int_param__in=None, str_param__between=None)
    filters_func = create_dynamic_filters(filter_config, FilterParamTypesModel)

    result = filters_func(int_param__in=[1, 2, 3], str_param__between=["a", "z"])

    assert result == {
        "int_param__in": [1, 2, 3],
        "str_param__between": ["a", "z"],
    }


def test_interpret_filters_rejects_non_relationship_dotted_path():
    """A dotted filter whose first segment is a column, not a relationship,
    must raise — otherwise we'd try to join on a non-existent mapper."""
    processor = FilterProcessor(FilterParamTypesModel)

    with pytest.raises(ValueError, match="Invalid relationship 'int_param'"):
        processor.interpret_filters(FilterConfig(**{"int_param.something__eq": None}))
