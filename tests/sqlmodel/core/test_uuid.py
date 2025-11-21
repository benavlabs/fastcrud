import pytest
from uuid import UUID, uuid4

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.types import TypeDecorator
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Field, SQLModel

from fastcrud import crud_router, FastCRUD
from fastcrud import FilterConfig
from fastcrud.core import create_dynamic_filters
from pydantic import ConfigDict


class UUIDType(TypeDecorator):
    """Platform-independent UUID type.
    Uses PostgreSQL's UUID type, otherwise CHAR(36)
    """

    impl = String
    cache_ok = True

    def __init__(self):
        super().__init__(36)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":  # pragma: no cover
            return dialect.type_descriptor(PostgresUUID(as_uuid=True))
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:  # pragma: no cover
            return value
        elif dialect.name == "postgresql":  # pragma: no cover
            return value
        else:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value  # pragma: no cover
        if not isinstance(value, UUID):
            return UUID(value)
        return value  # pragma: no cover


class UUIDModel(SQLModel, table=True):
    __tablename__ = "uuid_test"
    id: UUID = Field(
        default_factory=uuid4, sa_column=Column(UUIDType(), primary_key=True)
    )
    name: str = Field(sa_column=Column(String(255)))


class CustomUUID(TypeDecorator):
    """Custom UUID type for testing."""

    impl = String
    cache_ok = True

    def __init__(self):
        super().__init__(36)
        self.__visit_name__ = "uuid"

    def process_bind_param(self, value, dialect):
        if value is None:  # pragma: no cover
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:  # pragma: no cover
            return value
        return UUID(value)


class CustomUUIDModel(SQLModel, table=True):
    __tablename__ = "custom_uuid_test"
    id: UUID = Field(
        default_factory=uuid4, sa_column=Column(CustomUUID(), primary_key=True)
    )
    name: str = Field(sa_column=Column(String(255)))


class UUIDSchema(SQLModel):
    id: UUID
    name: str

    model_config = ConfigDict(from_attributes=True)


class CreateUUIDSchema(SQLModel):
    name: str

    model_config = ConfigDict(from_attributes=True)


class UpdateUUIDSchema(SQLModel):
    name: str

    model_config = ConfigDict(from_attributes=True)


@pytest.fixture
def uuid_client(async_session):
    app = FastAPI()

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=UUIDModel,
            crud=FastCRUD(UUIDModel),
            create_schema=CreateUUIDSchema,
            update_schema=UpdateUUIDSchema,
            path="/uuid-test",
            tags=["uuid-test"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "read_multi": "get_multi",
            },
        )
    )

    app.include_router(
        crud_router(
            session=lambda: async_session,
            model=CustomUUIDModel,
            crud=FastCRUD(CustomUUIDModel),
            create_schema=CreateUUIDSchema,
            update_schema=UpdateUUIDSchema,
            path="/custom-uuid-test",
            tags=["custom-uuid-test"],
            endpoint_names={
                "create": "create",
                "read": "get",
                "update": "update",
                "delete": "delete",
                "read_multi": "get_multi",
            },
        )
    )

    return TestClient(app)


@pytest.mark.asyncio
@pytest.mark.dialect("sqlite")
async def test_custom_uuid_crud(uuid_client):
    # v0.20.0 behavior: create endpoint without select_schema returns null
    response = uuid_client.post("/custom-uuid-test/create", json={"name": "test"})
    assert (
        response.status_code == 200
    ), f"Creation failed with response: {response.text}"

    # Verify that creation returns null (None) as expected in v0.20.0
    assert response.json() is None, f"Expected null response, got: {response.text}"

    # To get the created record, use get_multi endpoint to find it
    get_response = uuid_client.get("/custom-uuid-test/get_multi")
    assert get_response.status_code == 200
    items = get_response.json().get("data", [])

    # Find the created item
    test_item = None
    for item in items:
        if item["name"] == "test":
            test_item = item
            break

    assert test_item is not None, "Created item not found in get_multi response"
    uuid_id = test_item["id"]

    try:
        UUID(uuid_id)
    except ValueError:  # pragma: no cover
        pytest.fail("Invalid UUID format")

    response = uuid_client.get(f"/custom-uuid-test/get/{uuid_id}")
    assert response.status_code == 200
    assert response.json()["id"] == uuid_id
    assert response.json()["name"] == "test"

    update_response = uuid_client.patch(
        f"/custom-uuid-test/update/{uuid_id}", json={"name": "updated"}
    )
    response = uuid_client.get(f"/custom-uuid-test/get/{uuid_id}")
    assert update_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["name"] == "updated"

    response = uuid_client.delete(f"/custom-uuid-test/delete/{uuid_id}")
    assert response.status_code == 200

    response = uuid_client.get(f"/custom-uuid-test/get/{uuid_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_uuid_list_endpoint(uuid_client):
    # v0.20.0 behavior: create endpoint without select_schema returns null
    for i in range(3):
        response = uuid_client.post("/uuid-test/create", json={"name": f"test_{i}"})
        assert response.status_code == 200
        # Verify that creation returns null (None) as expected in v0.20.0
        assert response.json() is None, f"Expected null response, got: {response.text}"

    response = uuid_client.get("/uuid-test/get_multi")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 3

    for item in data:
        try:
            UUID(item["id"])
        except ValueError:  # pragma: no cover
            pytest.fail("Invalid UUID format in list response")


def test_create_dynamic_filters_type_conversion():
    filter_config = FilterConfig(uuid_field=None, int_field=None, str_field=None)
    column_types = {
        "uuid_field": UUID,
        "int_field": int,
        "str_field": str,
    }

    filters_func = create_dynamic_filters(filter_config, column_types)

    test_uuid = "123e4567-e89b-12d3-a456-426614174000"
    result = filters_func(uuid_field=test_uuid, int_field="123", str_field=456)

    assert isinstance(result["uuid_field"], UUID)
    assert result["uuid_field"] == UUID(test_uuid)
    assert isinstance(result["int_field"], int)
    assert result["int_field"] == 123
    assert isinstance(result["str_field"], str)
    assert result["str_field"] == "456"

    result = filters_func(
        uuid_field="not-a-uuid", int_field="not-an-int", str_field=456
    )

    assert result["uuid_field"] == "not-a-uuid"
    assert result["int_field"] == "not-an-int"
    assert isinstance(result["str_field"], str)

    result = filters_func(uuid_field=None, int_field="123", str_field=None)
    assert "uuid_field" not in result
    assert result["int_field"] == 123
    assert "str_field" not in result

    result = filters_func(unknown_field="test")
    assert result["unknown_field"] == "test"

    empty_filters_func = create_dynamic_filters(None, {})
    assert empty_filters_func() == {}
