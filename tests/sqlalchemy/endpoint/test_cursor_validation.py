"""
Tests for cursor pagination validation.

This test file verifies that cursor values are properly validated
against column type constraints (int32, int64, datetime, UUID).
"""

import pytest
from typing import Annotated
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from sqlalchemy import Column, Integer, BigInteger, SmallInteger, String, DateTime
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import declarative_base
from uuid import uuid4

from fastcrud import EndpointCreator
from fastcrud.core import CursorPaginatedRequestQuery
from fastcrud.endpoint.endpoint_creator import EndpointCreator as EC

Base = declarative_base()


class TestCursorModel(Base):
    __tablename__ = "test_cursor_validation"

    id = Column(Integer, primary_key=True)
    big_id = Column(BigInteger)
    small_id = Column(SmallInteger)
    name = Column(String)
    created_at = Column(DateTime)
    uuid_field = Column(PostgresUUID(as_uuid=True))


@pytest.fixture
def app_with_cursor_validation(async_session):
    """Create a FastAPI app with cursor validation."""
    app = FastAPI()

    from pydantic import BaseModel

    class CreateSchema(BaseModel):
        name: str

    class UpdateSchema(BaseModel):
        name: str

    endpoint_creator = EndpointCreator(
        session=lambda: async_session,
        model=TestCursorModel,
        create_schema=CreateSchema,
        update_schema=UpdateSchema,
    )

    # Create a custom endpoint with cursor validation
    validator = endpoint_creator._create_cursor_validator()

    @app.get("/items")
    async def get_items_with_validation(
        query: Annotated[CursorPaginatedRequestQuery, Depends(validator)],
    ):
        """Test endpoint with cursor validation."""
        return {
            "cursor": query.cursor,
            "limit": query.limit,
            "sort_column": query.sort_column,
            "sort_order": query.sort_order,
        }

    return app


@pytest.fixture
def client(app_with_cursor_validation):
    """Create a test client."""
    return TestClient(app_with_cursor_validation)


def test_cursor_int32_overflow(client):
    """Test that cursor values exceeding int32 range are rejected."""
    # int32 max is 2,147,483,647
    response = client.get("/items?cursor=10000000000&sort_column=id")
    assert response.status_code == 400
    data = response.json()
    assert "cursor" in data["detail"].lower() or "invalid" in data["detail"].lower()


def test_cursor_int64_overflow(client):
    """Test that cursor values exceeding int64 range are rejected."""
    # int64 max is 9,223,372,036,854,775,807
    huge_number = "99999999999999999999999999999"
    response = client.get(f"/items?cursor={huge_number}&sort_column=id")
    assert response.status_code == 400
    data = response.json()
    assert "exceeds valid" in data["detail"] and "range" in data["detail"]


def test_cursor_valid_int32(client):
    """Test that valid int32 cursor values are accepted."""
    response = client.get("/items?cursor=1000&sort_column=id")
    assert response.status_code == 200
    data = response.json()
    assert data["cursor"] == "1000" or data["cursor"] == 1000


def test_cursor_int32_overflow_within_int64(client):
    """Test that cursor values exceeding int32 but within int64 are rejected for INTEGER columns."""
    # 3 billion is > INT32_MAX (2.147 billion) but < INT64_MAX
    # Since TestCursorModel.id is Integer (INT32), this should be rejected
    large_int = "3000000000"
    response = client.get(f"/items?cursor={large_int}&sort_column=id")
    assert response.status_code == 400
    data = response.json()
    assert "exceeds valid INTEGER range" in data["detail"]


def test_cursor_datetime_invalid_format(client):
    """Test that invalid datetime cursors are rejected."""
    response = client.get("/items?cursor=not-a-date&sort_column=created_at")
    assert response.status_code == 400
    data = response.json()
    assert "Invalid cursor value" in data["detail"]
    assert "datetime" in data["detail"]


def test_cursor_datetime_valid_format(client):
    """Test that valid datetime cursors are accepted."""
    response = client.get("/items?cursor=2024-01-01T12:00:00&sort_column=created_at")
    assert response.status_code == 200
    data = response.json()
    assert data["cursor"] == "2024-01-01T12:00:00"


def test_cursor_datetime_valid_format_with_z(client):
    """Test that valid datetime cursors with Z suffix are accepted."""
    response = client.get("/items?cursor=2024-01-01T12:00:00Z&sort_column=created_at")
    assert response.status_code == 200
    data = response.json()
    assert data["cursor"] == "2024-01-01T12:00:00Z"


def test_cursor_uuid_invalid_format(client):
    """Test that invalid UUID cursors are rejected."""
    response = client.get("/items?cursor=not-a-uuid&sort_column=uuid_field")
    assert response.status_code == 400
    data = response.json()
    assert "Invalid cursor value" in data["detail"]
    assert "UUID" in data["detail"]


def test_cursor_uuid_valid_format(client):
    """Test that valid UUID cursors are accepted."""
    valid_uuid = str(uuid4())
    response = client.get(f"/items?cursor={valid_uuid}&sort_column=uuid_field")
    assert response.status_code == 200
    data = response.json()
    assert data["cursor"] == valid_uuid


def test_cursor_none_accepted(client):
    """Test that None/missing cursor is accepted."""
    response = client.get("/items?sort_column=id")
    assert response.status_code == 200
    data = response.json()
    assert data["cursor"] is None


def test_cursor_string_for_integer_column(client):
    """Test that string cursor for integer column is properly converted."""
    response = client.get("/items?cursor=123&sort_column=id")
    assert response.status_code == 200


def test_cursor_negative_integer(client):
    """Test that negative integers are handled correctly."""
    response = client.get("/items?cursor=-100&sort_column=id")
    assert response.status_code == 200
    data = response.json()
    assert str(data["cursor"]) == "-100" or data["cursor"] == -100


def test_cursor_invalid_string_for_integer(client):
    """Test that non-numeric string for integer column is rejected."""
    response = client.get("/items?cursor=abc&sort_column=id")
    assert response.status_code == 400
    data = response.json()
    assert "Invalid cursor value" in data["detail"]


def test_cursor_validation_without_sort_column(client):
    """Test that cursor without sort_column uses default 'id' column."""
    response = client.get("/items?cursor=100")
    assert response.status_code == 200
    data = response.json()
    assert data["sort_column"] == "id"


def test_cursor_validation_for_unknown_column(client):
    """Test cursor validation when sort_column type is unknown."""
    # If the column doesn't exist in the model, validation should pass
    # (the actual query will fail later, but validation shouldn't block it)
    response = client.get("/items?cursor=123&sort_column=nonexistent")
    assert response.status_code == 200
    data = response.json()
    assert data["cursor"] == "123" or data["cursor"] == 123


def test_cursor_zero_value(client):
    """Test that cursor value of 0 is handled correctly."""
    response = client.get("/items?cursor=0&sort_column=id")
    assert response.status_code == 200
    data = response.json()
    assert data["cursor"] == "0" or data["cursor"] == 0


def test_cursor_none_direct_validation():
    """Test that _validate_cursor_value returns early for None cursor."""
    EC._validate_cursor_value(None, Integer(), int, "id")


def test_cursor_bigint_valid(client):
    """Test that valid BigInteger cursor values are accepted."""
    response = client.get("/items?cursor=3000000000&sort_column=big_id")
    assert response.status_code == 200


def test_cursor_bigint_overflow(client):
    """Test that cursor values exceeding BigInteger range are rejected."""
    huge = "99999999999999999999"
    response = client.get(f"/items?cursor={huge}&sort_column=big_id")
    assert response.status_code == 400
    data = response.json()
    assert "BIGINT" in data["detail"]


def test_cursor_smallint_valid(client):
    """Test that valid SmallInteger cursor values are accepted."""
    response = client.get("/items?cursor=100&sort_column=small_id")
    assert response.status_code == 200


def test_cursor_smallint_overflow(client):
    """Test that cursor values exceeding SmallInteger range are rejected."""
    response = client.get("/items?cursor=40000&sort_column=small_id")
    assert response.status_code == 400
    data = response.json()
    assert "SMALLINT" in data["detail"]
