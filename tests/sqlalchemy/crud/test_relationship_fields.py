"""
Test for GitHub issue #199: get_multi returns boolean values for pydantic relationship fields
when a select schema is provided.

This test reproduces the issue where including relationship fields in a Pydantic schema
causes get_multi to execute a cartesian product between tables, resulting in boolean values
instead of the expected data.
"""

import pytest
from typing import Optional
from pydantic import BaseModel, ConfigDict
from fastcrud.crud.fast_crud import FastCRUD
from ...sqlalchemy.conftest import ModelTest, TierModel


class TierReadSchemaIssue199(BaseModel):
    """Schema for tier data in issue #199 tests."""

    name: str


class ReadSchemaWithRelationship(BaseModel):
    """Schema that includes a relationship field to test issue #199 fix."""

    model_config = ConfigDict(extra="forbid")
    name: str
    tier_id: int
    tier: TierReadSchemaIssue199 | None = None


@pytest.fixture(scope="function")
def issue_199_test_data() -> list[dict]:
    """Test data specific to issue #199 reproduction."""
    return [
        {"id": 1, "name": "Charlie", "tier_id": 1},
        {"id": 2, "name": "Alice", "tier_id": 2},
        {"id": 3, "name": "Bob", "tier_id": 1},
        {"id": 4, "name": "David", "tier_id": 2},
        {"id": 5, "name": "Eve", "tier_id": 1},
        {"id": 6, "name": "Frank", "tier_id": 2},
        {"id": 7, "name": "Grace", "tier_id": 1},
        {"id": 8, "name": "Hannah", "tier_id": 2},
        {"id": 9, "name": "Ivan", "tier_id": 1},
        {"id": 10, "name": "Judy", "tier_id": 2},
        {"id": 11, "name": "Alice", "tier_id": 1},
    ]


@pytest.fixture(scope="function")
def issue_199_tier_data() -> list[dict]:
    """Tier data specific to issue #199 reproduction."""
    return [{"id": 1, "name": "Premium"}, {"id": 2, "name": "Basic"}]


@pytest.mark.asyncio
async def test_get_multi_excludes_relationship_fields_from_select(
    async_session, issue_199_test_data, issue_199_tier_data
):
    """
    Test that reproduces issue #199: get_multi should not include relationship fields
    in the SELECT statement when they are present in the schema_to_select.

    This test verifies that:
    1. No cartesian product is created
    2. No boolean values are returned for relationship fields
    3. The relationship field is simply excluded from the result
    """
    # Setup test data
    for tier_item in issue_199_tier_data:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in issue_199_test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    # This should NOT create a cartesian product or return boolean values
    result = await crud.get_multi(
        db=async_session, schema_to_select=ReadSchemaWithRelationship
    )

    # Verify the result structure
    assert "data" in result
    assert "total_count" in result
    assert result["total_count"] == len(issue_199_test_data)

    # Verify no cartesian product was created (should have 11 records, not 22)
    assert len(result["data"]) == len(issue_199_test_data)

    # Verify that each record has the expected structure
    for item in result["data"]:
        assert "name" in item
        assert "tier_id" in item
        # The relationship field should either be excluded or None, but NOT a boolean
        if "tier" in item:
            assert item["tier"] is None or isinstance(item["tier"], dict)
            assert not isinstance(item["tier"], bool)

    # Verify specific data integrity
    names_in_result = [item["name"] for item in result["data"]]
    expected_names = [item["name"] for item in issue_199_test_data]
    assert sorted(names_in_result) == sorted(expected_names)


@pytest.mark.asyncio
async def test_get_multi_return_as_model_with_relationship_fields(
    async_session, issue_199_test_data, issue_199_tier_data
):
    """
    Test that get_multi with return_as_model=True works correctly when the schema
    contains relationship fields.
    """
    # Setup test data
    for tier_item in issue_199_tier_data:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in issue_199_test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    # This should work without validation errors
    result = await crud.get_multi(
        db=async_session,
        schema_to_select=ReadSchemaWithRelationship,
        return_as_model=True,
    )

    # Verify the result structure
    assert "data" in result
    assert "total_count" in result
    assert result["total_count"] == len(issue_199_test_data)
    assert len(result["data"]) == len(issue_199_test_data)

    # Verify that all items are instances of the schema
    for item in result["data"]:
        assert isinstance(item, ReadSchemaWithRelationship)
        assert hasattr(item, "name")
        assert hasattr(item, "tier_id")
        # The tier field should be None since it's not properly joined
        assert item.tier is None


@pytest.mark.asyncio
async def test_get_joined_functionality_unaffected_by_fix(
    async_session, issue_199_test_data, issue_199_tier_data
):
    """
    Test that get_joined still works correctly and can properly populate relationship fields.
    This ensures our fix doesn't break the intended functionality for joins.
    """
    # Setup test data
    for tier_item in issue_199_tier_data:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in issue_199_test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    # This should work correctly with proper joins
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=ReadSchemaWithRelationship,
        join_schema_to_select=TierReadSchemaIssue199,
        nest_joins=True,
    )

    # Verify the result has the tier information properly nested
    assert "name" in result
    assert "tier_id" in result
    assert "tier" in result
    assert isinstance(result["tier"], dict)
    assert "name" in result["tier"]


@pytest.mark.asyncio
async def test_relationship_field_exclusion_prevents_cartesian_product(
    async_session, issue_199_test_data, issue_199_tier_data
):
    """
    Test that specifically verifies the fix prevents cartesian products
    by checking the generated SQL doesn't include relationship tables.
    """
    # Setup test data
    for tier_item in issue_199_tier_data:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in issue_199_test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    # Generate the select statement to verify it doesn't include tier table
    stmt = await crud.select(schema_to_select=ReadSchemaWithRelationship)

    # Convert statement to string to check it doesn't contain tier table
    stmt_str = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # The statement should only select from the test table, not include tier
    assert "FROM test" in stmt_str
    assert "FROM test, tier" not in stmt_str  # No cartesian product
    assert "tier.id = test.tier_id" not in stmt_str  # No relationship field in SELECT
