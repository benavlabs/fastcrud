"""Tests for custom filter functionality."""

import pytest
from fastcrud import FastCRUD, FilterCallable
from fastcrud.core.filtering.operators import get_sqlalchemy_filter
from sqlalchemy import Column


@pytest.mark.asyncio
async def test_custom_filter_new_operator(async_session, test_model, test_data):
    """Test adding a new custom filter operator."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    # Define a custom "double_gt" operator that checks if column > value * 2
    def double_gt(col: Column) -> FilterCallable:
        def filter_fn(value):
            return col > value * 2

        return filter_fn

    crud = FastCRUD(test_model, custom_filters={"double_gt": double_gt})

    # tier_id > 0.5 * 2 = tier_id > 1, so only tier_id=2 records
    result = await crud.get_multi(async_session, tier_id__double_gt=0.5)

    assert all(item["tier_id"] == 2 for item in result["data"])


@pytest.mark.asyncio
async def test_custom_filter_override_builtin(async_session, test_model, test_data):
    """Test overriding a built-in filter operator."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    # Override "gt" to be "greater than or equal" instead of "greater than"
    def custom_gt(col: Column) -> FilterCallable:
        def filter_fn(value):
            return col >= value

        return filter_fn

    crud = FastCRUD(test_model, custom_filters={"gt": custom_gt})

    # With custom gt (>=), tier_id__gt=2 should include tier_id=2
    result = await crud.get_multi(async_session, tier_id__gt=2)

    assert all(item["tier_id"] == 2 for item in result["data"])
    assert len(result["data"]) > 0


@pytest.mark.asyncio
async def test_custom_filter_without_custom_uses_builtin(
    async_session, test_model, test_data
):
    """Test that without custom filters, built-in operators work normally."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Built-in gt (>), tier_id__gt=2 should return nothing (no tier_id > 2)
    result = await crud.get_multi(async_session, tier_id__gt=2)

    assert len(result["data"]) == 0


@pytest.mark.asyncio
async def test_custom_filter_combined_with_builtin(
    async_session, test_model, test_data
):
    """Test custom filter used alongside built-in filters."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    # Custom filter that checks if value is even
    def is_even(col: Column) -> FilterCallable:
        def filter_fn(value):
            if value:
                return col % 2 == 0
            return col % 2 != 0

        return filter_fn

    crud = FastCRUD(test_model, custom_filters={"is_even": is_even})

    # Get records where tier_id is even (tier_id=2) AND name starts with specific letter
    result = await crud.get_multi(
        async_session,
        tier_id__is_even=True,
        name__startswith="A",
    )

    assert all(item["tier_id"] == 2 for item in result["data"])
    assert all(item["name"].startswith("A") for item in result["data"])

    # Test the False branch - get records where tier_id is odd (tier_id=1)
    result_odd = await crud.get_multi(
        async_session,
        tier_id__is_even=False,
    )

    assert all(item["tier_id"] == 1 for item in result_odd["data"])


@pytest.mark.asyncio
async def test_custom_filter_on_string_column(async_session, test_model, test_data):
    """Test custom filter on string columns."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    # Custom filter that checks string length
    def length_gt(col: Column) -> FilterCallable:
        def filter_fn(value):
            from sqlalchemy import func

            return func.length(col) > value

        return filter_fn

    crud = FastCRUD(test_model, custom_filters={"length_gt": length_gt})

    # Get records where name length > 5
    result = await crud.get_multi(async_session, name__length_gt=5)

    assert all(len(item["name"]) > 5 for item in result["data"])


@pytest.mark.asyncio
async def test_custom_filter_preserves_instance_isolation(
    async_session, test_model, test_data
):
    """Test that custom filters on one instance don't affect other instances."""
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    def custom_op(col: Column) -> FilterCallable:
        def filter_fn(value):
            return col == value

        return filter_fn

    crud_with_custom = FastCRUD(test_model, custom_filters={"custom_op": custom_op})
    crud_without_custom = FastCRUD(test_model)

    # Custom filter should work on the instance that has it
    result = await crud_with_custom.get_multi(async_session, tier_id__custom_op=1)
    assert all(item["tier_id"] == 1 for item in result["data"])

    # Instance without custom filter should raise an error for unknown operator
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        await crud_without_custom.get_multi(async_session, tier_id__custom_op=1)


def test_get_sqlalchemy_filter_in_operator_requires_list():
    """Test that 'in' operator requires a list/tuple/set value."""
    with pytest.raises(ValueError, match="<in> filter must be tuple, list or set"):
        get_sqlalchemy_filter("in", "not_a_list")


def test_get_sqlalchemy_filter_not_in_operator_requires_list():
    """Test that 'not_in' operator requires a list/tuple/set value."""
    with pytest.raises(ValueError, match="<not_in> filter must be tuple, list or set"):
        get_sqlalchemy_filter("not_in", 123)


def test_get_sqlalchemy_filter_between_operator_requires_list():
    """Test that 'between' operator requires a list/tuple/set value."""
    with pytest.raises(ValueError, match="<between> filter must be tuple, list or set"):
        get_sqlalchemy_filter("between", "invalid")


def test_get_sqlalchemy_filter_between_requires_two_values():
    """Test that 'between' operator requires exactly 2 values."""
    with pytest.raises(ValueError, match="Between operator requires exactly 2 values"):
        get_sqlalchemy_filter("between", [1, 2, 3])

    with pytest.raises(ValueError, match="Between operator requires exactly 2 values"):
        get_sqlalchemy_filter("between", [1])
