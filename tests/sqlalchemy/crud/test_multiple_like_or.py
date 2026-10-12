import pytest
from fastcrud import FastCRUD


@pytest.mark.asyncio
async def test_or_filter_with_multiple_like_values(async_session, test_model):
    """Test OR filter with multiple values for the same operator (like)"""
    # Create test data
    test_data = [
        {"name": "Alice Johnson", "tier_id": 1, "category_id": 1},
        {"name": "Bob Smith", "tier_id": 2, "category_id": 1},
        {"name": "Charlie Brown", "tier_id": 3, "category_id": 2},
        {"name": "Frank Miller", "tier_id": 4, "category_id": 2},
        {"name": "Alice Cooper", "tier_id": 5, "category_id": 1},
        {"name": "Frank Sinatra", "tier_id": 6, "category_id": 2},
    ]

    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Test with multiple like patterns using list syntax
    result = await crud.get_multi(
        async_session, name__or={"like": ["Alice%", "Frank%"]}
    )

    assert (
        len(result["data"]) == 4
    )  # Alice Johnson, Alice Cooper, Frank Miller, Frank Sinatra
    names = [item["name"] for item in result["data"]]
    assert all(name.startswith("Alice") or name.startswith("Frank") for name in names)

    # Test with case-insensitive multiple patterns
    result = await crud.get_multi(
        async_session, name__or={"ilike": ["%cooper", "%sinatra"]}
    )

    assert len(result["data"]) == 2
    names = [item["name"] for item in result["data"]]
    assert "Alice Cooper" in names
    assert "Frank Sinatra" in names


@pytest.mark.asyncio
async def test_or_filter_with_multiple_operators_including_lists(
    async_session, test_model
):
    """Test OR filter mixing single values and lists"""
    test_data = [
        {"name": "Alice Johnson", "tier_id": 1, "category_id": 1},
        {"name": "Bob Smith", "tier_id": 10, "category_id": 1},
        {"name": "Charlie Brown", "tier_id": 3, "category_id": 2},
        {"name": "Frank Miller", "tier_id": 4, "category_id": 2},
    ]

    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Mix list and single value in OR condition
    result = await crud.get_multi(
        async_session, name__or={"like": ["Alice%", "Frank%"], "startswith": "Bob"}
    )

    assert len(result["data"]) == 3
    names = [item["name"] for item in result["data"]]
    assert "Alice Johnson" in names
    assert "Bob Smith" in names
    assert "Frank Miller" in names


@pytest.mark.asyncio
async def test_not_filter_with_multiple_like_values(async_session, test_model):
    """Test NOT filter with multiple values for the same operator"""
    test_data = [
        {"name": "Alice Johnson", "tier_id": 1, "category_id": 1},
        {"name": "Bob Smith", "tier_id": 2, "category_id": 1},
        {"name": "Charlie Brown", "tier_id": 3, "category_id": 2},
        {"name": "David Jones", "tier_id": 4, "category_id": 2},
    ]

    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Test NOT filter with multiple like patterns
    result = await crud.get_multi(async_session, name__not={"like": ["Alice%", "Bob%"]})

    assert len(result["data"]) == 2
    names = [item["name"] for item in result["data"]]
    assert "Charlie Brown" in names
    assert "David Jones" in names
    assert "Alice Johnson" not in names
    assert "Bob Smith" not in names


@pytest.mark.asyncio
async def test_or_filter_with_contains_operator_list(async_session, test_model):
    """Test OR filter with multiple contains values"""
    test_data = [
        {"name": "Alice Johnson", "tier_id": 1, "category_id": 1},
        {"name": "Bob Smith", "tier_id": 2, "category_id": 1},
        {"name": "Charlie Brown", "tier_id": 3, "category_id": 2},
        {"name": "David Jones", "tier_id": 4, "category_id": 2},
    ]

    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Test with multiple contains patterns
    result = await crud.get_multi(
        async_session, name__or={"contains": ["John", "Smith", "Jones"]}
    )

    assert len(result["data"]) == 3
    names = [item["name"] for item in result["data"]]
    assert "Alice Johnson" in names  # Contains "John"
    assert "Bob Smith" in names  # Contains "Smith"
    assert "David Jones" in names  # Contains "Jones"
    assert "Charlie Brown" not in names


@pytest.mark.asyncio
async def test_combined_filters_with_list_or(async_session, test_model):
    """Test combining regular filters with list-based OR filters"""
    test_data = [
        {"name": "Alice Johnson", "tier_id": 1, "category_id": 1},
        {"name": "Alice Cooper", "tier_id": 2, "category_id": 2},
        {"name": "Bob Smith", "tier_id": 3, "category_id": 1},
        {"name": "Frank Miller", "tier_id": 4, "category_id": 1},
        {"name": "Frank Sinatra", "tier_id": 5, "category_id": 2},
    ]

    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Combine regular filter with list-based OR filter
    result = await crud.get_multi(
        async_session,
        category_id=1,  # AND condition
        name__or={"like": ["Alice%", "Frank%"]},  # OR condition with list
    )

    assert len(result["data"]) == 2
    for item in result["data"]:
        assert item["category_id"] == 1
        assert item["name"].startswith("Alice") or item["name"].startswith("Frank")

    names = [item["name"] for item in result["data"]]
    assert "Alice Johnson" in names
    assert "Frank Miller" in names
    assert "Alice Cooper" not in names  # category_id=2
    assert "Frank Sinatra" not in names  # category_id=2


@pytest.mark.asyncio
async def test_or_filter_with_in_operator_list(async_session, test_model):
    """Test that 'in' operator still works with its expected list/tuple format"""
    test_data = [
        {"name": "Alice", "tier_id": 1, "category_id": 1},
        {"name": "Bob", "tier_id": 2, "category_id": 1},
        {"name": "Charlie", "tier_id": 3, "category_id": 2},
        {"name": "David", "tier_id": 4, "category_id": 2},
    ]

    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)

    # Regular 'in' filter should still work
    result = await crud.get_multi(async_session, name__in=["Alice", "Bob"])

    assert len(result["data"]) == 2
    names = [item["name"] for item in result["data"]]
    assert "Alice" in names
    assert "Bob" in names

    # Test OR with 'in' operator (though 'in' already handles multiple values)
    result = await crud.get_multi(
        async_session,
        tier_id__or={
            "in": [[1, 2], [3, 4]]
        },  # List of lists for multiple 'in' conditions
    )

    # This should match all tier_ids (1,2,3,4)
    assert len(result["data"]) == 4


@pytest.mark.asyncio
async def test_or_and_not_filters_with_flat_collection_values(
    async_session, test_model
):
    """A flat list is the value of in, not_in and between inside OR and NOT filters"""
    for tier_id, name in [(1, "Alice"), (2, "Bob"), (3, "Charlie"), (4, "David")]:
        async_session.add(test_model(name=name, tier_id=tier_id, category_id=1))
    await async_session.commit()

    crud = FastCRUD(test_model)

    async def tiers(**filters):
        result = await crud.get_multi(async_session, sort_columns="tier_id", **filters)
        return [item["tier_id"] for item in result["data"]]

    assert await tiers(tier_id__or={"in": [1, 4]}) == [1, 4]
    assert await tiers(tier_id__or={"between": [2, 3]}) == [2, 3]
    assert await tiers(tier_id__or={"in": [1], "gt": 3}) == [1, 4]
    assert await tiers(tier_id__not={"in": [1, 4]}) == [2, 3]
    assert await tiers(tier_id__not={"between": [2, 3]}) == [1, 4]
    assert await tiers(tier_id__not={"not_in": [2, 3]}) == [2, 3]

    # lists of collections still mean several collections
    assert await tiers(tier_id__or={"between": [[1, 1], [4, 4]]}) == [1, 4]
    assert await tiers(tier_id__not={"in": [[1], [4]]}) == [2, 3]
