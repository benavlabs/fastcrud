"""
Tests for nested_limit functionality in one-to-many relationships.

This module tests the SQL-level limiting feature using window functions
when fetching one-to-many related records via get_multi_joined.
"""

import pytest
from pydantic import BaseModel

from fastcrud import FastCRUD, JoinConfig
from ..conftest import (
    Article,
    Author,
)


class ArticleSchema(BaseModel):
    """Schema for selecting specific article fields."""

    id: int
    title: str
    author_id: int


@pytest.mark.asyncio
async def test_nested_limit_basic(async_session):
    """Test that nested_limit restricts the number of related records returned."""
    # Create test data: 2 authors with 5 articles each
    author1 = Author(id=1, name="Author 1")
    author2 = Author(id=2, name="Author 2")

    articles = []
    for i in range(1, 6):
        articles.append(
            Article(
                id=i,
                title=f"Article {i} by Author 1",
                content=f"Content {i}",
                author_id=1,
                published_date=f"2023-01-0{i}",
            )
        )
    for i in range(6, 11):
        articles.append(
            Article(
                id=i,
                title=f"Article {i} by Author 2",
                content=f"Content {i}",
                author_id=2,
                published_date=f"2023-02-{i-5:02d}",
            )
        )

    async_session.add_all([author1, author2] + articles)
    await async_session.commit()

    # Query with nested_limit=3
    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=3,
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    # Verify each author has at most 3 articles
    for author_data in result["data"]:
        assert len(author_data["articles"]) <= 3


@pytest.mark.asyncio
async def test_nested_limit_with_sorting(async_session):
    """Test nested_limit with custom sort order."""
    author1 = Author(id=1, name="Author 1")

    articles = [
        Article(
            id=1,
            title="Z Article",
            content="C1",
            author_id=1,
            published_date="2023-01-01",
        ),
        Article(
            id=2,
            title="A Article",
            content="C2",
            author_id=1,
            published_date="2023-01-02",
        ),
        Article(
            id=3,
            title="M Article",
            content="C3",
            author_id=1,
            published_date="2023-01-03",
        ),
        Article(
            id=4,
            title="B Article",
            content="C4",
            author_id=1,
            published_date="2023-01-04",
        ),
        Article(
            id=5,
            title="Y Article",
            content="C5",
            author_id=1,
            published_date="2023-01-05",
        ),
    ]

    async_session.add_all([author1] + articles)
    await async_session.commit()

    # Query with nested_limit=3, sorted by title ascending
    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=3,
            sort_columns="title",
            sort_orders="asc",
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    author_data = result["data"][0]
    # Should get first 3 alphabetically: A, B, M
    titles = [a["title"] for a in author_data["articles"]]
    assert titles == ["A Article", "B Article", "M Article"]


@pytest.mark.asyncio
async def test_nested_limit_with_descending_sort(async_session):
    """Test nested_limit with descending sort order."""
    author1 = Author(id=1, name="Author 1")

    articles = [
        Article(
            id=1,
            title="A Article",
            content="C1",
            author_id=1,
            published_date="2023-01-01",
        ),
        Article(
            id=2,
            title="B Article",
            content="C2",
            author_id=1,
            published_date="2023-01-02",
        ),
        Article(
            id=3,
            title="C Article",
            content="C3",
            author_id=1,
            published_date="2023-01-03",
        ),
        Article(
            id=4,
            title="D Article",
            content="C4",
            author_id=1,
            published_date="2023-01-04",
        ),
        Article(
            id=5,
            title="E Article",
            content="C5",
            author_id=1,
            published_date="2023-01-05",
        ),
    ]

    async_session.add_all([author1] + articles)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=2,
            sort_columns="title",
            sort_orders="desc",
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    author_data = result["data"][0]
    titles = [a["title"] for a in author_data["articles"]]
    # Should get last 2 alphabetically: E, D
    assert titles == ["E Article", "D Article"]


@pytest.mark.asyncio
async def test_nested_limit_with_multiple_sort_columns(async_session):
    """Test nested_limit with multiple sort columns."""
    author1 = Author(id=1, name="Author 1")

    articles = [
        Article(
            id=1, title="Same", content="C1", author_id=1, published_date="2023-01-03"
        ),
        Article(
            id=2, title="Same", content="C2", author_id=1, published_date="2023-01-01"
        ),
        Article(
            id=3, title="Same", content="C3", author_id=1, published_date="2023-01-02"
        ),
        Article(
            id=4,
            title="Different",
            content="C4",
            author_id=1,
            published_date="2023-01-04",
        ),
    ]

    async_session.add_all([author1] + articles)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=3,
            sort_columns=["title", "published_date"],
            sort_orders=["asc", "asc"],
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    author_data = result["data"][0]
    # Should be sorted by title first, then by published_date
    # "Different" comes before "Same", then "Same" sorted by date
    assert len(author_data["articles"]) == 3
    assert author_data["articles"][0]["title"] == "Different"


@pytest.mark.asyncio
async def test_nested_limit_with_schema_to_select(async_session):
    """Test nested_limit with a schema to select specific columns."""
    author1 = Author(id=1, name="Author 1")

    articles = [
        Article(
            id=1,
            title="Article 1",
            content="Content 1",
            author_id=1,
            published_date="2023-01-01",
        ),
        Article(
            id=2,
            title="Article 2",
            content="Content 2",
            author_id=1,
            published_date="2023-01-02",
        ),
        Article(
            id=3,
            title="Article 3",
            content="Content 3",
            author_id=1,
            published_date="2023-01-03",
        ),
    ]

    async_session.add_all([author1] + articles)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=2,
            schema_to_select=ArticleSchema,
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    author_data = result["data"][0]
    assert len(author_data["articles"]) == 2
    # Verify only schema fields are present
    for article in author_data["articles"]:
        assert "id" in article
        assert "title" in article
        assert "author_id" in article
        # content and published_date should not be in the result
        assert "content" not in article
        assert "published_date" not in article


@pytest.mark.asyncio
async def test_nested_limit_with_filters(async_session):
    """Test nested_limit with additional filters on the related model."""
    author1 = Author(id=1, name="Author 1")

    articles = [
        Article(
            id=1,
            title="Published 1",
            content="Content 1",
            author_id=1,
            published_date="2023-01-01",
        ),
        Article(
            id=2,
            title="Draft 1",
            content="Content 2",
            author_id=1,
            published_date="2023-01-02",
        ),
        Article(
            id=3,
            title="Published 2",
            content="Content 3",
            author_id=1,
            published_date="2023-01-03",
        ),
        Article(
            id=4,
            title="Published 3",
            content="Content 4",
            author_id=1,
            published_date="2023-01-04",
        ),
        Article(
            id=5,
            title="Draft 2",
            content="Content 5",
            author_id=1,
            published_date="2023-01-05",
        ),
    ]

    async_session.add_all([author1] + articles)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=2,
            filters={"published_date": "2023-01-01"},
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    author_data = result["data"][0]
    # Only articles with published_date="2023-01-01" should be returned
    assert len(author_data["articles"]) == 1
    assert author_data["articles"][0]["published_date"] == "2023-01-01"


@pytest.mark.asyncio
async def test_nested_limit_empty_parent_ids(async_session):
    """Test that empty parent results return empty nested data."""
    # Create only an author with no articles
    author1 = Author(id=1, name="Author 1")
    async_session.add(author1)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=5,
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    # Author should exist but have empty articles list
    assert len(result["data"]) == 1
    assert result["data"][0]["articles"] == []


@pytest.mark.asyncio
async def test_nested_limit_multiple_parents(async_session):
    """Test nested_limit correctly limits per parent, not globally."""
    # Create 3 authors with varying number of articles
    author1 = Author(id=1, name="Author 1")
    author2 = Author(id=2, name="Author 2")
    author3 = Author(id=3, name="Author 3")

    articles = []
    # Author 1: 5 articles
    for i in range(1, 6):
        articles.append(
            Article(
                id=i,
                title=f"A1-Art{i}",
                content=f"C{i}",
                author_id=1,
                published_date=f"2023-01-0{i}",
            )
        )
    # Author 2: 3 articles
    for i in range(6, 9):
        articles.append(
            Article(
                id=i,
                title=f"A2-Art{i}",
                content=f"C{i}",
                author_id=2,
                published_date=f"2023-02-0{i-5}",
            )
        )
    # Author 3: 1 article
    articles.append(
        Article(
            id=9,
            title="A3-Art9",
            content="C9",
            author_id=3,
            published_date="2023-03-01",
        )
    )

    async_session.add_all([author1, author2, author3] + articles)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=2,
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    # Verify each author has correct number of articles (capped at 2)
    results_by_id = {r["id"]: r for r in result["data"]}
    assert len(results_by_id[1]["articles"]) == 2  # Had 5, limited to 2
    assert len(results_by_id[2]["articles"]) == 2  # Had 3, limited to 2
    assert len(results_by_id[3]["articles"]) == 1  # Had 1, stays at 1


@pytest.mark.asyncio
async def test_nested_limit_with_join_on_reversed(async_session):
    """Test nested_limit when join_on condition is written in reverse order."""
    author1 = Author(id=1, name="Author 1")

    articles = [
        Article(
            id=1,
            title="Article 1",
            content="C1",
            author_id=1,
            published_date="2023-01-01",
        ),
        Article(
            id=2,
            title="Article 2",
            content="C2",
            author_id=1,
            published_date="2023-01-02",
        ),
        Article(
            id=3,
            title="Article 3",
            content="C3",
            author_id=1,
            published_date="2023-01-03",
        ),
    ]

    async_session.add_all([author1] + articles)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    # Note: join_on is reversed (Article.author_id == Author.id instead of Author.id == Article.author_id)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Article.author_id == Author.id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=2,
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    author_data = result["data"][0]
    assert len(author_data["articles"]) == 2


@pytest.mark.asyncio
async def test_nested_limit_no_matching_related_records(async_session):
    """Test nested_limit when filters exclude all related records."""
    author1 = Author(id=1, name="Author 1")

    articles = [
        Article(
            id=1,
            title="Article 1",
            content="C1",
            author_id=1,
            published_date="2023-01-01",
        ),
        Article(
            id=2,
            title="Article 2",
            content="C2",
            author_id=1,
            published_date="2023-01-02",
        ),
    ]

    async_session.add_all([author1] + articles)
    await async_session.commit()

    author_crud = FastCRUD(Author)
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=5,
            filters={"published_date": "1999-01-01"},  # No articles match this
        )
    ]

    result = await author_crud.get_multi_joined(
        db=async_session, joins_config=joins_config, nest_joins=True
    )

    author_data = result["data"][0]
    assert author_data["articles"] == []
