"""
Tests for nested_limit functionality in one-to-many relationships.

This module tests the SQL-level limiting feature using window functions
when fetching one-to-many related records via get_multi_joined.
"""

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import aliased

from fastcrud import FastCRUD, JoinConfig
from fastcrud.core.query import fetch_one_to_many_with_limit
from ..conftest import (
    Article,
    Author,
    ContractPoly,
    ProjectPoly,
    Story,
    TreeNode,
    Writer,
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


@pytest.mark.asyncio
async def test_nested_limit_filters_do_not_change_the_total_count(async_session):
    async_session.add_all(
        [
            Author(id=1, name="Author 1"),
            Author(id=2, name="Author 2"),
            Author(id=3, name="Author 3"),
            Article(
                id=1,
                title="Kept",
                content="C1",
                author_id=1,
                published_date="2023-01-01",
            ),
            Article(
                id=2,
                title="Dropped",
                content="C2",
                author_id=2,
                published_date="2023-01-02",
            ),
        ]
    )
    await async_session.commit()

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[
            JoinConfig(
                model=Article,
                join_on=Author.id == Article.author_id,
                join_prefix="articles_",
                relationship_type="one-to-many",
                nested_limit=5,
                filters={"published_date": "2023-01-01"},
            )
        ],
        nest_joins=True,
    )

    articles_by_author = {
        author["id"]: len(author["articles"]) for author in result["data"]
    }
    assert articles_by_author == {1: 1, 2: 0, 3: 0}
    assert result["total_count"] == 3


async def _two_authors_with_articles(async_session):
    async_session.add_all(
        [
            Author(id=1, name="Author 1"),
            Author(id=2, name="Author 2"),
            Article(
                id=1,
                title="Kept",
                content="C1",
                author_id=1,
                published_date="2023-01-01",
            ),
            Article(
                id=2,
                title="Dropped",
                content="C2",
                author_id=1,
                published_date="2023-01-02",
            ),
            Article(
                id=3,
                title="Dropped",
                content="C3",
                author_id=2,
                published_date="2023-01-03",
            ),
        ]
    )
    await async_session.commit()


def _articles_join(join_on):
    return JoinConfig(
        model=Article,
        join_on=join_on,
        join_prefix="articles_",
        relationship_type="one-to-many",
        nested_limit=5,
    )


@pytest.mark.asyncio
async def test_nested_limit_applies_the_join_on_conditions_on_the_related_model(
    async_session,
):
    await _two_authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[
            _articles_join((Author.id == Article.author_id) & (Article.title == "Kept"))
        ],
        nest_joins=True,
    )

    titles_by_author = {
        author["id"]: [article["title"] for article in author["articles"]]
        for author in result["data"]
    }
    assert titles_by_author == {1: ["Kept"], 2: []}
    assert result["total_count"] == 2


@pytest.mark.parametrize(
    ("join_on", "message"),
    [
        (
            (Author.id == Article.author_id) & (Author.name == "Author 1"),
            "can only add conditions on Article, not on Author",
        ),
        (
            Article.title == "Kept",
            "Cannot find the equality between Author and Article",
        ),
    ],
    ids=["condition-on-the-parent", "no-link-between-the-models"],
)
@pytest.mark.asyncio
async def test_nested_limit_refuses_a_join_on_it_cannot_apply(
    async_session, join_on, message
):
    await _two_authors_with_articles(async_session)

    with pytest.raises(ValueError, match=message):
        await FastCRUD(Author).get_multi_joined(
            db=async_session, joins_config=[_articles_join(join_on)], nest_joins=True
        )


@pytest.mark.asyncio
async def test_a_where_naming_a_model_fetched_separately_is_refused(async_session):
    await _two_authors_with_articles(async_session)

    with pytest.raises(ValueError, match="side by side without joining them"):
        await FastCRUD(Author).get_multi_joined(
            db=async_session,
            joins_config=[_articles_join(Author.id == Article.author_id)],
            nest_joins=True,
            where=Article.title == "Kept",
        )


@pytest.mark.asyncio
async def test_nested_limit_accepts_a_subquery_naming_the_parent(async_session):
    await _two_authors_with_articles(async_session)
    first_author = select(Author.id).where(Author.name == "Author 1")

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[
            _articles_join(
                (Author.id == Article.author_id) & Article.author_id.in_(first_author)
            )
        ],
        nest_joins=True,
    )

    titles_by_author = {
        author["id"]: [article["title"] for article in author["articles"]]
        for author in result["data"]
    }
    assert titles_by_author == {1: ["Kept", "Dropped"], 2: []}


@pytest.mark.asyncio
async def test_nested_limit_links_the_models_by_the_foreign_key(async_session):
    await _two_authors_with_articles(async_session)

    with pytest.raises(
        ValueError, match="not on Author: articles.title = authors.name$"
    ):
        await FastCRUD(Author).get_multi_joined(
            db=async_session,
            joins_config=[
                _articles_join(
                    (Article.title == Author.name) & (Author.id == Article.author_id)
                )
            ],
            nest_joins=True,
        )


@pytest.mark.asyncio
async def test_nested_limit_through_an_alias_returns_each_child_once(async_session):
    await _two_authors_with_articles(async_session)
    article = aliased(Article, name="article_alias")

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[
            JoinConfig(
                model=Article,
                alias=article,
                join_on=Author.id == article.author_id,
                join_prefix="articles_",
                relationship_type="one-to-many",
                nested_limit=5,
            )
        ],
        nest_joins=True,
    )

    ids_by_author = {
        author["id"]: [item["id"] for item in author["articles"]]
        for author in result["data"]
    }
    assert ids_by_author == {1: [1, 2], 2: [3]}


async def _a_tree(async_session):
    async_session.add_all(
        [
            TreeNode(id=1, name="root", parent_id=None),
            TreeNode(id=2, name="left", parent_id=1),
            TreeNode(id=3, name="right", parent_id=1),
            TreeNode(id=4, name="leaf", parent_id=2),
        ]
    )
    await async_session.commit()


@pytest.mark.asyncio
async def test_nested_limit_on_a_self_referential_join(async_session):
    await _a_tree(async_session)
    child = aliased(TreeNode, name="child")

    result = await FastCRUD(TreeNode).get_multi_joined(
        db=async_session,
        joins_config=[
            JoinConfig(
                model=TreeNode,
                alias=child,
                join_on=TreeNode.id == child.parent_id,
                join_prefix="children_",
                relationship_type="one-to-many",
                nested_limit=5,
            )
        ],
        nest_joins=True,
    )

    children_by_node = {
        node["id"]: [item["id"] for item in node["children"]] for node in result["data"]
    }
    assert children_by_node == {1: [2, 3], 2: [4], 3: [], 4: []}


@pytest.mark.asyncio
async def test_nested_limit_on_a_self_referential_join_needs_an_alias(async_session):
    await _a_tree(async_session)

    with pytest.raises(ValueError, match="TreeNode to itself needs an alias"):
        await FastCRUD(TreeNode).get_multi_joined(
            db=async_session,
            joins_config=[
                JoinConfig(
                    model=TreeNode,
                    join_on=TreeNode.id == TreeNode.parent_id,
                    join_prefix="children_",
                    relationship_type="one-to-many",
                    nested_limit=5,
                )
            ],
            nest_joins=True,
        )


async def _contracts_with_projects(async_session):
    async_session.add_all(
        [
            ContractPoly(id=1),
            ContractPoly(id=2),
            ProjectPoly(id=3, name="p3", contract_id=1),
            ProjectPoly(id=4, name="p4", contract_id=1),
            ProjectPoly(id=5, name="p5", contract_id=2),
        ]
    )
    await async_session.commit()


def _projects_by_contract(result):
    return {
        contract["id"]: [project["id"] for project in contract["projects"]]
        for contract in result["data"]
    }


@pytest.mark.parametrize(
    "alias",
    [
        None,
        aliased(ProjectPoly, flat=True),
        aliased(ProjectPoly, name="project"),
    ],
    ids=["no-alias", "flat-alias", "named-alias"],
)
@pytest.mark.asyncio
async def test_nested_limit_between_inheritance_siblings(async_session, alias):
    await _contracts_with_projects(async_session)
    project = alias if alias is not None else ProjectPoly

    result = await FastCRUD(ContractPoly).get_multi_joined(
        db=async_session,
        joins_config=[
            JoinConfig(
                model=ProjectPoly,
                alias=alias,
                join_on=ContractPoly.id == project.contract_id,
                join_prefix="projects_",
                relationship_type="one-to-many",
                nested_limit=5,
            )
        ],
        nest_joins=True,
    )

    assert _projects_by_contract(result) == {1: [3, 4], 2: [5]}


class ArticleTitleSchema(BaseModel):
    id: int
    title: str


@pytest.mark.asyncio
async def test_nested_limit_with_a_schema_leaving_out_the_foreign_key(async_session):
    await _two_authors_with_articles(async_session)
    join = _articles_join(Author.id == Article.author_id)
    join.schema_to_select = ArticleTitleSchema

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session, joins_config=[join], nest_joins=True
    )

    articles_by_author = {author["id"]: author["articles"] for author in result["data"]}
    assert articles_by_author == {
        1: [{"id": 1, "title": "Kept"}, {"id": 2, "title": "Dropped"}],
        2: [{"id": 3, "title": "Dropped"}],
    }


def _stories_join():
    return JoinConfig(
        model=Story,
        join_on=Writer.code == Story.writer_code,
        join_prefix="stories_",
        relationship_type="one-to-many",
        nested_limit=5,
    )


@pytest.mark.parametrize(
    ("writers", "stories", "expected"),
    [
        ([(1, 2), (2, 1)], [(10, 2), (20, 1)], {1: [10], 2: [20]}),
        ([(1, 7), (2, 7), (3, 8)], [(10, 7), (20, 8)], {1: [10], 2: [10], 3: [20]}),
    ],
    ids=["codes-unlike-the-ids", "writers-sharing-a-code"],
)
@pytest.mark.asyncio
async def test_nested_limit_links_parents_by_the_column_join_on_names(
    async_session, writers, stories, expected
):
    async_session.add_all(
        [Writer(id=id, code=code) for id, code in writers]
        + [Story(id=id, writer_code=code) for id, code in stories]
    )
    await async_session.commit()

    result = await FastCRUD(Writer).get_multi_joined(
        db=async_session, joins_config=[_stories_join()], nest_joins=True
    )

    stories_by_writer = {
        writer["id"]: [story["id"] for story in writer["stories"]]
        for writer in result["data"]
    }
    assert stories_by_writer == expected


class WriterIdSchema(BaseModel):
    id: int


@pytest.mark.asyncio
async def test_nested_limit_needs_the_column_join_on_names_selected(async_session):
    async_session.add_all([Writer(id=1, code=2), Story(id=10, writer_code=2)])
    await async_session.commit()

    with pytest.raises(ValueError, match="links parents by their 'code'"):
        await FastCRUD(Writer).get_multi_joined(
            db=async_session,
            schema_to_select=WriterIdSchema,
            joins_config=[_stories_join()],
            nest_joins=True,
        )


@pytest.mark.asyncio
async def test_the_separate_query_refuses_a_model_it_does_not_read(async_session):
    await _two_authors_with_articles(async_session)
    article = aliased(Article, name="article_alias")

    with pytest.raises(ValueError, match="side by side without joining them"):
        await fetch_one_to_many_with_limit(
            db=async_session,
            related_model=Article,
            fk_column=article.author_id,
            parent_ids=[1, 2],
            limit_per_parent=5,
        )
