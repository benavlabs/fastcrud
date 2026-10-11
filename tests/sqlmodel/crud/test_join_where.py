"""``JoinConfig.where`` and ``CountConfig.where``: which joined rows attach, never which parents come back."""

import pytest
from sqlalchemy.orm import aliased

from fastcrud import CountConfig, FastCRUD, JoinConfig
from ..conftest import (
    Article,
    Author,
    Card,
    Participant,
    Project,
    ProjectsParticipantsAssociation,
)

pytestmark = pytest.mark.asyncio

ON_EACH_DATABASE = [
    pytest.param("sqlite", marks=pytest.mark.dialect("sqlite"), id="sqlite"),
    pytest.param(
        "postgresql", marks=pytest.mark.dialect("postgresql"), id="postgresql"
    ),
    pytest.param("mysql", marks=pytest.mark.dialect("mysql"), id="mysql"),
]
WITH_AND_WITHOUT_NESTED_LIMIT = pytest.mark.parametrize(
    "nested_limit", [None, 5], ids=["joined", "nested-limit"]
)


async def _authors_with_articles(async_session):
    async_session.add_all(
        [
            Author(id=1, name="Ann"),
            Author(id=2, name="Bob"),
            Author(id=3, name="Cid"),
            Article(id=1, title="kept", author_id=1),
            Article(id=2, title="dropped", author_id=1),
            Article(id=3, title="dropped", author_id=2),
        ]
    )
    await async_session.commit()


def _articles_join(**overrides):
    return JoinConfig(
        **{
            "model": Article,
            "join_on": Author.id == Article.author_id,
            "join_prefix": "articles_",
            "relationship_type": "one-to-many",
            "where": Article.title == "kept",
            **overrides,
        }
    )


def _author_join(**overrides):
    return JoinConfig(
        **{
            "model": Author,
            "join_on": Article.author_id == Author.id,
            "join_prefix": "author_",
            "where": Author.name == "Ann",
            **overrides,
        }
    )


def _article_ids_by_author(result):
    return {
        author["id"]: sorted(article["id"] for article in author["articles"])
        for author in result["data"]
    }


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
@WITH_AND_WITHOUT_NESTED_LIMIT
async def test_where_attaches_only_matching_rows_and_keeps_every_parent(
    async_session, dialect, nested_limit
):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[_articles_join(nested_limit=nested_limit)],
        nest_joins=True,
    )

    assert _article_ids_by_author(result) == {1: [1], 2: [], 3: []}
    assert result["total_count"] == 3


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
async def test_where_on_an_inner_join_leaves_out_parents_without_a_match(
    async_session, dialect
):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[_articles_join(join_type="inner")],
        nest_joins=True,
    )

    assert _article_ids_by_author(result) == {1: [1]}
    assert result["total_count"] == 1


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
async def test_where_on_a_one_to_one_join_leaves_nulls(async_session, dialect):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Article).get_multi_joined(
        db=async_session, joins_config=[_author_join()], nest_joins=True
    )

    authors_by_article = {
        article["id"]: article["author"] and article["author"]["name"]
        for article in result["data"]
    }
    assert authors_by_article == {1: "Ann", 2: "Ann", 3: None}
    assert result["total_count"] == 3


async def test_where_on_a_one_to_one_join_without_nesting(async_session):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Article).get_multi_joined(
        db=async_session, joins_config=[_author_join()]
    )

    names_by_article = {
        article["id"]: article["author_name"] for article in result["data"]
    }
    assert names_by_article == {1: "Ann", 2: "Ann", 3: None}


async def test_where_on_get_joined_one_to_one(async_session):
    await _authors_with_articles(async_session)
    crud = FastCRUD(Article)

    matching = await crud.get_joined(
        db=async_session, joins_config=[_author_join()], nest_joins=True, id=1
    )
    not_matching = await crud.get_joined(
        db=async_session, joins_config=[_author_join()], nest_joins=True, id=3
    )

    assert matching is not None and matching["author"]["name"] == "Ann"
    assert not_matching is not None and not_matching["author"] is None


@WITH_AND_WITHOUT_NESTED_LIMIT
async def test_where_on_get_joined_one_to_many(async_session, nested_limit):
    await _authors_with_articles(async_session)
    crud = FastCRUD(Author)
    joins_config = [_articles_join(nested_limit=nested_limit)]

    ann = await crud.get_joined(
        db=async_session, joins_config=joins_config, nest_joins=True, id=1
    )
    bob = await crud.get_joined(
        db=async_session, joins_config=joins_config, nest_joins=True, id=2
    )

    assert ann is not None and [article["id"] for article in ann["articles"]] == [1]
    assert bob is not None and bob["articles"] == []


@WITH_AND_WITHOUT_NESTED_LIMIT
async def test_where_takes_a_list_of_conditions(async_session, nested_limit):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[
            _articles_join(
                nested_limit=nested_limit,
                where=[Article.title == "dropped", Article.author_id == 1],
            )
        ],
        nest_joins=True,
    )

    assert _article_ids_by_author(result) == {1: [2], 2: [], 3: []}


@WITH_AND_WITHOUT_NESTED_LIMIT
async def test_where_through_an_alias_names_the_alias(async_session, nested_limit):
    await _authors_with_articles(async_session)
    article = aliased(Article, name="article_alias")

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[
            _articles_join(
                alias=article,
                join_on=Author.id == article.author_id,
                where=article.title == "kept",
                nested_limit=nested_limit,
            )
        ],
        nest_joins=True,
    )

    assert _article_ids_by_author(result) == {1: [1], 2: [], 3: []}
    assert result["total_count"] == 3


@pytest.mark.parametrize(
    ("nested_limit", "refusal"),
    [
        (None, "side by side without joining them"),
        (
            5,
            r"can only add conditions on Article \(as article_alias\), not on articles",
        ),
    ],
    ids=["joined", "nested-limit"],
)
async def test_where_naming_the_unaliased_model_is_refused(
    async_session, nested_limit, refusal
):
    await _authors_with_articles(async_session)
    article = aliased(Article, name="article_alias")

    with pytest.raises(ValueError, match=refusal):
        await FastCRUD(Author).get_multi_joined(
            db=async_session,
            joins_config=[
                _articles_join(
                    alias=article,
                    join_on=Author.id == article.author_id,
                    where=Article.title == "kept",
                    nested_limit=nested_limit,
                )
            ],
            nest_joins=True,
        )


async def test_filters_choose_parents_where_where_does_not(async_session):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[_articles_join(where=None, filters={"title": "kept"})],
        nest_joins=True,
    )

    assert _article_ids_by_author(result) == {1: [1]}
    assert result["total_count"] == 1


@pytest.mark.parametrize("dialect", ON_EACH_DATABASE)
async def test_count_config_where_counts_only_matching_rows(async_session, dialect):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        counts_config=[
            CountConfig(
                model=Article,
                join_on=Article.author_id == Author.id,
                alias="kept_articles",
                where=Article.title == "kept",
            )
        ],
    )

    counts_by_author = {
        author["id"]: author["kept_articles"] for author in result["data"]
    }
    assert counts_by_author == {1: 1, 2: 0, 3: 0}
    assert result["total_count"] == 3


@pytest.mark.parametrize("config", [JoinConfig, CountConfig])
async def test_where_takes_only_sqlalchemy_conditions(config):
    with pytest.raises(TypeError, match="where= takes SQLAlchemy conditions"):
        config(model=Article, join_on=Author.id == Article.author_id, where="kept")


async def test_where_naming_the_parent_on_a_joined_query(async_session):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        joins_config=[_articles_join(where=Author.name == "Ann")],
        nest_joins=True,
    )

    assert _article_ids_by_author(result) == {1: [1, 2], 2: [], 3: []}
    assert result["total_count"] == 3


async def test_where_naming_the_parent_on_a_nested_limit_join_is_refused(
    async_session,
):
    await _authors_with_articles(async_session)

    with pytest.raises(
        ValueError, match="can only add conditions on Article, not on Author"
    ):
        await FastCRUD(Author).get_multi_joined(
            db=async_session,
            joins_config=[_articles_join(where=Author.name == "Ann", nested_limit=5)],
            nest_joins=True,
        )


async def test_an_inner_nested_limit_join_is_refused():
    with pytest.raises(ValueError, match="nested_limit can't be an inner join"):
        _articles_join(join_type="inner", nested_limit=5)


def _kept_articles_count(where):
    return CountConfig(
        model=Article,
        join_on=Article.author_id == Author.id,
        alias="counted_articles",
        where=where,
    )


async def test_count_config_where_naming_the_parent(async_session):
    await _authors_with_articles(async_session)

    result = await FastCRUD(Author).get_multi_joined(
        db=async_session,
        counts_config=[_kept_articles_count(Author.name == "Ann")],
    )

    counts_by_author = {
        author["id"]: author["counted_articles"] for author in result["data"]
    }
    assert counts_by_author == {1: 2, 2: 0, 3: 0}


async def test_count_config_where_naming_another_table_is_refused(async_session):
    await _authors_with_articles(async_session)
    async_session.add_all([Card(id=id, title=f"card {id}") for id in (1, 2, 3)])
    await async_session.commit()

    with pytest.raises(
        ValueError, match="CountConfig.where can only name Article .* not cards"
    ):
        await FastCRUD(Author).get_multi_joined(
            db=async_session,
            counts_config=[_kept_articles_count(Card.title != "zzz")],
        )


async def test_count_config_where_naming_an_association_table_its_join_on_reads(
    async_session,
):
    async_session.add_all(
        [
            Project(id=1, name="one"),
            Project(id=2, name="two"),
            Participant(id=1, name="ann"),
            Participant(id=2, name="bob"),
        ]
    )
    await async_session.flush()
    async_session.add_all(
        [
            ProjectsParticipantsAssociation(project_id=1, participant_id=1),
            ProjectsParticipantsAssociation(project_id=1, participant_id=2),
            ProjectsParticipantsAssociation(project_id=2, participant_id=2),
        ]
    )
    await async_session.commit()

    result = await FastCRUD(Project).get_multi_joined(
        db=async_session,
        counts_config=[
            CountConfig(
                model=Participant,
                join_on=(
                    Participant.id == ProjectsParticipantsAssociation.participant_id
                )
                & (ProjectsParticipantsAssociation.project_id == Project.id),
                alias="other_participants",
                where=ProjectsParticipantsAssociation.participant_id != 2,
            )
        ],
    )

    counts_by_project = {
        project["id"]: project["other_participants"] for project in result["data"]
    }
    assert counts_by_project == {1: 1, 2: 0}
