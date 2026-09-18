import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import select

from fastcrud.crud.fast_crud import FastCRUD

from ..conftest import Article, Author, ProjectPoly


class ProjectPolyCreate(BaseModel):
    name: str


class ProjectPolyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    entity_type: str


@pytest.mark.asyncio
async def test_create_successful(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    result = await crud.create(async_session, new_data)

    # v0.20.0 behavior: create() without schema_to_select returns None
    assert result is None

    stmt = select(test_model).where(test_model.name == "New Record")
    db_result = await async_session.execute(stmt)
    fetched_record = db_result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "New Record"
    assert fetched_record.tier_id == 1


@pytest.mark.asyncio
async def test_create_and_read_successful(
    async_session, test_model, create_schema, read_schema
):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    created_record = await crud.create(
        async_session, new_data, schema_to_select=read_schema
    )

    assert created_record is not None
    assert created_record["name"] == "New Record"
    assert created_record["tier_id"] == 1


@pytest.mark.asyncio
async def test_create_and_read_missing_schema(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    with pytest.raises(ValueError):
        await crud.create(async_session, new_data, return_as_model=True)


@pytest.mark.asyncio
async def test_create_and_read_successful_return_as_model(
    async_session, test_model, create_schema, read_schema
):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="New Record", tier_id=1)
    created_record = await crud.create(
        async_session,
        new_data,
        schema_to_select=read_schema,
        return_as_model=True,
    )

    assert created_record is not None
    assert created_record.name == "New Record"
    assert created_record.tier_id == 1


@pytest.mark.asyncio
async def test_create_no_commit(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="No Commit Record", tier_id=1)
    result = await crud.create(async_session, new_data, commit=False)

    # v0.20.0 behavior: create() without schema_to_select returns None
    assert result is None

    await async_session.rollback()

    stmt = select(test_model).where(test_model.name == "No Commit Record")
    db_result = await async_session.execute(stmt)
    fetched_record = db_result.scalar_one_or_none()

    assert fetched_record is None


@pytest.mark.asyncio
async def test_create_no_commit_read(
    async_session, test_model, create_schema, read_schema
):
    crud = FastCRUD(test_model)
    new_data = create_schema(name="No Commit Read", tier_id=2)
    created_record = await crud.create(
        async_session, new_data, commit=False, schema_to_select=read_schema
    )

    assert created_record is not None
    assert created_record["name"] == "No Commit Read"
    assert created_record["tier_id"] == 2

    await async_session.rollback()

    stmt = select(test_model).where(test_model.name == "No Commit Read")
    result = await async_session.execute(stmt)
    fetched_record = result.scalar_one_or_none()

    assert fetched_record is None


@pytest.mark.asyncio
async def test_create_with_various_valid_data(async_session, test_model, create_schema):
    valid_data_samples = [
        {"name": "Example 1", "tier_id": 1},
        {"name": "Example 2", "tier_id": 2},
    ]

    for data in valid_data_samples:
        crud = FastCRUD(test_model)
        new_data = create_schema(**data)
        result = await crud.create(async_session, new_data)

        # v0.20.0 behavior: create() without schema_to_select returns None
        assert result is None

        stmt = select(test_model).where(test_model.name == data["name"])
        db_result = await async_session.execute(stmt)
        fetched_record = db_result.scalar_one_or_none()

        assert fetched_record is not None
        assert fetched_record.name == data["name"]
        assert fetched_record.tier_id == data["tier_id"]


@pytest.mark.asyncio
async def test_create_with_missing_fields(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    incomplete_data = {"name": "Missing Tier"}
    with pytest.raises(ValidationError):
        await crud.create(async_session, create_schema(**incomplete_data))


@pytest.mark.asyncio
async def test_create_with_extra_fields(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    extra_data = {"name": "Extra", "tier_id": 1, "extra_field": "value"}
    with pytest.raises(ValidationError):
        await crud.create(async_session, create_schema(**extra_data))


@pytest.mark.asyncio
async def test_create_with_invalid_data_types(async_session, test_model, create_schema):
    crud = FastCRUD(test_model)
    invalid_data = {"name": 123, "tier_id": "invalid"}
    with pytest.raises(ValidationError):
        await crud.create(async_session, create_schema(**invalid_data))


@pytest.mark.asyncio
async def test_create_successful_multi_pk(
    async_session, multi_pk_model, multi_pk_test_create_schema
):
    crud = FastCRUD(multi_pk_model)
    new_data = multi_pk_test_create_schema(name="New Record", id=1, uuid="a")
    result = await crud.create(async_session, new_data)

    # v0.20.0 behavior: create() without schema_to_select returns None
    assert result is None

    stmt = select(multi_pk_model).where(multi_pk_model.name == "New Record")
    db_result = await async_session.execute(stmt)
    fetched_record = db_result.scalar_one_or_none()

    assert fetched_record is not None
    assert fetched_record.name == "New Record"
    assert fetched_record.id == 1
    assert fetched_record.uuid == "a"


@pytest.mark.asyncio
async def test_create_returns_inherited_columns(async_session):
    """create() must include columns from parent tables under joined-table inheritance.

    ProjectPoly inherits from EntityPoly, so ``entity_type`` lives on the parent
    table. Before the inspect()/column_attrs fix, ``__table__.columns`` only
    walked the child table and the discriminator went missing from the dict
    used to build the response schema.

    Not mirrored under tests/sqlmodel/ because SQLModel joined-table inheritance
    is fragile (subclass ``id`` redeclaration breaks the SQLModel metaclass
    mapping). The fix lives in shared code, so this sqlalchemy regression test
    is sufficient.
    """
    crud = FastCRUD(ProjectPoly)
    result = await crud.create(
        async_session,
        ProjectPolyCreate(name="Apollo"),
        schema_to_select=ProjectPolyRead,
    )

    assert result is not None
    assert result["name"] == "Apollo"
    # entity_type is on the parent table (entities_poly), not projects_poly
    assert result["entity_type"] == "project"


class AuthorCreate(BaseModel):
    name: str


class ArticleTitle(BaseModel):
    title: str


class ArticleWithAuthor(BaseModel):
    title: str
    author: AuthorCreate


class AuthorWithArticles(BaseModel):
    name: str
    articles: list[ArticleTitle]


@pytest.mark.asyncio
async def test_create_persists_a_nested_to_one_object(async_session):
    """A nested object is created alongside its owner and linked to it."""
    crud = FastCRUD(Article)

    await crud.create(
        async_session,
        ArticleWithAuthor(title="Nested", author=AuthorCreate(name="Nested Author")),
    )

    article = (
        await async_session.execute(select(Article).where(Article.title == "Nested"))
    ).scalar_one()
    author = (
        await async_session.execute(
            select(Author).where(Author.name == "Nested Author")
        )
    ).scalar_one()
    assert article.author_id == author.id


@pytest.mark.asyncio
async def test_create_persists_a_list_of_nested_objects(async_session):
    """A to-many relationship arrives as a list of dicts and is created as rows."""
    crud = FastCRUD(Author)

    await crud.create(
        async_session,
        AuthorWithArticles(
            name="Prolific Author",
            articles=[ArticleTitle(title="First"), ArticleTitle(title="Second")],
        ),
    )

    author = (
        await async_session.execute(
            select(Author).where(Author.name == "Prolific Author")
        )
    ).scalar_one()
    articles = (
        (
            await async_session.execute(
                select(Article).where(Article.author_id == author.id)
            )
        )
        .scalars()
        .all()
    )
    assert sorted(article.title for article in articles) == ["First", "Second"]


@pytest.mark.asyncio
async def test_create_nests_more_than_one_level_deep(async_session):
    """A nested object carrying its own nested objects is built all the way down."""

    class AuthorWithNested(BaseModel):
        name: str
        articles: list[ArticleTitle]

    class ArticleWithNestedAuthor(BaseModel):
        title: str
        author: AuthorWithNested

    crud = FastCRUD(Article)

    await crud.create(
        async_session,
        ArticleWithNestedAuthor(
            title="Outer",
            author=AuthorWithNested(
                name="Deep Author", articles=[ArticleTitle(title="Inner")]
            ),
        ),
    )

    author = (
        await async_session.execute(select(Author).where(Author.name == "Deep Author"))
    ).scalar_one()
    titles = sorted(
        article.title
        for article in (
            (
                await async_session.execute(
                    select(Article).where(Article.author_id == author.id)
                )
            )
            .scalars()
            .all()
        )
    )
    assert titles == ["Inner", "Outer"]


@pytest.mark.asyncio
async def test_create_rejects_a_single_object_for_a_to_many_relationship(async_session):
    """The wrong shape is named here, not deep inside SQLAlchemy."""

    class AuthorWithOneArticle(BaseModel):
        name: str
        articles: ArticleTitle

    crud = FastCRUD(Author)

    with pytest.raises(ValueError, match="'articles' is a to-many relationship"):
        await crud.create(
            async_session,
            AuthorWithOneArticle(
                name="Confused Author", articles=ArticleTitle(title="Only one")
            ),
        )


@pytest.mark.asyncio
async def test_create_rejects_a_list_for_a_to_one_relationship(async_session):
    """And the same the other way around."""

    class ArticleWithAuthors(BaseModel):
        title: str
        author: list[AuthorCreate]

    crud = FastCRUD(Article)

    with pytest.raises(ValueError, match="'author' is a to-one relationship"):
        await crud.create(
            async_session,
            ArticleWithAuthors(title="Confused", author=[AuthorCreate(name="A")]),
        )


@pytest.mark.asyncio
async def test_create_treats_a_none_relationship_as_nothing_to_relate(async_session):
    """An optional nested field left unset creates the row and no relations."""

    class AuthorOptionalArticles(BaseModel):
        name: str
        articles: list[ArticleTitle] | None = None

    await FastCRUD(Author).create(
        async_session, AuthorOptionalArticles(name="Unset Articles")
    )

    author = (
        await async_session.execute(
            select(Author).where(Author.name == "Unset Articles")
        )
    ).scalar_one()
    articles = (
        (
            await async_session.execute(
                select(Article).where(Article.author_id == author.id)
            )
        )
        .scalars()
        .all()
    )
    assert articles == []


@pytest.mark.asyncio
async def test_create_without_relationships_is_unchanged(async_session):
    """A flat schema still creates exactly one row."""
    crud = FastCRUD(Author)

    await crud.create(async_session, AuthorCreate(name="Flat Author"))

    author = (
        await async_session.execute(select(Author).where(Author.name == "Flat Author"))
    ).scalar_one()
    assert author.name == "Flat Author"
