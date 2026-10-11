"""``PolicyConfig.hidden_columns`` and ``read_only_columns``: what a caller may see, use and write."""

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import column, exists, func, literal_column, select, text
from sqlalchemy.orm import aliased

from fastcrud import AccessDenied, CountConfig, FastCRUD, JoinConfig, PolicyConfig
from ..conftest import Article, Author, CategoryModel, Story, Writer

pytestmark = pytest.mark.asyncio

NAME_HIDDEN = "read on ModelTest may not filter or sort by: name"


class ItemNamed(BaseModel):
    id: int
    name: str


class ItemMaybeNamed(BaseModel):
    id: int
    name: str | None = None


class ItemCreate(BaseModel):
    id: int
    category_id: int
    name: str | None = None
    tier_id: int | None = None


@pytest_asyncio.fixture
async def unrestricted(
    async_session, test_model, tier_model, test_data, test_data_tier, test_data_category
):
    for tier in test_data_tier:
        async_session.add(tier_model(**tier))
    for category in test_data_category:
        async_session.add(CategoryModel(**category))
    await async_session.flush()
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()
    return FastCRUD(test_model)


def _columns_policy(**overrides):
    return PolicyConfig(
        **{
            "read": True,
            "create": True,
            "update": True,
            "delete": True,
            "hidden_columns": {"name"},
            "read_only_columns": {"tier_id"},
            **overrides,
        }
    )


@pytest.fixture
def crud(unrestricted):
    return unrestricted.with_policy(_columns_policy())


def _tier_join(test_model, tier_model, **overrides):
    return JoinConfig(
        **{
            "model": tier_model,
            "join_on": test_model.tier_id == tier_model.id,
            "join_prefix": "tier_",
            **overrides,
        }
    )


@pytest.mark.parametrize(
    ("rule", "name", "refusal"),
    [
        ("hidden_columns", "nmae", "hidden_columns names nmae, which isn't a column"),
        ("read_only_columns", "tier", "read_only_columns names tier, which isn't a"),
    ],
    ids=["unknown-name", "relationship"],
)
async def test_column_rules_name_only_columns(unrestricted, rule, name, refusal):
    policy = PolicyConfig(read=True, **{rule: {name}})

    with pytest.raises(ValueError, match=refusal):
        FastCRUD(unrestricted.model, policy=policy)
    with pytest.raises(ValueError, match=refusal):
        unrestricted.with_policy(policy)


async def test_reads_leave_hidden_columns_out(
    async_session, crud, test_model, tier_model
):
    row = await crud.get(async_session, id=1)
    many = await crud.get_multi(async_session)
    cursor = await crud.get_multi_by_cursor(async_session)
    selected = (await async_session.execute(await crud.select())).mappings().first()
    joined = await crud.get_joined(
        async_session, joins_config=[_tier_join(test_model, tier_model)], id=1
    )
    many_joined = await crud.get_multi_joined(
        async_session, joins_config=[_tier_join(test_model, tier_model)]
    )

    assert row is not None and "name" not in row and row["tier_id"] == 1
    assert all("name" not in item for item in many["data"])
    assert all("name" not in item for item in cursor["data"])
    assert selected is not None and "name" not in selected
    assert joined is not None and "name" not in joined
    assert joined["tier_name"] == "Premium"
    assert all("name" not in item for item in many_joined["data"])


async def test_an_optional_hidden_field_is_left_out_of_a_response_model(
    async_session, crud
):
    row = await crud.get(
        async_session, schema_to_select=ItemMaybeNamed, return_as_model=True, id=1
    )

    assert row == ItemMaybeNamed(id=1, name=None)


@pytest.mark.parametrize(
    "call",
    [
        lambda crud, db: crud.get(
            db, schema_to_select=ItemNamed, return_as_model=True, id=1
        ),
        lambda crud, db: crud.get_multi(
            db, schema_to_select=ItemNamed, return_as_model=True
        ),
        lambda crud, db: crud.get_multi_by_cursor(
            db, schema_to_select=ItemNamed, return_as_model=True
        ),
        lambda crud, db: crud.create(
            db,
            ItemCreate(id=20, category_id=1),
            schema_to_select=ItemNamed,
            return_as_model=True,
        ),
        lambda crud, db: crud.update(
            db,
            {"category_id": 2},
            schema_to_select=ItemNamed,
            return_as_model=True,
            id=1,
        ),
    ],
    ids=["get", "get_multi", "cursor", "create", "update"],
)
async def test_a_response_model_requiring_a_hidden_column_is_refused_up_front(
    async_session, crud, unrestricted, call
):
    with pytest.raises(AccessDenied) as denied:
        await call(crud, async_session)

    assert denied.value.columns == ("name",)
    assert str(denied.value) == "read on ModelTest may not return: name"
    assert not await unrestricted.exists(async_session, id=20)
    row = await unrestricted.get(async_session, id=1)
    assert row is not None and row["category_id"] == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda crud, db, m: crud.get_multi(db, name="Alice"),
        lambda crud, db, m: crud.get_multi(db, name__startswith="A"),
        lambda crud, db, m: crud.get_multi(db, _or={"name__eq": "Alice", "id": 1}),
        lambda crud, db, m: crud.get_multi(db, sort_columns="name"),
        lambda crud, db, m: crud.get_multi_by_cursor(db, sort_column="name"),
        lambda crud, db, m: crud.count(db, where=m.name == "Alice"),
        lambda crud, db, m: crud.exists(db, where=aliased(m).name == "Alice"),
        lambda crud, db, m: crud.get_multi(
            db, where=m.id.in_(select(m.id).where(m.name == "Alice"))
        ),
        lambda crud, db, m: crud.update(db, {"category_id": 2}, name="Alice"),
        lambda crud, db, m: crud.delete(db, where=m.name == "Alice"),
        lambda crud, db, m: crud.db_delete(db, name="Alice"),
    ],
    ids=[
        "filter",
        "operator",
        "multi-field-or",
        "sort",
        "cursor-sort",
        "where",
        "alias",
        "subquery",
        "update-filter",
        "delete-where",
        "db-delete-filter",
    ],
)
async def test_using_a_hidden_column_is_a_read_refusal(
    async_session, crud, test_model, call
):
    with pytest.raises(AccessDenied) as denied:
        await call(crud, async_session, test_model)

    assert denied.value.operation == "read"
    assert denied.value.reason == "columns"
    assert denied.value.columns == ("name",)
    assert str(denied.value) == NAME_HIDDEN


async def test_a_joins_where_naming_a_hidden_base_column_is_refused(
    async_session, crud, test_model, tier_model
):
    with pytest.raises(AccessDenied, match=NAME_HIDDEN):
        await crud.get_multi_joined(
            async_session,
            joins_config=[
                _tier_join(test_model, tier_model, where=test_model.name == "Alice")
            ],
        )
    with pytest.raises(AccessDenied, match=NAME_HIDDEN):
        await crud.get_multi_joined(
            async_session,
            counts_config=[
                CountConfig(
                    model=tier_model,
                    join_on=tier_model.id == test_model.tier_id,
                    where=test_model.name == "Alice",
                )
            ],
        )


async def test_joined_models_columns_are_never_hidden(
    async_session, crud, test_model, tier_model
):
    result = await crud.get_multi_joined(
        async_session,
        joins_config=[
            _tier_join(test_model, tier_model, where=tier_model.name == "Premium")
        ],
        **{"tier.name": "Premium"},
    )

    assert result["total_count"] == 6
    assert all(item["tier_name"] == "Premium" for item in result["data"])


async def test_the_policys_own_conditions_may_use_hidden_columns(
    async_session, unrestricted, test_model
):
    crud = unrestricted.with_policy(_columns_policy(read=test_model.name != "Alice"))

    assert await crud.count(async_session) == 9


@pytest.mark.parametrize(
    ("data", "columns"),
    [
        ({"name": "New"}, ("name",)),
        ({"tier_id": 1}, ("tier_id",)),
        ({"name": None, "tier_id": None}, ("name", "tier_id")),
    ],
    ids=["hidden", "read-only", "set-to-none"],
)
async def test_create_refuses_setting_an_unwritable_column(
    async_session, crud, unrestricted, data, columns
):
    with pytest.raises(AccessDenied) as denied:
        await crud.create(async_session, ItemCreate(id=20, category_id=1, **data))

    assert denied.value.operation == "create"
    assert denied.value.columns == columns
    assert (
        str(denied.value) == f"create on ModelTest may not write: {', '.join(columns)}"
    )
    assert not await unrestricted.exists(async_session, id=20)


async def test_create_leaves_unset_unwritable_columns_to_the_database(
    async_session, crud, unrestricted
):
    created = await crud.create(
        async_session, ItemCreate(id=20, category_id=1), schema_to_select=ItemCreate
    )

    row = await unrestricted.get(async_session, id=20)
    assert created is not None and "name" not in created
    assert row is not None and row["name"] is None and row["tier_id"] is None


async def test_server_code_writes_a_read_only_column_without_the_policy(
    async_session, crud, unrestricted
):
    await crud.with_policy(None).create(
        async_session, ItemCreate(id=20, category_id=1, tier_id=2)
    )

    row = await unrestricted.get(async_session, id=20)
    assert row is not None and row["tier_id"] == 2


@pytest.mark.parametrize(
    ("data", "columns"),
    [({"name": "New"}, ("name",)), ({"tier_id": 2}, ("tier_id",))],
    ids=["hidden", "read-only"],
)
async def test_update_refuses_setting_an_unwritable_column(
    async_session, crud, unrestricted, data, columns
):
    with pytest.raises(AccessDenied) as denied:
        await crud.update(async_session, data, id=1)

    assert denied.value.operation == "update"
    assert denied.value.columns == columns
    row = await unrestricted.get(async_session, id=1)
    assert row is not None and (row["name"], row["tier_id"]) == ("Charlie", 1)


async def test_update_returns_no_hidden_columns(async_session, crud):
    updated = await crud.update(
        async_session,
        {"category_id": 2},
        return_columns=["id", "name", "category_id"],
        id=1,
    )

    assert updated == {"id": 1, "category_id": 2}


async def test_fastcrud_writes_its_own_columns_even_when_read_only(
    async_session, unrestricted
):
    crud = unrestricted.with_policy(
        _columns_policy(read_only_columns={"is_deleted", "deleted_at"})
    )

    await crud.delete(async_session, id=1)

    deleted = await unrestricted.get(async_session, id=1)
    assert deleted is not None and deleted["is_deleted"] is True


class ItemCreateWithCategoryDefault(BaseModel):
    id: int
    name: str
    tier_id: int
    category_id: int | None = 2


async def test_create_leaves_out_an_unset_read_only_columns_schema_default(
    async_session, unrestricted
):
    crud = unrestricted.with_policy(
        _columns_policy(hidden_columns=frozenset(), read_only_columns={"category_id"})
    )

    await crud.create(
        async_session, ItemCreateWithCategoryDefault(id=20, name="Named", tier_id=1)
    )

    row = await unrestricted.get(async_session, id=20)
    assert row is not None and row["category_id"] is None


@pytest.mark.parametrize(
    "condition",
    [
        text("tier_id = 1"),
        column("name") == "Alice",
        literal_column("name = 'Alice'"),
    ],
    ids=["text", "column", "literal_column"],
)
async def test_raw_sql_in_where_is_refused_when_columns_are_hidden(
    async_session, crud, unrestricted, condition
):
    with pytest.raises(AccessDenied) as denied:
        await crud.get_multi(async_session, where=condition)

    assert denied.value.columns == ()
    assert "may not filter by raw SQL such as text()" in str(denied.value)
    assert await unrestricted.count(async_session, where=condition) > 0


async def test_only_column_refusals_carry_an_action(test_model):
    assert AccessDenied("read", test_model).action is None
    assert AccessDenied("update", test_model, reason="check").action is None
    assert AccessDenied("read", test_model, ["name"], reason="columns").action == (
        "filter"
    )


async def _authors_with_articles(async_session):
    async_session.add_all(
        [
            Author(id=1, name="Ann"),
            Author(id=2, name="Bob"),
            Article(id=1, title="a1", author_id=1),
            Article(id=2, title="a2", author_id=1),
            Article(id=3, title="b1", author_id=2),
        ]
    )
    await async_session.commit()


class AuthorMaybeIdentified(BaseModel):
    id: int | None = None
    name: str
    articles: list[dict] = []


@pytest.mark.parametrize("nested_limit", [None, 5], ids=["joined", "nested-limit"])
async def test_joined_reads_nest_by_a_hidden_primary_key(async_session, nested_limit):
    await _authors_with_articles(async_session)
    crud = FastCRUD(Author).with_policy(PolicyConfig(read=True, hidden_columns={"id"}))
    joins_config = [
        JoinConfig(
            model=Article,
            join_on=Author.id == Article.author_id,
            join_prefix="articles_",
            relationship_type="one-to-many",
            nested_limit=nested_limit,
        )
    ]

    many = await crud.get_multi_joined(
        async_session, joins_config=joins_config, nest_joins=True
    )
    one = await crud.get_joined(
        async_session, joins_config=joins_config, nest_joins=True, name="Ann"
    )
    as_model = await crud.get_joined(
        async_session,
        joins_config=joins_config,
        nest_joins=True,
        schema_to_select=AuthorMaybeIdentified,
        return_as_model=True,
        name="Bob",
    )

    titles = {
        author["name"]: sorted(article["title"] for article in author["articles"])
        for author in many["data"]
    }
    assert titles == {"Ann": ["a1", "a2"], "Bob": ["b1"]}
    assert all("id" not in author for author in many["data"])
    assert one is not None and "id" not in one
    assert sorted(article["title"] for article in one["articles"]) == ["a1", "a2"]
    assert as_model is not None and as_model.id is None
    assert [article["title"] for article in as_model.articles] == ["b1"]


async def test_a_nested_limit_join_links_parents_by_a_hidden_column(async_session):
    async_session.add_all(
        [
            Writer(id=1, code=7),
            Writer(id=2, code=8),
            Story(id=10, writer_code=7),
            Story(id=20, writer_code=8),
        ]
    )
    await async_session.commit()
    crud = FastCRUD(Writer).with_policy(
        PolicyConfig(read=True, hidden_columns={"code"})
    )

    result = await crud.get_multi_joined(
        async_session,
        joins_config=[
            JoinConfig(
                model=Story,
                join_on=Writer.code == Story.writer_code,
                join_prefix="stories_",
                relationship_type="one-to-many",
                nested_limit=5,
            )
        ],
        nest_joins=True,
    )

    stories_by_writer = {
        writer["id"]: [story["id"] for story in writer["stories"]]
        for writer in result["data"]
    }
    assert stories_by_writer == {1: [10], 2: [20]}
    assert all("code" not in writer for writer in result["data"])


@pytest.mark.parametrize(
    "condition",
    [
        lambda m, t: m.tier.has(t.name == "Premium"),
        lambda m, t: exists().where(t.id == m.tier_id, t.name == "Premium"),
        lambda m, t: select(t.id)
        .where(t.id == m.tier_id, t.name == "Premium")
        .exists(),
        lambda m, t: select(func.count())
        .where(t.id == m.tier_id, t.name == "Premium")
        .scalar_subquery()
        > 0,
    ],
    ids=["has", "exists", "select-exists", "count-subquery"],
)
async def test_sqlalchemys_own_subqueries_work_under_hidden_columns(
    async_session, crud, test_model, tier_model, condition
):
    assert await crud.count(async_session, where=condition(test_model, tier_model)) == 6


async def test_any_works_under_hidden_columns(async_session):
    await _authors_with_articles(async_session)
    crud = FastCRUD(Author).with_policy(
        PolicyConfig(read=True, hidden_columns={"name"})
    )

    result = await crud.get_multi(
        async_session, where=Author.articles.any(Article.title == "b1")
    )

    assert [author["id"] for author in result["data"]] == [2]
    assert "name" not in result["data"][0]
