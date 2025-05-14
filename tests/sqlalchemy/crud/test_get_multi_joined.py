from typing import Annotated
import pytest
from fastcrud import FastCRUD, JoinConfig, aliased
from pydantic import BaseModel, Field
from ...sqlalchemy.conftest import (
    ModelTest,
    TierModel,
    CreateSchemaTest,
    TierSchemaTest,
    ReadSchemaTest,
    CategoryModel,
    CategorySchemaTest,
    BookingModel,
    BookingSchema,
    Project,
    Participant,
    ProjectsParticipantsAssociation,
    Article,
    Card,
    ArticleSchema,
    CardSchema,
    Client,
    Department,
    User,
    Task,
    ClientRead,
    DepartmentRead,
    UserReadSub,
    TaskRead,
)


class JoinedTestTier(BaseModel):
    name: str
    tier_id: int
    tier_name: str


class CustomCreateSchemaTest(BaseModel):
    name: Annotated[str, Field(max_length=20)]
    tier_id: int


@pytest.mark.asyncio
async def test_get_multi_joined_basic(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        offset=0,
        limit=10,
    )

    assert len(result["data"]) == min(10, len(test_data))
    assert result["total_count"] == len(test_data)
    assert all("tier_name" in item for item in result["data"])


@pytest.mark.asyncio
async def test_get_multi_joined_unpaginated(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        offset=0,
        limit=None,
    )

    assert len(result["data"]) == len(test_data)
    assert result["total_count"] == len(test_data)
    assert all("tier_name" in item for item in result["data"])


@pytest.mark.asyncio
async def test_get_multi_joined_sorting(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        sort_columns=["name"],
        sort_orders=["asc"],
        offset=0,
        limit=10,
    )

    assert len(result["data"]) <= 10
    assert all(
        result["data"][i]["name"] <= result["data"][i + 1]["name"]
        for i in range(len(result["data"]) - 1)
    )


@pytest.mark.asyncio
async def test_get_multi_joined_filtering(async_session, test_data, test_data_tier):
    specific_user_name = "Charlie"
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        name=specific_user_name,
        offset=0,
        limit=10,
    )

    assert len(result["data"]) <= 10
    assert all(item["name"] == specific_user_name for item in result["data"])


@pytest.mark.asyncio
async def test_get_multi_joined_different_join_types(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    for join_type in ["left", "inner"]:
        result = await crud.get_multi_joined(
            db=async_session,
            join_model=TierModel,
            join_type=join_type,
            schema_to_select=CreateSchemaTest,
            join_schema_to_select=TierSchemaTest,
            offset=0,
            limit=10,
        )

        assert len(result["data"]) <= 10


@pytest.mark.asyncio
async def test_get_multi_joined_return_model(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=JoinedTestTier,
        join_schema_to_select=TierSchemaTest,
        join_prefix="tier_",
        return_as_model=True,
        offset=0,
        limit=10,
    )

    assert all(isinstance(item, JoinedTestTier) for item in result["data"])


@pytest.mark.asyncio
async def test_get_multi_joined_no_results(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        offset=0,
        limit=10,
        name="NonExistingName",
    )

    assert len(result["data"]) == 0
    assert result["total_count"] == 0


@pytest.mark.asyncio
async def test_get_multi_joined_large_offset(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        offset=1000,
        limit=10,
    )

    assert len(result["data"]) == 0


@pytest.mark.asyncio
async def test_get_multi_joined_invalid_limit_offset(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError):
        await crud.get_multi_joined(
            db=async_session,
            join_model=TierModel,
            schema_to_select=CreateSchemaTest,
            join_schema_to_select=TierSchemaTest,
            offset=-1,
            limit=10,
        )
    with pytest.raises(ValueError):
        await crud.get_multi_joined(
            db=async_session,
            join_model=TierModel,
            schema_to_select=CreateSchemaTest,
            join_schema_to_select=TierSchemaTest,
            offset=0,
            limit=-1,
        )


@pytest.mark.asyncio
async def test_get_multi_joined_advanced_filtering(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    advanced_filter_result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=ReadSchemaTest,
        join_schema_to_select=TierSchemaTest,
        join_prefix="tier_",
        offset=0,
        limit=10,
        id__gt=5,
    )

    assert (
        len(advanced_filter_result["data"]) > 0
    ), "Should fetch records with ID greater than 5"
    assert all(
        item["id"] > 5 for item in advanced_filter_result["data"]
    ), "All fetched records should meet the advanced filter condition"


@pytest.mark.asyncio
async def test_get_multi_joined_with_additional_join_model(
    async_session, test_data, test_data_tier, test_data_category
):
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    await async_session.commit()

    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_multi_joined(
        db=async_session,
        joins_config=[
            JoinConfig(
                model=TierModel,
                join_prefix="tier_",
                schema_to_select=TierSchemaTest,
                join_on=ModelTest.tier_id == TierModel.id,
                join_type="left",
            ),
            JoinConfig(
                model=CategoryModel,
                join_prefix="category_",
                schema_to_select=CategorySchemaTest,
                join_on=ModelTest.category_id == CategoryModel.id,
                join_type="left",
            ),
        ],
        schema_to_select=ReadSchemaTest,
        offset=0,
        limit=10,
    )

    assert len(result["data"]) == min(10, len(test_data))
    assert result["total_count"] == len(test_data)
    assert all(
        "tier_name" in item and "category_name" in item for item in result["data"]
    )


@pytest.mark.asyncio
async def test_get_multi_joined_with_aliases(
    async_session, test_data, test_data_tier, test_data_category, test_data_booking
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    for booking_item in test_data_booking:
        async_session.add(BookingModel(**booking_item))
    await async_session.commit()

    crud = FastCRUD(BookingModel)

    expected_owner_name = "Charlie"
    expected_user_name = "Alice"

    owner_alias = aliased(ModelTest, name="owner")
    user_alias = aliased(ModelTest, name="user")

    result = await crud.get_multi_joined(
        db=async_session,
        schema_to_select=BookingSchema,
        joins_config=[
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.owner_id == owner_alias.id,
                join_prefix="owner_",
                alias=owner_alias,
                schema_to_select=ReadSchemaTest,
            ),
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.user_id == user_alias.id,
                join_prefix="user_",
                alias=user_alias,
                schema_to_select=ReadSchemaTest,
            ),
        ],
        offset=0,
        limit=10,
        sort_columns=["booking_date"],
        sort_orders=["asc"],
    )

    assert "data" in result and isinstance(
        result["data"], list
    ), "The result should have a 'data' key with a list of records."
    for booking in result["data"]:
        assert (
            "owner_name" in booking
        ), "Each record should include 'owner_name' from the joined owner ModelTest data."
        assert (
            "user_name" in booking
        ), "Each record should include 'user_name' from the joined user ModelTest data."
    assert result is not None
    assert result["total_count"] >= 1, "Expected at least one booking record"
    first_result = result["data"][0]
    assert (
        first_result["owner_name"] == expected_owner_name
    ), "Owner name does not match expected value"
    assert (
        first_result["user_name"] == expected_user_name
    ), "User name does not match expected value"


@pytest.mark.asyncio
async def test_get_multi_joined_with_aliases_no_schema(
    async_session, test_data, test_data_tier, test_data_category, test_data_booking
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    for booking_item in test_data_booking:
        async_session.add(BookingModel(**booking_item))
    await async_session.commit()

    crud = FastCRUD(BookingModel)

    expected_owner_name = "Charlie"
    expected_user_name = "Alice"

    owner_alias = aliased(ModelTest, name="owner")
    user_alias = aliased(ModelTest, name="user")

    result = await crud.get_multi_joined(
        db=async_session,
        schema_to_select=BookingSchema,
        joins_config=[
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.owner_id == owner_alias.id,
                join_prefix="owner_",
                alias=owner_alias,
            ),
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.user_id == user_alias.id,
                join_prefix="user_",
                alias=user_alias,
            ),
        ],
        offset=0,
        limit=10,
        sort_columns=["booking_date"],
        sort_orders=["asc"],
    )

    assert "data" in result and isinstance(
        result["data"], list
    ), "The result should have a 'data' key with a list of records."
    for booking in result["data"]:
        assert (
            "owner_name" in booking
        ), "Each record should include 'owner_name' from the joined owner ModelTest data."
        assert (
            "user_name" in booking
        ), "Each record should include 'user_name' from the joined user ModelTest data."
    assert result is not None
    assert result["total_count"] >= 1, "Expected at least one booking record"
    first_result = result["data"][0]
    assert (
        first_result["owner_name"] == expected_owner_name
    ), "Owner name does not match expected value"
    assert (
        first_result["user_name"] == expected_user_name
    ), "User name does not match expected value"


@pytest.mark.asyncio
async def test_many_to_many_joined(async_session):
    project1 = Project(id=1, name="Project 1", description="First Project")
    project2 = Project(id=2, name="Project 2", description="Second Project")

    participant1 = Participant(id=1, name="Participant 1", role="Developer")
    participant2 = Participant(id=2, name="Participant 2", role="Designer")

    async_session.add_all([project1, project2, participant1, participant2])
    await async_session.commit()

    projects_participants1 = ProjectsParticipantsAssociation(
        project_id=1, participant_id=1
    )
    projects_participants2 = ProjectsParticipantsAssociation(
        project_id=1, participant_id=2
    )
    projects_participants3 = ProjectsParticipantsAssociation(
        project_id=2, participant_id=1
    )

    async_session.add_all(
        [projects_participants1, projects_participants2, projects_participants3]
    )
    await async_session.commit()

    crud_project = FastCRUD(Project)

    join_condition_1 = Project.id == ProjectsParticipantsAssociation.project_id
    join_condition_2 = ProjectsParticipantsAssociation.participant_id == Participant.id

    joins_config = [
        JoinConfig(
            model=ProjectsParticipantsAssociation,
            join_on=join_condition_1,
            join_type="inner",
            join_prefix="pp_",
        ),
        JoinConfig(
            model=Participant,
            join_on=join_condition_2,
            join_type="inner",
            join_prefix="participant_",
        ),
    ]

    records = await crud_project.get_multi_joined(
        db=async_session,
        joins_config=joins_config,
    )

    expected_results = [
        {
            "project_id": 1,
            "participant_id": 1,
            "participant_name": "Participant 1",
            "participant_role": "Developer",
        },
        {
            "project_id": 1,
            "participant_id": 2,
            "participant_name": "Participant 2",
            "participant_role": "Designer",
        },
        {
            "project_id": 2,
            "participant_id": 1,
            "participant_name": "Participant 1",
            "participant_role": "Developer",
        },
    ]

    assert len(records["data"]) == 3, "Expected three project-participant associations"
    assert (
        len(records["data"]) == records["total_count"]
    ), "Number of records should be the same in total_count and len"

    for expected, actual in zip(expected_results, records["data"]):
        assert (
            actual["id"] == expected["project_id"]
        ), f"Project ID mismatch. Expected: {expected['project_id']}, Got: {actual['id']}"
        assert (
            actual["participant_id"] == expected["participant_id"]
        ), f"Participant ID mismatch. Expected: {expected['participant_id']}, Got: {actual['participant_id']}"
        assert (
            actual["participant_name"] == expected["participant_name"]
        ), f"Participant name mismatch. Expected: {expected['participant_name']}, Got: {actual['participant_name']}"
        assert (
            actual["participant_role"] == expected["participant_role"]
        ), f"Participant role mismatch. Expected: {expected['participant_role']}, Got: {actual['participant_role']}"


@pytest.mark.asyncio
async def test_get_multi_joined_conflicting_join_parameters(async_session):
    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError) as exc_info:
        await crud.get_multi_joined(
            db=async_session,
            join_model=TierModel,
            joins_config=[
                JoinConfig(model=TierModel, join_on=TierModel.id == ModelTest.tier_id)
            ],
        )
    assert (
        "Cannot use both single join parameters and joins_config simultaneously"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_get_multi_joined_missing_join_parameters(async_session):
    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError) as exc_info:
        await crud.get_multi_joined(db=async_session)
    assert "You need one of join_model or joins_config" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_multi_joined_unsupported_join_type(async_session, test_data):
    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError) as exc_info:
        await crud.get_multi_joined(
            db=async_session,
            join_model=TierModel,
            join_type="unsupported_join_type",
        )
    assert "Unsupported join type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_multi_joined_with_joined_model_filters(
    async_session, test_data, test_data_tier
):
    for tier_data in test_data_tier:
        async_session.add(TierModel(**tier_data))
    await async_session.commit()

    for test_item in test_data:
        async_session.add(ModelTest(**test_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        join_filters={"name": "Premium"},
        schema_to_select=ReadSchemaTest,
        join_schema_to_select=TierSchemaTest,
        join_prefix="tier_",
        offset=0,
        limit=10,
    )

    assert (
        len(result["data"]) > 0
    ), "Expected to find at least one ModelTest record associated with the 'Premium' tier."
    for item in result["data"]:
        assert (
            item["tier_name"] == "Premium"
        ), "Expected tier_name to be 'Premium' for all fetched records."


@pytest.mark.asyncio
async def test_get_multi_joined_missing_schema_to_select(async_session, test_data):
    for test_item in test_data:
        async_session.add(ModelTest(**test_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError) as exc_info:
        await crud.get_multi_joined(
            db=async_session,
            join_model=TierModel,
            return_as_model=True,
        )
    assert "schema_to_select must be provided when return_as_model is True" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_get_multi_joined_validation_error(
    async_session, test_data, test_model, test_data_tier
):
    for tier_data in test_data_tier:
        async_session.add(TierModel(**tier_data))
    await async_session.commit()

    for test_item in test_data:
        async_session.add(ModelTest(**test_item))
    await async_session.commit()

    invalid_test_data = {
        "name": "Extremely Long Name That Exceeds The Limits Of CustomCreateSchemaTest",
        "tier_id": 1,
    }
    async_session.add(test_model(**invalid_test_data))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError) as exc_info:
        await crud.get_multi_joined(
            db=async_session,
            join_model=TierModel,
            return_as_model=True,
            schema_to_select=CustomCreateSchemaTest,
        )

    assert "Data validation error for schema CustomCreateSchemaTest:" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_get_multi_joined_with_nesting(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    result = await crud.get_multi_joined(
        db=async_session,
        joins_config=[
            JoinConfig(
                model=TierModel,
                join_on=ModelTest.tier_id == TierModel.id,
                join_prefix="tier_",
                schema_to_select=TierSchemaTest,
                join_type="left",
            ),
            JoinConfig(
                model=CategoryModel,
                join_on=ModelTest.category_id == CategoryModel.id,
                join_prefix="category_",
                schema_to_select=CategorySchemaTest,
                join_type="left",
            ),
        ],
        schema_to_select=CreateSchemaTest,
        nest_joins=True,
        offset=0,
        limit=10,
    )

    assert result is not None, "Expected non-None result for multi joined query"
    assert "data" in result, "Result should contain 'data' key"
    assert isinstance(result["data"], list), "'data' should be a list"

    if result["data"]:
        for item in result["data"]:
            assert "tier" in item, "Nested tier data should be present under key 'tier'"
            assert (
                "category" in item
            ), "Nested category data should be present under key 'category'"
            assert item["tier"] is None or isinstance(
                item["tier"], dict
            ), "Nested tier data should be a dictionary"
            assert item["category"] is None or isinstance(
                item["category"], dict
            ), "Nested category data should be a dictionary"
            if item["tier"] is not None:
                assert (
                    "tier_" not in item["tier"]
                ), "No prefix should be present in the nested tier keys"
            if item["category"] is not None:  # pragma: no cover
                assert (
                    "category_" not in item["category"]
                ), "No prefix should be present in the nested category keys"


@pytest.mark.asyncio
async def test_get_multi_joined_no_prefix_regular(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        limit=10,
    )

    assert result and result["data"], "Expected data in the result."
    for item in result["data"]:
        assert "name" in item, "Expected user name in each item."
        assert "name_1" in item, "Expected tier name in each item without prefix."


@pytest.mark.asyncio
async def test_get_multi_joined_no_prefix_nested(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    result = await crud.get_multi_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        nest_joins=True,
        limit=10,
    )

    assert result and result["data"], "Expected data in the result."
    for item in result["data"]:
        assert "name" in item, "Expected user name in each item."
        assert (
            TierModel.__tablename__ in item
        ), f"Expected nested '{TierModel.__tablename__}' key in each item."
        assert (
            "name" in item[TierModel.__tablename__]
        ), f"Expected 'name' field inside nested '{TierModel.__tablename__}' dictionary."


@pytest.mark.asyncio
async def test_get_multi_joined_card_with_articles(async_session):
    cards = [
        Card(title="Test Card"),
        Card(title="Test Card 2"),
        Card(title="Test Card 3"),
    ]
    async_session.add_all(cards)
    await async_session.flush()

    articles = [
        Article(title="Article 1", card_id=cards[0].id),
        Article(title="Article 2", card_id=cards[1].id),
        Article(title="Article 3", card_id=cards[1].id),
    ]
    async_session.add_all(articles)
    await async_session.commit()

    card_crud = FastCRUD(Card)

    result = await card_crud.get_multi_joined(
        db=async_session,
        nest_joins=True,
        joins_config=[
            JoinConfig(
                model=Article,
                join_on=Article.card_id == Card.id,
                join_prefix="articles_",
                join_type="left",
                relationship_type="one-to-many",
            )
        ],
    )

    assert result is not None, "No data returned from the database."
    assert "data" in result, "Result should contain 'data' key."
    data = result["data"]
    assert isinstance(data, list), "Result data should be a list."
    assert len(data) == 3, "Expected three card records."

    card1 = next((c for c in data if c["id"] == cards[0].id), None)
    card2 = next((c for c in data if c["id"] == cards[1].id), None)
    card3 = next((c for c in data if c["id"] == cards[2].id), None)

    assert (
        card1 is not None and "articles" in card1
    ), "Card 1 should have nested articles."
    assert len(card1["articles"]) == 1, "Card 1 should have one article."
    assert (
        card1["articles"][0]["title"] == "Article 1"
    ), "Card 1's article title should be 'Article 1'."

    assert (
        card2 is not None and "articles" in card2
    ), "Card 2 should have nested articles."
    assert len(card2["articles"]) == 2, "Card 2 should have two articles."
    assert (
        card2["articles"][0]["title"] == "Article 2"
    ), "Card 2's first article title should be 'Article 2'."
    assert (
        card2["articles"][1]["title"] == "Article 3"
    ), "Card 2's second article title should be 'Article 3'."

    assert (
        card3 is not None and "articles" in card3
    ), "Card 3 should have nested articles."
    assert len(card3["articles"]) == 0, "Card 3 should have no articles."


@pytest.mark.asyncio
async def test_get_multi_joined_card_with_multiple_articles(async_session):
    cards = [
        Card(title="Card A"),
        Card(title="Card B"),
        Card(title="Card C"),
        Card(title="Card D"),
    ]
    async_session.add_all(cards)
    await async_session.flush()

    articles = [
        Article(title="Article 1", card_id=cards[0].id),
        Article(title="Article 2", card_id=cards[0].id),
        Article(title="Article 3", card_id=cards[1].id),
        Article(title="Article 4", card_id=cards[1].id),
        Article(title="Article 5", card_id=cards[2].id),
    ]
    async_session.add_all(articles)
    await async_session.commit()

    card_crud = FastCRUD(Card)

    result = await card_crud.get_multi_joined(
        db=async_session,
        nest_joins=True,
        joins_config=[
            JoinConfig(
                model=Article,
                join_on=Article.card_id == Card.id,
                join_prefix="articles_",
                join_type="left",
                relationship_type="one-to-many",
            )
        ],
    )

    assert result is not None, "No data returned from the database."
    assert "data" in result, "Result should contain 'data' key."
    data = result["data"]
    assert isinstance(data, list), "Result data should be a list."
    assert len(data) == 4, "Expected four card records."

    card_a = next((c for c in data if c["id"] == cards[0].id), None)
    card_b = next((c for c in data if c["id"] == cards[1].id), None)
    card_c = next((c for c in data if c["id"] == cards[2].id), None)
    card_d = next((c for c in data if c["id"] == cards[3].id), None)

    assert (
        card_a is not None and "articles" in card_a
    ), "Card A should have nested articles."
    assert len(card_a["articles"]) == 2, "Card A should have two articles."
    assert (
        card_a["articles"][0]["title"] == "Article 1"
    ), "Card A's first article title should be 'Article 1'."
    assert (
        card_a["articles"][1]["title"] == "Article 2"
    ), "Card A's second article title should be 'Article 2'."

    assert (
        card_b is not None and "articles" in card_b
    ), "Card B should have nested articles."
    assert len(card_b["articles"]) == 2, "Card B should have two articles."
    assert (
        card_b["articles"][0]["title"] == "Article 3"
    ), "Card B's first article title should be 'Article 3'."
    assert (
        card_b["articles"][1]["title"] == "Article 4"
    ), "Card B's second article title should be 'Article 4'."

    assert (
        card_c is not None and "articles" in card_c
    ), "Card C should have nested articles."
    assert len(card_c["articles"]) == 1, "Card C should have one article."
    assert (
        card_c["articles"][0]["title"] == "Article 5"
    ), "Card C's article title should be 'Article 5'."

    assert (
        card_d is not None and "articles" in card_d
    ), "Card D should have nested articles."
    assert len(card_d["articles"]) == 0, "Card D should have no articles."


@pytest.mark.asyncio
async def test_get_multi_joined_card_with_multiple_articles_as_models(async_session):
    cards = [
        Card(title="Card A"),
        Card(title="Card B"),
        Card(title="Card C"),
        Card(title="Card D"),
    ]
    async_session.add_all(cards)
    await async_session.flush()

    articles = [
        Article(title="Article 1", card_id=cards[0].id),
        Article(title="Article 2", card_id=cards[0].id),
        Article(title="Article 3", card_id=cards[1].id),
        Article(title="Article 4", card_id=cards[1].id),
        Article(title="Article 5", card_id=cards[2].id),
    ]
    async_session.add_all(articles)
    await async_session.commit()

    card_crud = FastCRUD(Card)

    result = await card_crud.get_multi_joined(
        db=async_session,
        nest_joins=True,
        return_as_model=True,
        schema_to_select=CardSchema,
        nested_schema_to_select={"articles_": ArticleSchema},
        joins_config=[
            JoinConfig(
                model=Article,
                join_on=Article.card_id == Card.id,
                join_prefix="articles_",
                join_type="left",
                relationship_type="one-to-many",
            )
        ],
    )

    assert result is not None, "No data returned from the database."
    assert "data" in result, "Result should contain 'data' key."
    data = result["data"]
    assert isinstance(data, list), "Result data should be a list."
    assert len(data) == 4, "Expected four card records."
    assert all(
        isinstance(card, CardSchema) for card in data
    ), "All items should be instances of CardSchema."

    card_a = next((c for c in data if c.id == cards[0].id), None)
    card_b = next((c for c in data if c.id == cards[1].id), None)
    card_c = next((c for c in data if c.id == cards[2].id), None)
    card_d = next((c for c in data if c.id == cards[3].id), None)

    assert card_a is not None and hasattr(
        card_a, "articles"
    ), "Card A should have nested articles."
    assert len(card_a.articles) == 2, "Card A should have two articles."
    assert (
        card_a.articles[0].title == "Article 1"
    ), "Card A's first article title should be 'Article 1'."
    assert (
        card_a.articles[1].title == "Article 2"
    ), "Card A's second article title should be 'Article 2'."
    assert all(
        isinstance(article, ArticleSchema) for article in card_a.articles
    ), "All articles in Card A should be instances of ArticleSchema."

    assert card_b is not None and hasattr(
        card_b, "articles"
    ), "Card B should have nested articles."
    assert len(card_b.articles) == 2, "Card B should have two articles."
    assert (
        card_b.articles[0].title == "Article 3"
    ), "Card B's first article title should be 'Article 3'."
    assert (
        card_b.articles[1].title == "Article 4"
    ), "Card B's second article title should be 'Article 4'."
    assert all(
        isinstance(article, ArticleSchema) for article in card_b.articles
    ), "All articles in Card B should be instances of ArticleSchema."

    assert card_c is not None and hasattr(
        card_c, "articles"
    ), "Card C should have nested articles."
    assert len(card_c.articles) == 1, "Card C should have one article."
    assert (
        card_c.articles[0].title == "Article 5"
    ), "Card C's article title should be 'Article 5'."
    assert all(
        isinstance(article, ArticleSchema) for article in card_c.articles
    ), "All articles in Card C should be instances of ArticleSchema."

    assert card_d is not None and hasattr(
        card_d, "articles"
    ), "Card D should have nested articles."
    assert len(card_d.articles) == 0, "Card D should have no articles."


@pytest.mark.asyncio
async def test_get_multi_joined_nested_data_none_dict(async_session):
    clients = [
        Client(
            name="Client A",
            contact="Contact A",
            phone="111-111-1111",
            email="a@client.com",
        ),
        Client(
            name="Client B",
            contact="Contact B",
            phone="222-222-2222",
            email="b@client.com",
        ),
    ]
    async_session.add_all(clients)
    await async_session.flush()

    departments = [
        Department(name="Department A"),
        Department(name="Department B"),
    ]
    async_session.add_all(departments)
    await async_session.flush()

    users = [
        User(
            name="User A",
            username="usera",
            email="usera@example.com",
            phone="123-123-1234",
        ),
        User(
            name="User B",
            username="userb",
            email="userb@example.com",
            phone="234-234-2345",
        ),
    ]
    async_session.add_all(users)
    await async_session.flush()

    tasks = [
        Task(
            name="Task 1",
            description="Task 1 Description",
            client_id=clients[0].id,
            department_id=departments[0].id,
            assignee_id=users[0].id,
        ),
        Task(
            name="Task 2",
            description="Task 2 Description",
            client_id=clients[1].id,
            department_id=departments[1].id,
            assignee_id=users[1].id,
        ),
        Task(
            name="Task 3",
            description="Task 3 Description",
            client_id=None,
            department_id=None,
            assignee_id=None,
        ),
    ]
    async_session.add_all(tasks)
    await async_session.commit()

    task_crud = FastCRUD(Task)

    result = await task_crud.get_multi_joined(
        db=async_session,
        nest_joins=True,
        schema_to_select=TaskRead,
        joins_config=[
            JoinConfig(
                model=Client,
                join_on=Task.client_id == Client.id,
                join_prefix="client",
                schema_to_select=ClientRead,
                join_type="left",
            ),
            JoinConfig(
                model=Department,
                join_on=Task.department_id == Department.id,
                join_prefix="department",
                schema_to_select=DepartmentRead,
                join_type="left",
            ),
            JoinConfig(
                model=User,
                join_on=Task.assignee_id == User.id,
                join_prefix="assignee",
                schema_to_select=UserReadSub,
                join_type="left",
            ),
        ],
    )

    assert result is not None, "No data returned from the database."
    data = result["data"]
    assert len(data) == 3, "Expected three tasks."

    task1 = data[0]
    assert task1["client"] is not None, "Task 1 should have a client."
    assert task1["department"] is not None, "Task 1 should have a department."
    assert task1["assignee"] is not None, "Task 1 should have an assignee."

    task3 = data[2]
    assert task3["client"] is None, "Task 3 should have no client."
    assert task3["department"] is None, "Task 3 should have no department."
    assert task3["assignee"] is None, "Task 3 should have no assignee."


@pytest.mark.asyncio
async def test_get_multi_joined_no_duplicate_entries_in_one_to_many(async_session):
    """Test that one-to-many relationships don't produce duplicate entries when using multiple joins."""
    dept = Department(name="Test Department")
    async_session.add(dept)
    await async_session.flush()

    users = [
        User(
            name=f"User {i}",
            username=f"user{i}",
            email=f"user{i}@example.com",
            phone=f"{i}11-111-1111",
            department_id=dept.id,
        )
        for i in range(2)
    ]
    async_session.add_all(users)
    await async_session.flush()

    cards = []
    for user in users:
        for i in range(2):
            card = Card(title=f"Card {i} for {user.username}")
            cards.append(card)
    async_session.add_all(cards)
    await async_session.flush()

    articles = []
    for i, card in enumerate(cards):
        article = Article(title=f"Article for card {card.title}", card_id=card.id)
        articles.append(article)
    async_session.add_all(articles)
    await async_session.commit()

    dept_crud = FastCRUD(Department)
    result = await dept_crud.get_multi_joined(
        db=async_session,
        nest_joins=True,
        schema_to_select=DepartmentRead,
        joins_config=[
            JoinConfig(
                model=User,
                join_on=Department.id == User.department_id,
                join_prefix="users_",
                schema_to_select=UserReadSub,
                relationship_type="one-to-many",
            )
        ],
    )

    assert len(result["data"]) == 1, "Should return exactly one department"
    department = result["data"][0]

    assert len(department["users"]) == 2, "Should have exactly 2 users"


@pytest.mark.asyncio
async def test_get_multi_joined_explicit_join_preserves_condition(async_session):
    """Test that explicitly provided join conditions are preserved and used correctly."""
    dept = Department(name="Engineering")
    async_session.add(dept)
    await async_session.flush()

    users = [
        User(
            name="John",
            username="john",
            email="john@test.com",
            phone="123-456-7890",
            department_id=dept.id,
        ),
        User(
            name="Jane",
            username="jane",
            email="jane@test.com",
            phone="098-765-4321",
            department_id=dept.id,
        ),
    ]
    async_session.add_all(users)
    await async_session.flush()

    tasks = [
        Task(
            name="Task 1",
            description="First task",
            assignee_id=users[0].id,
            department_id=dept.id,
        ),
        Task(
            name="Task 2",
            description="Second task",
            assignee_id=users[1].id,
            department_id=dept.id,
        ),
        Task(
            name="Task 3",
            description="Unassigned task",
            assignee_id=None,
            department_id=dept.id,
        ),
    ]
    async_session.add_all(tasks)
    await async_session.commit()

    task_crud = FastCRUD(Task)

    result = await task_crud.get_multi_joined(
        db=async_session,
        nest_joins=True,
        joins_config=[
            JoinConfig(
                model=User,
                join_on=Task.assignee_id == User.id,
                join_prefix="assignee_",
                schema_to_select=UserReadSub,
                join_type="left",
            ),
            JoinConfig(
                model=Department,
                join_on=Task.department_id == Department.id,
                join_prefix="department_",
                schema_to_select=DepartmentRead,
                join_type="left",
            ),
        ],
    )

    tasks_data = result["data"]
    assigned_tasks = [t for t in tasks_data if t.get("assignee") is not None]
    unassigned_tasks = [t for t in tasks_data if t.get("assignee") is None]

    assert len(tasks_data) == 3, "Should return all three tasks"
    assert len(assigned_tasks) == 2, "Should have two assigned tasks"
    assert len(unassigned_tasks) == 1, "Should have one unassigned task"

    task1 = next((t for t in tasks_data if t["name"] == "Task 1"), None)
    task2 = next((t for t in tasks_data if t["name"] == "Task 2"), None)
    task3 = next((t for t in tasks_data if t["name"] == "Task 3"), None)

    assert task1 is not None, "Task 1 should exist"
    assert task2 is not None, "Task 2 should exist"
    assert task3 is not None, "Task 3 should exist"

    assert task1["assignee"]["username"] == "john", "Task 1 should be assigned to John"
    assert task2["assignee"]["username"] == "jane", "Task 2 should be assigned to Jane"
    assert task3["assignee"] is None, "Task 3 should have no assignee"

    assert (
        task1["department"]["name"] == "Engineering"
    ), "Task 1 should be in Engineering department"
    assert (
        task2["department"]["name"] == "Engineering"
    ), "Task 2 should be in Engineering department"
    assert (
        task3["department"]["name"] == "Engineering"
    ), "Task 3 should be in Engineering department"


@pytest.mark.asyncio
async def test_get_multi_joined_sorting_nested_items_one_to_many(async_session):
    cards = [
        Card(title="Card A"),
        Card(title="Card B"),
        Card(title="Card C"),
    ]
    async_session.add_all(cards)
    await async_session.flush()

    articles = [
        Article(title="Article 3", card_id=cards[0].id),
        Article(title="Article 1", card_id=cards[0].id),
        Article(title="Article 2", card_id=cards[0].id),
        Article(title="Article 2", card_id=cards[1].id),
        Article(title="Article 1", card_id=cards[1].id),
    ]
    async_session.add_all(articles)
    await async_session.commit()

    card_crud = FastCRUD(Card)

    result = await card_crud.get_multi_joined(
        db=async_session,
        nest_joins=True,
        joins_config=[
            JoinConfig(
                model=Article,
                join_on=Article.card_id == Card.id,
                join_prefix="articles_",
                join_type="left",
                relationship_type="one-to-many",
                sort_columns=["title"],
                sort_orders=["asc"],
            )
        ],
    )

    assert result is not None, "No data returned from the database."
    assert "data" in result, "Result should contain 'data' key."
    data = result["data"]
    assert isinstance(data, list), "Result data should be a list."
    assert len(data) == 3, "Expected three card records."

    card_a = next((c for c in data if c["id"] == cards[0].id), None)
    card_b = next((c for c in data if c["id"] == cards[1].id), None)
    card_c = next((c for c in data if c["id"] == cards[2].id), None)

    assert (
        card_a is not None and "articles" in card_a
    ), "Card A should have nested articles."
    assert len(card_a["articles"]) == 3, "Card A should have three articles."
    assert (
        card_a["articles"][0]["title"] == "Article 1"
    ), "Card A's first article title should be 'Article 1'."
    assert (
        card_a["articles"][1]["title"] == "Article 2"
    ), "Card A's second article title should be 'Article 2'."
    assert (
        card_a["articles"][2]["title"] == "Article 3"
    ), "Card A's third article title should be 'Article 3'."

    assert (
        card_b is not None and "articles" in card_b
    ), "Card B should have nested articles."
    assert len(card_b["articles"]) == 2, "Card B should have two articles."
    assert (
        card_b["articles"][0]["title"] == "Article 1"
    ), "Card B's first article title should be 'Article 1'."
    assert (
        card_b["articles"][1]["title"] == "Article 2"
    ), "Card B's second article title should be 'Article 2'."

    assert (
        card_c is not None and "articles" in card_c
    ), "Card C should have nested articles."
    assert len(card_c["articles"]) == 0, "Card C should have no articles."


@pytest.mark.asyncio
async def test_get_multi_joined_sorting_nested_items_many_to_many(async_session):
    project1 = Project(id=1, name="Project 1", description="First Project")
    project2 = Project(id=2, name="Project 2", description="Second Project")

    participant1 = Participant(id=1, name="Participant 3", role="Developer")
    participant2 = Participant(id=2, name="Participant 1", role="Designer")
    participant3 = Participant(id=3, name="Participant 2", role="Manager")

    async_session.add_all([project1, project2, participant1, participant2, participant3])
    await async_session.commit()

    projects_participants1 = ProjectsParticipantsAssociation(
        project_id=1, participant_id=1
    )
    projects_participants2 = ProjectsParticipantsAssociation(
        project_id=1, participant_id=2
    )
    projects_participants3 = ProjectsParticipantsAssociation(
        project_id=1, participant_id=3
    )
    projects_participants4 = ProjectsParticipantsAssociation(
        project_id=2, participant_id=1
    )
    projects_participants5 = ProjectsParticipantsAssociation(
        project_id=2, participant_id=2
    )

    async_session.add_all(
        [
            projects_participants1,
            projects_participants2,
            projects_participants3,
            projects_participants4,
            projects_participants5,
        ]
    )
    await async_session.commit()

    crud_project = FastCRUD(Project)

    join_condition_1 = Project.id == ProjectsParticipantsAssociation.project_id
    join_condition_2 = ProjectsParticipantsAssociation.participant_id == Participant.id

    joins_config = [
        JoinConfig(
            model=ProjectsParticipantsAssociation,
            join_on=join_condition_1,
            join_type="inner",
            join_prefix="pp_",
        ),
        JoinConfig(
            model=Participant,
            join_on=join_condition_2,
            join_type="inner",
            join_prefix="participant_",
            relationship_type="one-to-many",
            sort_columns=["name"],
            sort_orders=["asc"],
        ),
    ]

    records = await crud_project.get_multi_joined(
        db=async_session,
        nest_joins=True,
        joins_config=joins_config,
    )

    assert records is not None, "No data returned from the database."
    assert "data" in records, "Result should contain 'data' key."
    data = records["data"]
    assert isinstance(data, list), "Result data should be a list."
    assert len(data) == 2, "Expected two project records."

    project_1 = next((p for p in data if p["id"] == project1.id), None)
    project_2 = next((p for p in data if p["id"] == project2.id), None)

    assert (
        project_1 is not None and "participants" in project_1
    ), "Project 1 should have nested participants."
    assert len(project_1["participants"]) == 3, "Project 1 should have three participants."
    assert (
        project_1["participants"][0]["name"] == "Participant 1"
    ), "Project 1's first participant name should be 'Participant 1'."
    assert (
        project_1["participants"][1]["name"] == "Participant 2"
    ), "Project 1's second participant name should be 'Participant 2'."
    assert (
        project_1["participants"][2]["name"] == "Participant 3"
    ), "Project 1's third participant name should be 'Participant 3'."

    assert (
        project_2 is not None and "participants" in project_2
    ), "Project 2 should have nested participants."
    assert len(project_2["participants"]) == 2, "Project 2 should have two participants."
    assert (
        project_2["participants"][0]["name"] == "Participant 1"
    ), "Project 2's first participant name should be 'Participant 1'."
    assert (
        project_2["participants"][1]["name"] == "Participant 2"
    ), "Project 2's second participant name should be 'Participant 2'."
