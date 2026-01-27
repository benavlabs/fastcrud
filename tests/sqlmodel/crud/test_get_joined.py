import pytest
from typing import Optional
from pydantic import BaseModel
from sqlalchemy import and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastcrud import FastCRUD, JoinConfig, aliased
from ...sqlmodel.conftest import (
    ModelTest,
    TierModel,
    CreateSchemaTest,
    TierSchemaTest,
    CategoryModel,
    CategorySchemaTest,
    BookingModel,
    BookingSchema,
    ReadSchemaTest,
    Article,
    Card,
    Client,
    Department,
    User,
    Task,
    ClientRead,
    DepartmentRead,
    UserReadSub,
    TaskRead,
)


# Schema that includes joined fields for testing
class JoinedTestTier(BaseModel):
    name: str
    tier_id: int
    tier_name: str


# Flattened schema for task with joined fields
class TaskWithJoinedData(BaseModel):
    id: int
    name: str
    description: str | None = None
    assignee_name: str | None = None
    department_name: str | None = None


@pytest.mark.asyncio
async def test_get_joined_basic(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
    )

    assert result is not None
    assert "name" in result
    assert "tier_name" in result


@pytest.mark.asyncio
async def test_get_joined_custom_condition(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    user_data_with_condition = [item for item in test_data if item["name"] == "Alice"]
    for user_item in user_data_with_condition:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_on=and_(ModelTest.tier_id == TierModel.id),
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        name="Alice",
    )

    assert result is not None
    assert result["name"] == "Alice"
    assert "tier_name" in result


@pytest.mark.asyncio
async def test_get_joined_with_prefix(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
    )

    assert result is not None
    assert "name" in result
    assert "tier_name" in result


@pytest.mark.asyncio
async def test_get_joined_different_join_types(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result_left = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_type="left",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
    )

    result_inner = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_type="inner",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
    )

    assert result_left is not None
    assert result_inner is not None


@pytest.mark.asyncio
async def test_get_joined_with_filters(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        name="Alice",
    )

    assert result is not None
    assert result["name"] == "Alice"


@pytest.mark.asyncio
async def test_update_multiple_records_allow_multiple(
    async_session, test_model, test_data
):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    await crud.update(
        db=async_session,
        object={"name": "Updated Name"},
        allow_multiple=True,
        name="Alice",
    )

    updated_records = await crud.get_multi(db=async_session, name="Updated Name")
    assert (
        len(updated_records["data"]) > 1
    ), "Should update multiple records when allow_multiple is True"


@pytest.mark.asyncio
async def test_count_with_advanced_filters(async_session, test_model, test_data):
    for item in test_data:
        async_session.add(test_model(**item))
    await async_session.commit()

    crud = FastCRUD(test_model)
    count_gt = await crud.count(async_session, id__gt=1)
    assert count_gt > 0, "Should count records with ID greater than 1"

    count_lt = await crud.count(async_session, id__lt=10)
    assert count_lt > 0, "Should count records with ID less than 10"


@pytest.mark.asyncio
async def test_get_joined_multiple_models(
    async_session, test_data, test_data_tier, test_data_category
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    await async_session.commit()

    for user_item in test_data:
        user_item_modified = user_item.copy()
        user_item_modified["category_id"] = 1
        async_session.add(ModelTest(**user_item_modified))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
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
        schema_to_select=CreateSchemaTest,
    )

    assert result is not None
    assert "name" in result
    assert "tier_name" in result
    assert "category_name" in result


@pytest.mark.asyncio
async def test_get_joined_with_aliases(
    async_session, test_data, test_data_tier, test_data_category, test_data_booking
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    for booking_item in test_data_booking:
        async_session.add(BookingModel(**booking_item))
    await async_session.commit()

    crud = FastCRUD(BookingModel)

    specific_booking_id = 1
    expected_owner_name = "Charlie"
    expected_user_name = "Alice"

    owner = aliased(ModelTest, name="owner")
    user = aliased(ModelTest, name="user")

    result = await crud.get_joined(
        db=async_session,
        schema_to_select=BookingSchema,
        joins_config=[
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.owner_id == owner.id,
                join_prefix="owner_",
                alias=owner,
                schema_to_select=ReadSchemaTest,
            ),
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.user_id == user.id,
                join_prefix="user_",
                alias=user,
                schema_to_select=ReadSchemaTest,
            ),
        ],
        id=specific_booking_id,
    )

    assert result is not None
    assert (
        result["owner_name"] == expected_owner_name
    ), "Owner name does not match expected value"
    assert (
        result["user_name"] == expected_user_name
    ), "User name does not match expected value"


@pytest.mark.asyncio
async def test_get_joined_with_aliases_no_schema(
    async_session, test_data, test_data_tier, test_data_category, test_data_booking
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    for booking_item in test_data_booking:
        async_session.add(BookingModel(**booking_item))
    await async_session.commit()

    crud = FastCRUD(BookingModel)

    specific_booking_id = 1
    expected_owner_name = "Charlie"
    expected_user_name = "Alice"

    owner = aliased(ModelTest, name="owner")
    user = aliased(ModelTest, name="user")

    result = await crud.get_joined(
        db=async_session,
        schema_to_select=BookingSchema,
        joins_config=[
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.owner_id == owner.id,
                join_prefix="owner_",
                alias=owner,
            ),
            JoinConfig(
                model=ModelTest,
                join_on=BookingModel.user_id == user.id,
                join_prefix="user_",
                alias=user,
            ),
        ],
        id=specific_booking_id,
    )

    assert result is not None
    assert (
        result["owner_name"] == expected_owner_name
    ), "Owner name does not match expected value"
    assert (
        result["user_name"] == expected_user_name
    ), "User name does not match expected value"


@pytest.mark.asyncio
async def test_get_joined_with_both_single_and_joins_config_raises_value_error(
    async_session, test_data
):
    crud = FastCRUD(ModelTest)

    with pytest.raises(ValueError) as excinfo:
        await crud.get_joined(
            db=async_session,
            join_model=TierModel,
            joins_config=[
                JoinConfig(
                    model=TierModel,
                    join_on=ModelTest.tier_id == TierModel.id,
                )
            ],
        )

    assert (
        "Cannot use both single join parameters and joins_config simultaneously."
        in str(excinfo.value)
    )


@pytest.mark.asyncio
async def test_get_joined_without_join_model_or_joins_config_raises_value_error(
    async_session, test_data
):
    crud = FastCRUD(ModelTest)

    with pytest.raises(ValueError) as excinfo:
        await crud.get_joined(db=async_session)

    assert (
        "You need one of join_model, joins_config, or auto_detect_relationships."
        in str(excinfo.value)
    )


@pytest.mark.asyncio
async def test_get_joined_with_unsupported_join_type_raises_value_error(
    async_session, test_data
):
    crud = FastCRUD(ModelTest)

    with pytest.raises(ValueError) as excinfo:
        await crud.get_joined(
            db=async_session,
            join_model=TierModel,
            join_type="unsupported_type",
        )

    assert "Unsupported join type" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_joined_returns_none_when_no_record_matches(async_session, test_data):
    crud = FastCRUD(ModelTest)

    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        name="Nonexistent Name",
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_joined_with_joined_model_filters(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    join_filters = {"name": "Premium"}

    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_filters=join_filters,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        join_prefix="tier_",
    )

    assert result is not None, "Expected to find at least one matching record"
    assert (
        result["tier_name"] == "Premium"
    ), "Expected joined record to meet the filter criteria"


@pytest.mark.asyncio
async def test_get_joined_nest_joins(async_session, test_data, test_data_tier):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        nest_joins=True,
    )

    assert result is not None, "No result returned, expected a nested join result."
    assert "name" in result, "Expected primary model field 'name' in result."
    assert "tier" in result, "Expected nested 'tier' key in result for joined fields."
    assert (
        "name" in result["tier"]
    ), "Expected 'name' inside nested 'tier' dictionary from TierModel."
    assert (
        "tier_name" not in result
    ), "'tier_name' should not be at the top level in the result."


@pytest.mark.asyncio
async def test_get_joined_nested_no_prefix_provided(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
        join_prefix="tier_",
        nest_joins=True,
    )

    assert result is not None, "No result returned, expected a nested join result."
    assert "name" in result, "Expected primary model field 'name' in result."
    assert (
        "name" in result["tier"]
    ), "Expected 'name' field inside nested 'tier' dictionary."


@pytest.mark.asyncio
async def test_get_joined_no_prefix_no_nesting(
    async_session, test_data, test_data_tier
):
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)

    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        schema_to_select=CreateSchemaTest,
        join_schema_to_select=TierSchemaTest,
    )

    assert result is not None, "Expected to retrieve a result from joined query."
    assert (
        "name" in result
    ), "Expected 'name' field from the primary model in the result."
    assert "tier_id" in result, "Expected 'tier_id' (foreign key) in the result."
    assert (
        "tier_name" not in result
    ), "Field 'tier_name' should not exist unless specifically prefixed or nested."


@pytest.mark.asyncio
async def test_get_joined_card_with_articles(async_session):
    card = Card(title="Test Card")
    async_session.add(card)
    async_session.add_all(
        [
            Article(title="Article 1", card=card),
            Article(title="Article 2", card=card),
            Article(title="Article 3", card=card),
        ]
    )
    await async_session.commit()

    card_crud = FastCRUD(Card)

    result = await card_crud.get_joined(
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
    assert "title" in result, "Card title should be present in the result."
    assert "articles" in result, "Articles should be nested under 'articles'."
    assert isinstance(result["articles"], list), "Articles should be a list."
    assert len(result["articles"]) == 3, "There should be three articles."
    assert all(
        "title" in article for article in result["articles"]
    ), "Each article should have a title."


@pytest.mark.asyncio
async def test_get_joined_nested_data_none_dict(async_session):
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

    joins_config = [
        JoinConfig(
            model=Client,
            join_on=Task.client_id == Client.id,
            join_prefix="client_",
            schema_to_select=ClientRead,
            join_type="left",
        ),
        JoinConfig(
            model=Department,
            join_on=Task.department_id == Department.id,
            join_prefix="department_",
            schema_to_select=DepartmentRead,
            join_type="left",
        ),
        JoinConfig(
            model=User,
            join_on=Task.assignee_id == User.id,
            join_prefix="assignee_",
            schema_to_select=UserReadSub,
            join_type="left",
        ),
    ]

    task1_result = await task_crud.get_joined(
        db=async_session,
        id=tasks[0].id,
        schema_to_select=TaskRead,
        joins_config=joins_config,
        nest_joins=True,
    )

    assert task1_result is not None, "No data returned from the database."
    assert (
        "client" in task1_result
    ), "Nested client data should be present under key 'client'"
    assert (
        "department" in task1_result
    ), "Nested department data should be present under key 'department'"
    assert (
        "assignee" in task1_result
    ), "Nested assignee data should be present under key 'assignee'"
    assert task1_result["client"] is not None, "Task 1 should have a client."
    assert task1_result["department"] is not None, "Task 1 should have a department."
    assert task1_result["assignee"] is not None, "Task 1 should have an assignee."

    task3_result = await task_crud.get_joined(
        db=async_session,
        id=tasks[2].id,
        schema_to_select=TaskRead,
        joins_config=joins_config,
        nest_joins=True,
    )

    assert task3_result is not None, "No data returned from the database."
    assert (
        "client" in task3_result
    ), "Nested client data should be present under key 'client'"
    assert (
        "department" in task3_result
    ), "Nested department data should be present under key 'department'"
    assert (
        "assignee" in task3_result
    ), "Nested assignee data should be present under key 'assignee'"
    assert task3_result["client"] is None, "Task 3 should have no client."
    assert task3_result["department"] is None, "Task 3 should have no department."
    assert task3_result["assignee"] is None, "Task 3 should have no assignee."


@pytest.mark.asyncio
async def test_get_joined_return_as_model_true_sqlmodel(
    async_session, test_data, test_data_tier
):
    """Test get_joined with return_as_model=True returns Pydantic model instance (SQLModel)."""
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=JoinedTestTier,
        join_schema_to_select=TierSchemaTest,
        return_as_model=True,
    )

    assert result is not None
    assert isinstance(
        result, JoinedTestTier
    ), "Result should be a JoinedTestTier Pydantic model instance"
    assert hasattr(result, "name"), "Result should have name attribute"
    assert hasattr(result, "tier_name"), "Result should have tier_name attribute"
    assert result.tier_name is not None, "tier_name should have a value"


@pytest.mark.asyncio
async def test_get_joined_return_as_model_false_sqlmodel(
    async_session, test_data, test_data_tier
):
    """Test get_joined with return_as_model=False returns dict (default behavior, SQLModel)."""
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        join_model=TierModel,
        join_prefix="tier_",
        schema_to_select=ReadSchemaTest,
        join_schema_to_select=TierSchemaTest,
        return_as_model=False,
    )

    assert result is not None
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "name" in result
    assert "tier_name" in result


@pytest.mark.asyncio
async def test_get_joined_return_as_model_without_schema_raises_error_sqlmodel(
    async_session, test_data, test_data_tier
):
    """Test get_joined with return_as_model=True but no schema raises ValueError (SQLModel)."""
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError) as exc_info:
        await crud.get_joined(
            db=async_session,
            join_model=TierModel,
            join_prefix="tier_",
            return_as_model=True,  # Missing schema_to_select
        )

    assert "schema_to_select must be provided when return_as_model is True" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_get_joined_return_as_model_with_joins_config_sqlmodel(async_session):
    """Test get_joined with return_as_model=True using joins_config (SQLModel)."""
    # Create test data
    department = Department(name="Engineering")
    async_session.add(department)
    await async_session.flush()

    user = User(
        name="John Doe",
        username="john",
        email="john@example.com",
        department_id=department.id,
    )
    async_session.add(user)
    await async_session.flush()

    task = Task(
        name="Test Task",
        description="Test Task",
        assignee_id=user.id,
        department_id=department.id,
    )
    async_session.add(task)
    await async_session.commit()

    # Test with joins_config and return_as_model=True
    task_crud = FastCRUD(Task)
    joins_config = [
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
    ]

    result = await task_crud.get_joined(
        db=async_session,
        id=task.id,
        schema_to_select=TaskWithJoinedData,
        joins_config=joins_config,
        return_as_model=True,
    )

    assert result is not None
    assert isinstance(
        result, TaskWithJoinedData
    ), "Result should be a TaskWithJoinedData Pydantic model instance"
    assert hasattr(result, "description"), "Result should have description attribute"
    assert result.description == "Test Task"
    assert hasattr(
        result, "assignee_name"
    ), "Result should have assignee_name attribute"
    assert result.assignee_name == "John Doe"


# Tests for auto_detect_relationships parameter
@pytest.mark.asyncio
async def test_get_joined_auto_detect_relationships_sqlmodel(
    async_session, test_data, test_data_tier, test_data_category
):
    """Test auto-detecting all relationships (SQLModel)."""
    # Setup tier and category data
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    for category_item in test_data_category:
        async_session.add(CategoryModel(**category_item))
    await async_session.commit()

    # Add test data
    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        schema_to_select=ReadSchemaTest,
        auto_detect_relationships=True,
        nest_joins=True,
        name="Alice",
    )

    assert result is not None
    assert "name" in result
    assert result["name"] == "Alice"
    # Should have tier relationship data (nested)
    assert "tier" in result or "tier_id" in result


@pytest.mark.asyncio
async def test_get_joined_auto_detect_with_manual_error_sqlmodel(async_session):
    """Test that auto_detect + manual joins raises error (SQLModel)."""
    crud = FastCRUD(ModelTest)
    with pytest.raises(ValueError, match="Cannot use auto_detect_relationships"):
        await crud.get_joined(
            db=async_session,
            join_model=TierModel,
            auto_detect_relationships=True,
            id=1,
        )


@pytest.mark.asyncio
async def test_get_joined_auto_detect_no_relationships_sqlmodel(
    async_session: AsyncSession,
) -> None:
    """Test that auto_detect gracefully handles models with no relationships (SQLModel)."""
    from ...sqlmodel.conftest import ModelWithCustomColumns

    class CustomSchema(BaseModel):
        id: int
        meta: str
        name: str

    # ModelWithCustomColumns has NO relationships defined
    custom = ModelWithCustomColumns(id=1, meta="test", name="Test Name")
    async_session.add(custom)
    await async_session.commit()
    await async_session.refresh(custom)

    crud: FastCRUD = FastCRUD(ModelWithCustomColumns)  # type: ignore[type-arg]
    # Should work without error - falls back to regular get()
    result = await crud.get_joined(
        db=async_session,
        schema_to_select=CustomSchema,
        auto_detect_relationships=True,
        id=custom.id,
    )

    assert result is not None
    assert result["id"] == custom.id
    assert result["name"] == custom.name


@pytest.mark.asyncio
async def test_get_joined_auto_detect_with_nest_joins_sqlmodel(
    async_session, test_data, test_data_tier
):
    """Test auto-detection with nested joins enabled (SQLModel)."""
    for tier_item in test_data_tier:
        async_session.add(TierModel(**tier_item))
    await async_session.commit()

    for user_item in test_data:
        async_session.add(ModelTest(**user_item))
    await async_session.commit()

    crud = FastCRUD(ModelTest)
    result = await crud.get_joined(
        db=async_session,
        schema_to_select=ReadSchemaTest,
        auto_detect_relationships=True,
        nest_joins=True,
        name="Alice",
    )

    assert result is not None
    assert "name" in result
    # With nest_joins, joined data should be nested
    assert "tier" in result or "category" in result
