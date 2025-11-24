import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastcrud import EndpointCreator, FilterConfig
from tests.sqlalchemy.conftest import (
    Company,
    UserModel,
    CompanySchema,
    UserModelSchema,
)


@pytest.fixture
@pytest.mark.asyncio
async def setup_test_data(async_session):
    """Set up test data for joined model filtering tests."""
    # Create companies
    company1 = Company(name="TechCorp", industry="Technology")
    company2 = Company(name="HealthInc", industry="Healthcare")
    company3 = Company(name="FinanceGroup", industry="Finance")

    async_session.add_all([company1, company2, company3])
    await async_session.commit()

    # Create users
    users = [
        UserModel(
            name="Alice Johnson", email="alice@techcorp.com", company_id=company1.id
        ),
        UserModel(name="Bob Smith", email="bob@techcorp.com", company_id=company1.id),
        UserModel(
            name="Charlie Brown", email="charlie@healthinc.com", company_id=company2.id
        ),
        UserModel(
            name="Diana Prince", email="diana@financegroup.com", company_id=company3.id
        ),
        UserModel(name="Eve Wilson", email="eve@techcorp.com", company_id=company1.id),
    ]

    async_session.add_all(users)
    await async_session.commit()

    return {"companies": [company1, company2, company3], "users": users}


@pytest.fixture
def user_app_with_joined_filters(async_session) -> FastAPI:
    """Create FastAPI app with UserModel endpoints that support joined model filtering."""
    app = FastAPI()

    endpoint_creator = EndpointCreator(
        session=lambda: async_session,
        model=UserModel,
        create_schema=UserModelSchema,
        update_schema=UserModelSchema,
        path="/users",
        filter_config=FilterConfig(
            # Regular filters
            name=None,
            email=None,
            # Joined model filters
            **{"company.name": None, "company.industry": None},
        ),
    )
    endpoint_creator.add_routes_to_router(included_methods=["read_multi"])
    app.include_router(endpoint_creator.router)
    return app


@pytest.fixture
def company_app_with_joined_filters(async_session) -> FastAPI:
    """Create FastAPI app with Company endpoints that support joined model filtering."""
    app = FastAPI()

    endpoint_creator = EndpointCreator(
        session=lambda: async_session,
        model=Company,
        create_schema=CompanySchema,
        update_schema=CompanySchema,
        path="/companies",
        filter_config=FilterConfig(
            # Regular filters
            name=None,
            industry=None,
            # Joined model filters - this would require more complex logic for one-to-many
            # For now, we'll focus on many-to-one relationships
        ),
    )
    endpoint_creator.add_routes_to_router(included_methods=["read_multi"])
    app.include_router(endpoint_creator.router)
    return app


@pytest.mark.asyncio
async def test_filter_users_by_company_name(
    user_app_with_joined_filters, setup_test_data
):
    """Test filtering users by their company name."""
    # Await the setup_test_data fixture
    await setup_test_data
    client = TestClient(user_app_with_joined_filters)

    # First, let's test a simple request without filters to make sure the endpoint works
    response = client.get("/users")
    print(f"Simple request status: {response.status_code}")
    if response.status_code != 200:
        print(f"Simple request error: {response.text}")

    # Now test with the joined filter
    response = client.get("/users?company.name=TechCorp")
    print(f"Joined filter status: {response.status_code}")
    if response.status_code != 200:
        print(f"Joined filter error: {response.text}")

    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    users = data["data"]

    # Should return 3 users from TechCorp
    assert len(users) == 3
    for user in users:
        # All users should be from TechCorp (company_id should be 1)
        assert user["company_id"] == 1


@pytest.mark.asyncio
async def test_filter_users_by_company_industry(
    user_app_with_joined_filters, setup_test_data
):
    """Test filtering users by their company industry."""
    # Await the setup_test_data fixture
    await setup_test_data
    client = TestClient(user_app_with_joined_filters)

    # Filter users by company industry "Healthcare"
    response = client.get("/users?company.industry=Healthcare")
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    users = data["data"]

    # Should return 1 user from HealthInc
    assert len(users) == 1
    assert users[0]["name"] == "Charlie Brown"
    assert users[0]["company_id"] == 2


@pytest.mark.asyncio
async def test_filter_users_by_multiple_joined_filters(
    user_app_with_joined_filters, setup_test_data
):
    """Test filtering users by multiple joined model filters."""
    # Await the setup_test_data fixture
    await setup_test_data
    client = TestClient(user_app_with_joined_filters)

    # Filter users by company name and industry
    response = client.get("/users?company.name=TechCorp&company.industry=Technology")
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    users = data["data"]

    # Should return 3 users from TechCorp in Technology industry
    assert len(users) == 3
    for user in users:
        assert user["company_id"] == 1


@pytest.mark.asyncio
async def test_filter_users_combined_regular_and_joined(
    user_app_with_joined_filters, setup_test_data
):
    """Test filtering users by combining regular and joined model filters."""
    # Await the setup_test_data fixture
    await setup_test_data
    client = TestClient(user_app_with_joined_filters)

    # Filter by user name and company name
    response = client.get("/users?name=Alice Johnson&company.name=TechCorp")
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    users = data["data"]

    # Should return 1 user: Alice Johnson from TechCorp
    assert len(users) == 1
    assert users[0]["name"] == "Alice Johnson"
    assert users[0]["company_id"] == 1


@pytest.mark.asyncio
async def test_filter_users_no_matches(user_app_with_joined_filters, setup_test_data):
    """Test filtering users with no matching results."""
    # Await the setup_test_data fixture
    await setup_test_data
    client = TestClient(user_app_with_joined_filters)

    # Filter by non-existent company name
    response = client.get("/users?company.name=NonExistentCorp")
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    users = data["data"]

    # Should return no users
    assert len(users) == 0


@pytest.mark.asyncio
async def test_filter_config_validation_valid_joined_filter():
    """Test that FilterConfig properly validates valid joined model filters."""
    # This should not raise an exception
    filter_config = FilterConfig(
        **{"company.name": None, "company.industry": "Technology"}
    )

    assert "company.name" in filter_config.filters
    assert "company.industry" in filter_config.filters
    assert filter_config.is_joined_filter("company.name")
    assert filter_config.is_joined_filter("company.industry")


@pytest.mark.asyncio
async def test_filter_config_parse_joined_filter():
    """Test that FilterConfig properly parses joined model filters."""
    filter_config = FilterConfig()

    # Test simple joined filter
    relationship_path, final_field, operator = filter_config.parse_joined_filter(
        "company.name"
    )
    assert relationship_path == ["company"]
    assert final_field == "name"
    assert operator is None

    # Test joined filter with operator
    relationship_path, final_field, operator = filter_config.parse_joined_filter(
        "company.name__eq"
    )
    assert relationship_path == ["company"]
    assert final_field == "name"
    assert operator == "eq"

    # Test nested joined filter
    relationship_path, final_field, operator = filter_config.parse_joined_filter(
        "company.department.name"
    )
    assert relationship_path == ["company", "department"]
    assert final_field == "name"
    assert operator is None


def test_filter_config_invalid_joined_filter():
    """Test that FilterConfig raises error for invalid joined model filters."""
    filter_config = FilterConfig()

    # Test invalid format (no dot)
    with pytest.raises(ValueError, match="Invalid joined filter format"):
        filter_config.parse_joined_filter("invalid_format")
