"""
Test OR filter across joined tables in get_multi_joined().

This test verifies that the `_or` parameter works correctly with dot notation
for filtering across both the main model and joined tables.

The user's desired SQL behavior:
    WHERE
        task.name ILIKE '%abc%'
     OR task.description ILIKE '%abc%'
     OR client.name ILIKE '%abc%'
     OR department.name ILIKE '%abc%'
"""

import pytest
from fastcrud import FastCRUD, JoinConfig
from ...sqlalchemy.conftest import (
    Task,
    Client,
    Department,
)


def get_joins_config():
    """Helper to create JoinConfig with explicit join conditions."""
    return [
        JoinConfig(
            model=Client,
            join_on=Task.client_id == Client.id,
            join_prefix="client_",
        ),
        JoinConfig(
            model=Department,
            join_on=Task.department_id == Department.id,
            join_prefix="department_",
        ),
    ]


@pytest.mark.asyncio
async def test_or_filter_across_joined_tables(async_session):
    """Test OR filter with dot notation for joined table fields."""
    # Create test data
    client1 = Client(
        name="Acme Corporation",
        contact="John Doe",
        phone="555-0001",
        email="acme@example.com",
    )
    client2 = Client(
        name="Tech Solutions",
        contact="Jane Smith",
        phone="555-0002",
        email="tech@example.com",
    )
    client3 = Client(
        name="Global Industries",
        contact="Bob Wilson",
        phone="555-0003",
        email="global@example.com",
    )

    async_session.add_all([client1, client2, client3])
    await async_session.commit()

    dept1 = Department(name="Engineering")
    dept2 = Department(name="Marketing")
    dept3 = Department(name="Sales")

    async_session.add_all([dept1, dept2, dept3])
    await async_session.commit()

    # Refresh to get IDs
    await async_session.refresh(client1)
    await async_session.refresh(client2)
    await async_session.refresh(client3)
    await async_session.refresh(dept1)
    await async_session.refresh(dept2)
    await async_session.refresh(dept3)

    # Create tasks with different combinations
    task1 = Task(
        name="Build API",
        description="Create REST endpoints",
        client_id=client1.id,
        department_id=dept1.id,
    )
    task2 = Task(
        name="Design Logo",
        description="Brand identity work",
        client_id=client2.id,
        department_id=dept2.id,
    )
    task3 = Task(
        name="Sales Report",
        description="Quarterly analysis",
        client_id=client3.id,
        department_id=dept3.id,
    )
    task4 = Task(
        name="Tech Review",
        description="Code audit",
        client_id=client1.id,
        department_id=dept1.id,
    )

    async_session.add_all([task1, task2, task3, task4])
    await async_session.commit()

    crud = FastCRUD(Task)

    # Test 1: Search for "Tech" - should match:
    # - task2 via client.name ("Tech Solutions")
    # - task4 via task.name ("Tech Review")
    result = await crud.get_multi_joined(
        db=async_session,
        joins_config=get_joins_config(),
        _or={
            "name__ilike": "%Tech%",
            "description__ilike": "%Tech%",
            "client.name__ilike": "%Tech%",
            "department.name__ilike": "%Tech%",
        },
    )

    assert result["total_count"] == 2
    task_names = [item["name"] for item in result["data"]]
    assert "Tech Review" in task_names  # matches task.name
    assert "Design Logo" in task_names  # matches client.name "Tech Solutions"


@pytest.mark.asyncio
async def test_or_filter_search_keyword_across_models(async_session):
    """Test global search pattern across main and joined models."""
    # Create test data
    client = Client(
        name="ABC Company",
        contact="Contact Person",
        phone="555-1234",
        email="abc@example.com",
    )
    async_session.add(client)
    await async_session.commit()
    await async_session.refresh(client)

    dept = Department(name="Research ABC")
    async_session.add(dept)
    await async_session.commit()
    await async_session.refresh(dept)

    # Task with "ABC" in name
    task1 = Task(
        name="ABC Project",
        description="Some work",
        client_id=client.id,
        department_id=dept.id,
    )
    # Task with "ABC" in description
    task2 = Task(
        name="Other Task",
        description="ABC related work",
        client_id=client.id,
        department_id=dept.id,
    )
    # Task without "ABC" but client has "ABC"
    task3 = Task(
        name="Simple Task",
        description="Nothing special",
        client_id=client.id,  # client.name = "ABC Company"
        department_id=dept.id,
    )

    async_session.add_all([task1, task2, task3])
    await async_session.commit()

    crud = FastCRUD(Task)

    # Search for "ABC" across all fields
    keyword = "ABC"
    result = await crud.get_multi_joined(
        db=async_session,
        joins_config=get_joins_config(),
        _or={
            "name__ilike": f"%{keyword}%",
            "description__ilike": f"%{keyword}%",
            "client.name__ilike": f"%{keyword}%",
            "department.name__ilike": f"%{keyword}%",
        },
    )

    # All 3 tasks should match because:
    # - task1: name contains "ABC"
    # - task2: description contains "ABC"
    # - task3: client.name contains "ABC" OR department.name contains "ABC"
    assert result["total_count"] == 3


@pytest.mark.asyncio
async def test_or_filter_combined_with_regular_filter(async_session):
    """Test _or filter combined with regular AND filters."""
    # Create test data
    client1 = Client(
        name="Alpha Corp", contact="A", phone="111", email="alpha@example.com"
    )
    client2 = Client(
        name="Beta Corp", contact="B", phone="222", email="beta@example.com"
    )
    async_session.add_all([client1, client2])
    await async_session.commit()
    await async_session.refresh(client1)
    await async_session.refresh(client2)

    dept = Department(name="Engineering")
    async_session.add(dept)
    await async_session.commit()
    await async_session.refresh(dept)

    # Tasks for client1
    task1 = Task(
        name="Alpha Task",
        description="Work for Alpha",
        client_id=client1.id,
        department_id=dept.id,
    )
    task2 = Task(
        name="Other Alpha",
        description="More work",
        client_id=client1.id,
        department_id=dept.id,
    )

    # Tasks for client2
    task3 = Task(
        name="Alpha Related",  # Has "Alpha" in name but different client
        description="Beta work",
        client_id=client2.id,
        department_id=dept.id,
    )

    async_session.add_all([task1, task2, task3])
    await async_session.commit()

    crud = FastCRUD(Task)

    # Search for "Alpha" but only for client1
    result = await crud.get_multi_joined(
        db=async_session,
        joins_config=get_joins_config(),
        client_id=client1.id,  # AND filter
        _or={
            "name__ilike": "%Alpha%",
            "description__ilike": "%Alpha%",
        },
    )

    # Should only return tasks from client1 that match "Alpha"
    assert result["total_count"] == 2
    for item in result["data"]:
        assert item["client_id"] == client1.id


@pytest.mark.asyncio
async def test_or_filter_no_matches(async_session):
    """Test _or filter when no records match."""
    client = Client(
        name="Test Client", contact="Test", phone="000", email="test@example.com"
    )
    async_session.add(client)
    await async_session.commit()
    await async_session.refresh(client)

    dept = Department(name="Test Dept")
    async_session.add(dept)
    await async_session.commit()
    await async_session.refresh(dept)

    task = Task(
        name="Regular Task",
        description="Normal description",
        client_id=client.id,
        department_id=dept.id,
    )
    async_session.add(task)
    await async_session.commit()

    crud = FastCRUD(Task)

    # Search for something that doesn't exist
    result = await crud.get_multi_joined(
        db=async_session,
        joins_config=get_joins_config(),
        _or={
            "name__ilike": "%NONEXISTENT%",
            "client.name__ilike": "%NONEXISTENT%",
            "department.name__ilike": "%NONEXISTENT%",
        },
    )

    assert result["total_count"] == 0
    assert len(result["data"]) == 0


@pytest.mark.asyncio
async def test_or_filter_with_multiple_operators(async_session):
    """Test _or filter with different operators across joined tables."""
    client = Client(
        name="Premium Client", contact="VIP", phone="999", email="premium@example.com"
    )
    async_session.add(client)
    await async_session.commit()
    await async_session.refresh(client)

    dept1 = Department(name="Dept A")
    dept2 = Department(name="Dept B")
    async_session.add_all([dept1, dept2])
    await async_session.commit()
    await async_session.refresh(dept1)
    await async_session.refresh(dept2)

    task1 = Task(
        name="Priority Task",
        description="Urgent",
        client_id=client.id,
        department_id=dept1.id,
    )
    task2 = Task(
        name="Normal Task",
        description="Regular",
        client_id=client.id,
        department_id=dept2.id,
    )

    async_session.add_all([task1, task2])
    await async_session.commit()

    crud = FastCRUD(Task)

    # Mix of operators: startswith on main model, exact match on joined
    result = await crud.get_multi_joined(
        db=async_session,
        joins_config=get_joins_config(),
        _or={
            "name__startswith": "Priority",
            "department.name__eq": "Dept B",
        },
    )

    # Should match:
    # - task1: name starts with "Priority"
    # - task2: department.name == "Dept B"
    assert result["total_count"] == 2
