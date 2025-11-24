import pytest
from sqlalchemy import Column, Integer, String, ForeignKey, text
from sqlalchemy.orm import DeclarativeBase
from fastcrud import FastCRUD, CountConfig
from ...sqlalchemy.conftest import (
    Project,
    Participant,
    ProjectsParticipantsAssociation,
    Author,
    Article,
)


class Base(DeclarativeBase):
    pass


class CustomPKModel(Base):
    __tablename__ = "custom_pk_model"
    user_code = Column(String(10), primary_key=True)
    name = Column(String(50))


class RelatedModel(Base):
    __tablename__ = "related_model"
    id = Column(Integer, primary_key=True)
    user_code = Column(String(10), ForeignKey("custom_pk_model.user_code"))
    description = Column(String(100))


class CompositePKModel(Base):
    __tablename__ = "composite_pk_model"
    part_a = Column(String(10), primary_key=True)
    part_b = Column(Integer, primary_key=True)
    data = Column(String(50))


class CompositePKRelated(Base):
    __tablename__ = "composite_pk_related"
    id = Column(Integer, primary_key=True)
    part_a = Column(String(10))
    part_b = Column(Integer)
    info = Column(String(50))


@pytest.mark.asyncio
async def test_count_config_simple_many_to_many(async_session, test_data):
    """Test counting related objects through a many-to-many relationship."""
    # Create test data
    project1 = Project(id=1, name="Project Alpha", description="First project")
    project2 = Project(id=2, name="Project Beta", description="Second project")
    project3 = Project(id=3, name="Project Gamma", description="Third project")

    participant1 = Participant(id=1, name="Alice", role="Developer")
    participant2 = Participant(id=2, name="Bob", role="Designer")
    participant3 = Participant(id=3, name="Charlie", role="Developer")

    async_session.add_all(
        [project1, project2, project3, participant1, participant2, participant3]
    )
    await async_session.commit()

    # Create associations
    assoc1 = ProjectsParticipantsAssociation(project_id=1, participant_id=1)
    assoc2 = ProjectsParticipantsAssociation(project_id=1, participant_id=2)
    assoc3 = ProjectsParticipantsAssociation(project_id=1, participant_id=3)
    assoc4 = ProjectsParticipantsAssociation(project_id=2, participant_id=1)
    assoc5 = ProjectsParticipantsAssociation(project_id=2, participant_id=2)
    # Project 3 has no participants

    async_session.add_all([assoc1, assoc2, assoc3, assoc4, assoc5])
    await async_session.commit()

    # Test counting participants for each project
    project_crud = FastCRUD(Project)

    count_config = CountConfig(
        model=Participant,
        join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
        & (ProjectsParticipantsAssociation.project_id == Project.id),
        alias="participants_count",
    )

    result = await project_crud.get_multi_joined(
        db=async_session,
        counts_config=[count_config],
    )

    assert result["total_count"] == 3
    assert len(result["data"]) == 3

    # Find each project in the results
    projects_by_id = {p["id"]: p for p in result["data"]}

    assert projects_by_id[1]["participants_count"] == 3
    assert projects_by_id[2]["participants_count"] == 2
    assert projects_by_id[3]["participants_count"] == 0


@pytest.mark.asyncio
async def test_count_config_with_filters(async_session, test_data):
    """Test counting with filters applied to the count query."""
    # Create test data
    project1 = Project(id=1, name="Project Alpha", description="First project")

    participant1 = Participant(id=1, name="Alice", role="Developer")
    participant2 = Participant(id=2, name="Bob", role="Designer")
    participant3 = Participant(id=3, name="Charlie", role="Developer")

    async_session.add_all([project1, participant1, participant2, participant3])
    await async_session.commit()

    # Create associations
    assoc1 = ProjectsParticipantsAssociation(project_id=1, participant_id=1)
    assoc2 = ProjectsParticipantsAssociation(project_id=1, participant_id=2)
    assoc3 = ProjectsParticipantsAssociation(project_id=1, participant_id=3)

    async_session.add_all([assoc1, assoc2, assoc3])
    await async_session.commit()

    # Test counting only developers
    project_crud = FastCRUD(Project)

    count_config = CountConfig(
        model=Participant,
        join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
        & (ProjectsParticipantsAssociation.project_id == Project.id),
        alias="developers_count",
        filters={"role": "Developer"},
    )

    result = await project_crud.get_multi_joined(
        db=async_session,
        counts_config=[count_config],
    )

    assert result["total_count"] == 1
    assert len(result["data"]) == 1
    assert result["data"][0]["developers_count"] == 2  # Only Alice and Charlie


@pytest.mark.asyncio
async def test_count_config_multiple_counts(async_session, test_data):
    """Test multiple count configurations in a single query."""
    # Create test data
    project1 = Project(id=1, name="Project Alpha", description="First project")

    participant1 = Participant(id=1, name="Alice", role="Developer")
    participant2 = Participant(id=2, name="Bob", role="Designer")
    participant3 = Participant(id=3, name="Charlie", role="Developer")
    participant4 = Participant(id=4, name="Diana", role="Manager")

    async_session.add_all(
        [project1, participant1, participant2, participant3, participant4]
    )
    await async_session.commit()

    # Create associations
    assoc1 = ProjectsParticipantsAssociation(project_id=1, participant_id=1)
    assoc2 = ProjectsParticipantsAssociation(project_id=1, participant_id=2)
    assoc3 = ProjectsParticipantsAssociation(project_id=1, participant_id=3)
    assoc4 = ProjectsParticipantsAssociation(project_id=1, participant_id=4)

    async_session.add_all([assoc1, assoc2, assoc3, assoc4])
    await async_session.commit()

    # Test counting all participants and developers separately
    project_crud = FastCRUD(Project)

    all_participants_count = CountConfig(
        model=Participant,
        join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
        & (ProjectsParticipantsAssociation.project_id == Project.id),
        alias="all_participants_count",
    )

    developers_count = CountConfig(
        model=Participant,
        join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
        & (ProjectsParticipantsAssociation.project_id == Project.id),
        alias="developers_count",
        filters={"role": "Developer"},
    )

    result = await project_crud.get_multi_joined(
        db=async_session,
        counts_config=[all_participants_count, developers_count],
    )

    assert result["total_count"] == 1
    assert len(result["data"]) == 1
    assert result["data"][0]["all_participants_count"] == 4
    assert result["data"][0]["developers_count"] == 2


@pytest.mark.asyncio
async def test_count_config_one_to_many(async_session, test_data):
    """Test counting in a one-to-many relationship."""
    # Create test data
    author1 = Author(id=1, name="Author One")
    author2 = Author(id=2, name="Author Two")
    author3 = Author(id=3, name="Author Three")

    async_session.add_all([author1, author2, author3])
    await async_session.commit()

    # Create articles
    article1 = Article(id=1, title="Article 1", content="Content 1", author_id=1)
    article2 = Article(id=2, title="Article 2", content="Content 2", author_id=1)
    article3 = Article(id=3, title="Article 3", content="Content 3", author_id=1)
    article4 = Article(id=4, title="Article 4", content="Content 4", author_id=2)
    # Author 3 has no articles

    async_session.add_all([article1, article2, article3, article4])
    await async_session.commit()

    # Test counting articles for each author
    author_crud = FastCRUD(Author)

    count_config = CountConfig(
        model=Article,
        join_on=Article.author_id == Author.id,
        alias="articles_count",
    )

    result = await author_crud.get_multi_joined(
        db=async_session,
        counts_config=[count_config],
    )

    assert result["total_count"] == 3
    assert len(result["data"]) == 3

    # Find each author in the results
    authors_by_id = {a["id"]: a for a in result["data"]}

    assert authors_by_id[1]["articles_count"] == 3
    assert authors_by_id[2]["articles_count"] == 1
    assert authors_by_id[3]["articles_count"] == 0


@pytest.mark.asyncio
async def test_count_config_default_alias(async_session, test_data):
    """Test that default alias is generated from table name."""
    # Create test data
    author1 = Author(id=1, name="Author One")
    async_session.add(author1)
    await async_session.commit()

    article1 = Article(id=1, title="Article 1", content="Content 1", author_id=1)
    async_session.add(article1)
    await async_session.commit()

    # Test without specifying alias
    author_crud = FastCRUD(Author)

    count_config = CountConfig(
        model=Article,
        join_on=Article.author_id == Author.id,
        # No alias specified - should default to "articles_count"
    )

    result = await author_crud.get_multi_joined(
        db=async_session,
        counts_config=[count_config],
    )

    assert result["total_count"] == 1
    assert len(result["data"]) == 1
    assert "articles_count" in result["data"][0]
    assert result["data"][0]["articles_count"] == 1


@pytest.mark.asyncio
async def test_count_config_with_joins(async_session, test_data):
    """Test using counts_config alongside joins_config."""
    # Create test data
    project1 = Project(id=1, name="Project Alpha", description="First project")
    project2 = Project(id=2, name="Project Beta", description="Second project")

    participant1 = Participant(id=1, name="Alice", role="Developer")
    participant2 = Participant(id=2, name="Bob", role="Designer")

    async_session.add_all([project1, project2, participant1, participant2])
    await async_session.commit()

    # Project1 has 2 participants, Project2 has 1
    assoc1 = ProjectsParticipantsAssociation(project_id=1, participant_id=1)
    assoc2 = ProjectsParticipantsAssociation(project_id=1, participant_id=2)
    assoc3 = ProjectsParticipantsAssociation(project_id=2, participant_id=1)
    async_session.add_all([assoc1, assoc2, assoc3])
    await async_session.commit()

    # Test using both joins_config and counts_config

    project_crud = FastCRUD(Project)

    # We can use joins_config to get participant details and counts_config to get the count
    count_config = CountConfig(
        model=Participant,
        join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
        & (ProjectsParticipantsAssociation.project_id == Project.id),
        alias="participants_count",
    )

    result = await project_crud.get_multi_joined(
        db=async_session,
        counts_config=[count_config],
    )

    assert result["total_count"] == 2
    assert len(result["data"]) == 2

    # Find each project in the results
    projects_by_id = {p["id"]: p for p in result["data"]}

    assert projects_by_id[1]["participants_count"] == 2
    assert projects_by_id[2]["participants_count"] == 1


@pytest.mark.asyncio
async def test_count_config_custom_primary_key(async_session, test_data):
    """Test counting with models that have custom primary key names."""
    # Create tables for the test models
    await async_session.execute(
        text("""
        CREATE TABLE IF NOT EXISTS custom_pk_model (
            user_code VARCHAR(10) PRIMARY KEY,
            name VARCHAR(50)
        )
        """)
    )
    await async_session.execute(
        text("""
        CREATE TABLE IF NOT EXISTS related_model (
            id INTEGER PRIMARY KEY,
            user_code VARCHAR(10) REFERENCES custom_pk_model(user_code),
            description VARCHAR(100)
        )
        """)
    )
    await async_session.commit()

    # Insert test data using raw SQL
    await async_session.execute(
        text("""
        INSERT INTO custom_pk_model (user_code, name) VALUES 
        ('USER001', 'John Doe'),
        ('USER002', 'Jane Smith'),
        ('USER003', 'Bob Wilson')
        """)
    )
    await async_session.execute(
        text("""
        INSERT INTO related_model (user_code, description) VALUES 
        ('USER001', 'First item for USER001'),
        ('USER001', 'Second item for USER001'),
        ('USER001', 'Third item for USER001'),
        ('USER002', 'Item for USER002')
        """)
    )
    await async_session.commit()

    # Test counting related objects for model with custom primary key
    custom_crud = FastCRUD(CustomPKModel)

    count_config = CountConfig(
        model=RelatedModel,
        join_on=RelatedModel.user_code == CustomPKModel.user_code,
        alias="related_count",
    )

    result = await custom_crud.get_multi_joined(
        db=async_session,
        counts_config=[count_config],
    )

    assert result["total_count"] == 3
    assert len(result["data"]) == 3

    # Find each user in the results
    users_by_code = {u["user_code"]: u for u in result["data"]}

    assert users_by_code["USER001"]["related_count"] == 3
    assert users_by_code["USER002"]["related_count"] == 1
    assert users_by_code["USER003"]["related_count"] == 0


@pytest.mark.asyncio
async def test_count_config_composite_primary_key(async_session, test_data):
    """Test counting with models that have composite primary keys."""
    # Create tables for the test models
    await async_session.execute(
        text("""
        CREATE TABLE IF NOT EXISTS composite_pk_model (
            part_a VARCHAR(10),
            part_b INTEGER,
            data VARCHAR(50),
            PRIMARY KEY (part_a, part_b)
        )
        """)
    )
    await async_session.execute(
        text("""
        CREATE TABLE IF NOT EXISTS composite_pk_related (
            id INTEGER PRIMARY KEY,
            part_a VARCHAR(10),
            part_b INTEGER,
            info VARCHAR(50)
        )
        """)
    )
    await async_session.commit()

    # Insert test data using raw SQL
    await async_session.execute(
        text("""
        INSERT INTO composite_pk_model (part_a, part_b, data) VALUES 
        ('A', 1, 'Data A1'),
        ('A', 2, 'Data A2'),
        ('B', 1, 'Data B1')
        """)
    )
    await async_session.execute(
        text("""
        INSERT INTO composite_pk_related (part_a, part_b, info) VALUES 
        ('A', 1, 'Related to A1 - item 1'),
        ('A', 1, 'Related to A1 - item 2'),
        ('A', 2, 'Related to A2 - item 1'),
        ('B', 1, 'Related to B1 - item 1'),
        ('B', 1, 'Related to B1 - item 2'),
        ('B', 1, 'Related to B1 - item 3')
        """)
    )
    await async_session.commit()

    # Test counting related objects for model with composite primary key
    composite_crud = FastCRUD(CompositePKModel)

    count_config = CountConfig(
        model=CompositePKRelated,
        join_on=(CompositePKRelated.part_a == CompositePKModel.part_a)
        & (CompositePKRelated.part_b == CompositePKModel.part_b),
        alias="related_count",
    )

    result = await composite_crud.get_multi_joined(
        db=async_session,
        counts_config=[count_config],
    )

    assert result["total_count"] == 3
    assert len(result["data"]) == 3

    # Find each composite key record in the results
    records_by_key = {(r["part_a"], r["part_b"]): r for r in result["data"]}

    assert records_by_key[("A", 1)]["related_count"] == 2
    assert records_by_key[("A", 2)]["related_count"] == 1
    assert records_by_key[("B", 1)]["related_count"] == 3
