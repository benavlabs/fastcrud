import pytest
from fastcrud import FastCRUD, CountConfig
from ...sqlalchemy.conftest import (
    Project,
    Participant,
    ProjectsParticipantsAssociation,
    Author,
    Article,
)


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
    
    async_session.add_all([project1, project2, project3, participant1, participant2, participant3])
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
    
    async_session.add_all([project1, participant1, participant2, participant3, participant4])
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
    from fastcrud import JoinConfig
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

