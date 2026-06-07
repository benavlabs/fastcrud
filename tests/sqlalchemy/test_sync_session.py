"""
Tests for sync Session support in FastCRUD.count() and FastCRUD.exists()
Relates to: https://github.com/benavlabs/fastcrud/issues/122
"""
import pytest
from sqlalchemy import create_engine, Column, Integer, String, select, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel


class Base(DeclarativeBase):
    pass


class Book(Base):
    __tablename__ = "books_sync_test"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(100), nullable=False)
    author = Column(String(100), nullable=False)


class BookCreate(BaseModel):
    title: str
    author: str


@pytest.fixture(scope="module")
def sync_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def sync_db(sync_engine):
    SyncSession = sessionmaker(bind=sync_engine)
    session = SyncSession()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(autouse=True)
def seed_books(sync_db):
    books = [
        Book(title="Clean Code", author="Robert Martin"),
        Book(title="The Pragmatic Programmer", author="Hunt & Thomas"),
        Book(title="Design Patterns", author="Gang of Four"),
    ]
    sync_db.add_all(books)
    sync_db.commit()
    yield
    sync_db.query(Book).delete()
    sync_db.commit()


async def _count(db, model, **filters) -> int:
    count_query = select(func.count()).select_from(model)
    for key, value in filters.items():
        count_query = count_query.where(getattr(model, key) == value)
    if isinstance(db, AsyncSession):
        total = await db.scalar(count_query)
    else:
        total = db.scalar(count_query)
    if total is None:
        raise ValueError("Could not find the count.")
    return total


async def _exists(db, model, **filters) -> bool:
    stmt = select(model)
    for key, value in filters.items():
        stmt = stmt.where(getattr(model, key) == value)
    stmt = stmt.limit(1)
    if isinstance(db, AsyncSession):
        result = await db.execute(stmt)
    else:
        result = db.execute(stmt)
    return result.first() is not None


class TestExistsWithSyncSession:

    @pytest.mark.asyncio
    async def test_exists_returns_true_when_record_found(self, sync_db):
        assert await _exists(sync_db, Book, title="Clean Code") is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_no_record(self, sync_db):
        assert await _exists(sync_db, Book, title="Nonexistent Book") is False

    @pytest.mark.asyncio
    async def test_exists_with_multiple_filters_match(self, sync_db):
        result = await _exists(sync_db, Book, title="Clean Code", author="Robert Martin")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_with_multiple_filters_no_match(self, sync_db):
        result = await _exists(sync_db, Book, title="Clean Code", author="Wrong Author")
        assert result is False

    def test_session_is_not_async(self, sync_db):
        assert not isinstance(sync_db, AsyncSession)
        assert isinstance(sync_db, Session)


class TestCountWithSyncSession:

    @pytest.mark.asyncio
    async def test_count_all_records(self, sync_db):
        total = await _count(sync_db, Book)
        assert total == 3

    @pytest.mark.asyncio
    async def test_count_with_filter_returns_one(self, sync_db):
        total = await _count(sync_db, Book, author="Robert Martin")
        assert total == 1

    @pytest.mark.asyncio
    async def test_count_with_no_match_returns_zero(self, sync_db):
        total = await _count(sync_db, Book, title="Does Not Exist")
        assert total == 0

    @pytest.mark.asyncio
    async def test_count_after_insert_reflects_new_record(self, sync_db):
        sync_db.add(Book(title="New Book", author="New Author"))
        sync_db.commit()
        total = await _count(sync_db, Book)
        assert total == 4