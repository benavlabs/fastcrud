"""
Test for the cartesian product bug in get_joined when using multiple one-to-many relationships.

This test reproduces the issue where multiple one-to-many joins create a cartesian product
and FastCRUD doesn't properly deduplicate the nested lists, resulting in duplicate entries.

Bug report: When using get_joined() with multiple one-to-many relationships and nest_joins=True,
FastCRUD returns duplicate entries in nested lists due to SQL cartesian product not being deduplicated.
"""

import pytest
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from fastcrud import FastCRUD, JoinConfig
from pydantic import BaseModel, ConfigDict
from typing import List, Optional

from ...sqlalchemy.conftest import Base


# Test Models for reproducing the cartesian product bug
class BookCartesianTest(Base):
    __tablename__ = "books_cartesian_test"

    id = Column(Integer, primary_key=True)
    title = Column(String(100))

    # One-to-many relationships
    authors = relationship("AuthorCartesianTest", back_populates="book")
    genres = relationship("GenreCartesianTest", back_populates="book")


class AuthorCartesianTest(Base):
    __tablename__ = "authors_cartesian_test"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books_cartesian_test.id"))
    name = Column(String(100))

    book = relationship("BookCartesianTest", back_populates="authors")


class GenreCartesianTest(Base):
    __tablename__ = "genres_cartesian_test"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books_cartesian_test.id"))
    name = Column(String(100))

    book = relationship("BookCartesianTest", back_populates="genres")


# Pydantic Schemas
class AuthorCartesianTestSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    book_id: int


class GenreCartesianTestSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    book_id: int


class BookCartesianTestSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    authors: list[AuthorCartesianTestSchema] | None = []
    genres: list[GenreCartesianTestSchema] | None = []


@pytest.mark.asyncio
async def test_get_joined_multiple_one_to_many_cartesian_product_bug(async_session):
    """
    Test that reproduces the cartesian product bug in get_joined.

    When joining multiple one-to-many relationships, the SQL query creates a cartesian product.
    For example: 1 book with 2 authors and 2 genres results in 4 SQL rows (2×2).

    The current get_joined implementation processes each row individually and doesn't
    deduplicate properly, resulting in duplicated entries in the nested lists.

    Expected: 2 authors + 2 genres
    Current bug: 4 authors + 4 genres (each author appears twice, each genre appears twice)
    """
    # Create test data: 1 book with 2 authors and 2 genres
    book = BookCartesianTest(title="Mystery Novel")
    async_session.add(book)
    await async_session.flush()  # Get the book ID

    author1 = AuthorCartesianTest(book_id=book.id, name="Author One")
    author2 = AuthorCartesianTest(book_id=book.id, name="Author Two")
    async_session.add_all([author1, author2])

    genre1 = GenreCartesianTest(book_id=book.id, name="Fiction")
    genre2 = GenreCartesianTest(book_id=book.id, name="Mystery")
    async_session.add_all([genre1, genre2])

    await async_session.commit()

    # Query with FastCRUD using multiple one-to-many joins
    book_crud = FastCRUD(BookCartesianTest)

    result = await book_crud.get_joined(
        db=async_session,
        schema_to_select=BookCartesianTestSchema,
        joins_config=[
            JoinConfig(
                model=AuthorCartesianTest,
                join_on=BookCartesianTest.id == AuthorCartesianTest.book_id,
                join_prefix="authors_",
                join_type="left",
                relationship_type="one-to-many",
            ),
            JoinConfig(
                model=GenreCartesianTest,
                join_on=BookCartesianTest.id == GenreCartesianTest.book_id,
                join_prefix="genres_",
                join_type="left",
                relationship_type="one-to-many",
            ),
        ],
        nest_joins=True,
        id=book.id,
    )

    # Verify the result exists
    assert result is not None, "Expected a result from the query"
    assert result["title"] == "Mystery Novel", "Book title should match"

    # Check the cartesian product bug
    authors = result.get("authors", [])
    genres = result.get("genres", [])

    print("Expected: 2 authors + 2 genres")
    print(f"Got:      {len(authors)} authors + {len(genres)} genres")
    print(f"Authors: {[a['name'] for a in authors]}")
    print(f"Genres:  {[g['name'] for g in genres]}")

    # This test will currently FAIL due to the cartesian product bug
    # After the fix, these assertions should pass:

    # Expected behavior: exactly 2 unique authors and 2 unique genres
    assert (
        len(authors) == 2
    ), f"Expected 2 authors, got {len(authors)}. Bug: cartesian product creates duplicates"
    assert (
        len(genres) == 2
    ), f"Expected 2 genres, got {len(genres)}. Bug: cartesian product creates duplicates"

    # Check that we have the correct unique authors
    author_names = [a["name"] for a in authors]
    expected_authors = ["Author One", "Author Two"]
    assert sorted(author_names) == sorted(
        expected_authors
    ), f"Expected {expected_authors}, got {author_names}"

    # Check that we have the correct unique genres
    genre_names = [g["name"] for g in genres]
    expected_genres = ["Fiction", "Mystery"]
    assert sorted(genre_names) == sorted(
        expected_genres
    ), f"Expected {expected_genres}, got {genre_names}"

    # Verify no duplicates (this is what currently fails)
    assert len(set(author_names)) == len(
        author_names
    ), "Authors list should not contain duplicates"
    assert len(set(genre_names)) == len(
        genre_names
    ), "Genres list should not contain duplicates"


@pytest.mark.asyncio
async def test_get_joined_single_one_to_many_works_correctly(async_session):
    """
    Test that single one-to-many relationships work correctly (no cartesian product).
    This should pass even before the fix.
    """
    # Create test data: 1 book with 2 authors only
    book = BookCartesianTest(title="Adventure Novel")
    async_session.add(book)
    await async_session.flush()

    author1 = AuthorCartesianTest(book_id=book.id, name="Adventure Author One")
    author2 = AuthorCartesianTest(book_id=book.id, name="Adventure Author Two")
    async_session.add_all([author1, author2])

    await async_session.commit()

    # Query with only one one-to-many join (should work correctly)
    book_crud = FastCRUD(BookCartesianTest)

    result = await book_crud.get_joined(
        db=async_session,
        schema_to_select=BookCartesianTestSchema,
        joins_config=[
            JoinConfig(
                model=AuthorCartesianTest,
                join_on=BookCartesianTest.id == AuthorCartesianTest.book_id,
                join_prefix="authors_",
                join_type="left",
                relationship_type="one-to-many",
            ),
        ],
        nest_joins=True,
        id=book.id,
    )

    # This should work correctly (no cartesian product with single one-to-many)
    assert result is not None
    assert result["title"] == "Adventure Novel"

    authors = result.get("authors", [])
    assert len(authors) == 2, f"Expected 2 authors, got {len(authors)}"

    author_names = [a["name"] for a in authors]
    expected_authors = ["Adventure Author One", "Adventure Author Two"]
    assert sorted(author_names) == sorted(expected_authors)


@pytest.mark.asyncio
async def test_get_multi_joined_works_correctly_with_multiple_one_to_many(
    async_session,
):
    """
    Test that get_multi_joined handles multiple one-to-many relationships correctly.
    This should work correctly even before the fix because get_multi_joined uses JoinProcessor.
    """
    # Create test data: 1 book with 2 authors and 2 genres
    book = BookCartesianTest(title="Science Fiction Novel")
    async_session.add(book)
    await async_session.flush()

    author1 = AuthorCartesianTest(book_id=book.id, name="Sci-Fi Author One")
    author2 = AuthorCartesianTest(book_id=book.id, name="Sci-Fi Author Two")
    async_session.add_all([author1, author2])

    genre1 = GenreCartesianTest(book_id=book.id, name="Science Fiction")
    genre2 = GenreCartesianTest(book_id=book.id, name="Space Opera")
    async_session.add_all([genre1, genre2])

    await async_session.commit()

    # Use get_multi_joined (should work correctly due to JoinProcessor)
    book_crud = FastCRUD(BookCartesianTest)

    result = await book_crud.get_multi_joined(
        db=async_session,
        schema_to_select=BookCartesianTestSchema,
        joins_config=[
            JoinConfig(
                model=AuthorCartesianTest,
                join_on=BookCartesianTest.id == AuthorCartesianTest.book_id,
                join_prefix="authors_",
                join_type="left",
                relationship_type="one-to-many",
            ),
            JoinConfig(
                model=GenreCartesianTest,
                join_on=BookCartesianTest.id == GenreCartesianTest.book_id,
                join_prefix="genres_",
                join_type="left",
                relationship_type="one-to-many",
            ),
        ],
        nest_joins=True,
        id=book.id,
    )

    # get_multi_joined should handle this correctly
    assert result is not None
    assert "data" in result
    assert len(result["data"]) == 1

    book_data = result["data"][0]
    assert book_data["title"] == "Science Fiction Novel"

    authors = book_data.get("authors", [])
    genres = book_data.get("genres", [])

    # This should work correctly with get_multi_joined
    assert len(authors) == 2, f"Expected 2 authors, got {len(authors)}"
    assert len(genres) == 2, f"Expected 2 genres, got {len(genres)}"

    author_names = [a["name"] for a in authors]
    expected_authors = ["Sci-Fi Author One", "Sci-Fi Author Two"]
    assert sorted(author_names) == sorted(expected_authors)

    genre_names = [g["name"] for g in genres]
    expected_genres = ["Science Fiction", "Space Opera"]
    assert sorted(genre_names) == sorted(expected_genres)
