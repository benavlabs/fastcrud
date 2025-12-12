"""Type checking tests for get_multi response types.

Run with: uv run mypy tests/type_checking/test_get_multi_typing.py --ignore-missing-imports

This file is for static type checking only and is not meant to be executed.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fastcrud import FastCRUD
from fastcrud.types import GetMultiResponseModel, GetMultiResponseDict


class Base(DeclarativeBase):
    pass


class MyModel(Base):
    __tablename__ = "my_model"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]


class MySchema(BaseModel):
    id: int
    name: str


# Create a properly typed crud instance for type checking
crud: FastCRUD[MyModel, MySchema, MySchema, MySchema, MySchema, MySchema] = FastCRUD(
    MyModel
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    async def test_get_multi_with_model(db: AsyncSession) -> None:
        """Test that result["data"] is properly typed as list[MySchema]."""
        result: GetMultiResponseModel[MySchema] = await crud.get_multi(
            db=db,
            schema_to_select=MySchema,
            return_as_model=True,
        )

        # This should NOT produce a type error anymore
        # Previously: Object of type `list[MySchema] | int` may not be iterable
        data = result["data"]

        # data should be list[MySchema], not list[MySchema] | int
        for item in data:
            # item should be MySchema, accessing .name should work
            print(item.name)

        # total_count is NotRequired[int], so it might not exist
        # Using .get() is safer
        total = result.get("total_count", 0)
        print(f"Total: {total}")

    async def test_get_multi_dict(db: AsyncSession) -> None:
        """Test that result["data"] is properly typed as list[dict[str, Any]]."""
        result: GetMultiResponseDict = await crud.get_multi(
            db=db,
            return_as_model=False,
        )

        # This should work without type errors
        data = result["data"]

        for item in data:
            # item should be dict[str, Any]
            print(item["name"])

    async def test_direct_iteration(db: AsyncSession) -> None:
        """Test the exact use case from the bug report."""
        result: GetMultiResponseModel[MySchema] = await crud.get_multi(
            db=db,
            schema_to_select=MySchema,
            return_as_model=True,
        )

        # The original bug: this produced a type error because
        # data was typed as list[MySchema] | int
        data = result.get("data", [])

        # This should NOT produce: "Object of type `list[MySchema] | int` may not be iterable"
        for item in data:
            print(item.name)
