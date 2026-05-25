"""Type checking tests for schema_to_select override return types.

Run with: uv run mypy tests/type_checking/test_schema_override_typing.py --ignore-missing-imports

Regression test for issue #320: passing schema_to_select=Subclass should yield
a return type of Subclass, not the class-level SelectSchemaType.

This file is for static type checking only and is not meant to be executed.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fastcrud import FastCRUD
from fastcrud.types import GetMultiResponseModel


class Base(DeclarativeBase):
    pass


class Location(Base):
    __tablename__ = "locations"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]


class LocationRead(BaseModel):
    id: int
    name: str


class LocationReadDetailed(LocationRead):
    extended: str = "details"


# Class-level select schema is LocationRead (the "base" schema).
crud: FastCRUD[
    Location, LocationRead, LocationRead, LocationRead, LocationRead, LocationRead
] = FastCRUD(Location)


if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession

    async def test_get_override_propagates(db: AsyncSession) -> None:
        """get() return type should be the schema passed to schema_to_select, not the class-level one."""
        result = await crud.get(
            db=db,
            schema_to_select=LocationReadDetailed,
            return_as_model=True,
            id=1,
        )
        # result should be LocationReadDetailed | None — accessing `.extended`
        # must type-check without a cast.
        if result is not None:
            reveal: str = result.extended  # noqa: F841

    async def test_get_multi_override_propagates(db: AsyncSession) -> None:
        """get_multi() return type should carry the overriding schema."""
        result: GetMultiResponseModel[LocationReadDetailed] = await crud.get_multi(
            db=db,
            schema_to_select=LocationReadDetailed,
            return_as_model=True,
        )
        for item in result["data"]:
            reveal: str = item.extended  # noqa: F841
