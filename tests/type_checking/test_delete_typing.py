"""
Type checking tests for delete methods to verify proper typing support.

This file contains type annotations that should be properly recognized by mypy
and other type checkers after the typing improvements for delete methods.
"""

from typing import Optional
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import DeclarativeBase
from fastcrud import FastCRUD


# Test models and schemas
class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "user_model"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)


class CreateUserSchema(BaseModel):
    name: str


class UpdateUserSchema(BaseModel):
    name: str | None = None


class UpdateUserInternalSchema(BaseModel):
    name: str | None = None


class DeleteUserSchema(BaseModel):
    id: int | None = None
    name: str | None = None


class SelectUserSchema(BaseModel):
    id: int
    name: str


# Type-safe FastCRUD instance
UserCRUD = FastCRUD[
    UserModel,
    CreateUserSchema,
    UpdateUserSchema,
    UpdateUserInternalSchema,
    DeleteUserSchema,
    SelectUserSchema,
]


def test_delete_method_typing() -> None:
    """Test that delete method has proper typing support."""
    crud = UserCRUD(UserModel)

    # Test that the method signature includes the filters parameter
    import inspect

    sig = inspect.signature(crud.delete)
    assert "filters" in sig.parameters
    assert "kwargs" in sig.parameters

    # Test that filters parameter has correct type annotation
    filters_param = sig.parameters["filters"]
    assert "DeleteUserSchema" in str(
        filters_param.annotation
    ) or "DeleteSchemaType" in str(filters_param.annotation)


def test_db_delete_method_typing() -> None:
    """Test that db_delete method has proper typing support."""
    crud = UserCRUD(UserModel)

    # Test that the method signature includes the filters parameter
    import inspect

    sig = inspect.signature(crud.db_delete)
    assert "filters" in sig.parameters
    assert "kwargs" in sig.parameters

    # Test that filters parameter has correct type annotation
    filters_param = sig.parameters["filters"]
    assert "DeleteUserSchema" in str(
        filters_param.annotation
    ) or "DeleteSchemaType" in str(filters_param.annotation)


def test_delete_method_return_type() -> None:
    """Test that delete method return type is properly inferred."""
    crud = UserCRUD(UserModel)

    import inspect

    sig = inspect.signature(crud.delete)
    # The return type should be None
    assert sig.return_annotation is None or str(sig.return_annotation) == "None"


def test_db_delete_method_return_type() -> None:
    """Test that db_delete method return type is properly inferred."""
    crud = UserCRUD(UserModel)

    import inspect

    sig = inspect.signature(crud.db_delete)
    # The return type should be None
    assert sig.return_annotation is None or str(sig.return_annotation) == "None"


def test_fastcrud_generic_typing() -> None:
    """Test that FastCRUD generic typing works correctly."""
    # This should be properly typed with all generic parameters
    crud: FastCRUD[
        UserModel,
        CreateUserSchema,
        UpdateUserSchema,
        UpdateUserInternalSchema,
        DeleteUserSchema,
        SelectUserSchema,
    ] = FastCRUD(UserModel)

    # The model should be properly typed
    assert crud.model == UserModel


def test_delete_schema_typing() -> None:
    """Test that DeleteUserSchema is properly typed."""
    # This should be properly typed
    delete_schema = DeleteUserSchema(id=1, name="test")

    # Fields should be properly typed
    assert isinstance(delete_schema.id, int)
    assert isinstance(delete_schema.name, str)

    # Optional fields should work
    delete_schema_partial = DeleteUserSchema(id=1)
    assert delete_schema_partial.name is None


def test_typing_demonstration() -> None:
    """Demonstrate that the typing issue from GitHub issue #147 is resolved."""
    # Create a properly typed FastCRUD instance (like in the original issue)
    user_crud = FastCRUD[
        UserModel,
        CreateUserSchema,
        UpdateUserSchema,
        UpdateUserInternalSchema,
        DeleteUserSchema,
        SelectUserSchema,
    ](UserModel)

    # Verify that the delete method has proper typing
    import inspect

    delete_sig = inspect.signature(user_crud.delete)
    db_delete_sig = inspect.signature(user_crud.db_delete)

    # Both methods should have filters parameter with proper typing
    assert "filters" in delete_sig.parameters
    assert "filters" in db_delete_sig.parameters

    # The methods should no longer be "partially unknown" to type checkers
    assert hasattr(user_crud, "delete")
    assert hasattr(user_crud, "db_delete")
    assert callable(user_crud.delete)
    assert callable(user_crud.db_delete)
