"""
Database-specific CRUD operations.

This module contains database engine specific operations like upserts
that vary between PostgreSQL, SQLite, and MySQL.
"""

from typing import Union, Any, TYPE_CHECKING
from sqlalchemy import Insert, and_
from sqlalchemy.sql.elements import ColumnElement, TextClause
from sqlalchemy.dialects import postgresql, sqlite, mysql

if TYPE_CHECKING:  # pragma: no cover
    from ..types import UpdateSchemaType, CreateSchemaType


async def upsert_multi_postgresql(
    model_class: Any,
    primary_keys: list[str],
    instances: list[Union["UpdateSchemaType", "CreateSchemaType"]],
    filters: list[ColumnElement | TextClause],
    update_set_override: dict[str, Any],
) -> tuple[Insert, list[dict]]:
    """
    Create PostgreSQL-specific upsert statement.

    Args:
        model_class: SQLAlchemy model class
        primary_keys: List of primary key column names
        instances: Data instances to upsert
        filters: Additional filters for the upsert
        update_set_override: Override values for update

    Returns:
        Tuple of (statement, parameters)
    """
    statement = postgresql.insert(model_class)
    statement = statement.on_conflict_do_update(
        index_elements=primary_keys,
        set_={
            column.name: getattr(statement.excluded, column.name)
            for column in model_class.__table__.columns
            if not column.primary_key and not column.unique
        }
        | update_set_override,
        where=and_(*filters) if filters else None,
    )
    params = [model_class(**instance.model_dump()).__dict__ for instance in instances]
    return statement, params


async def upsert_multi_sqlite(
    model_class: Any,
    primary_keys: list[str],
    instances: list[Union["UpdateSchemaType", "CreateSchemaType"]],
    filters: list[ColumnElement | TextClause],
    update_set_override: dict[str, Any],
) -> tuple[Insert, list[dict]]:
    """
    Create SQLite-specific upsert statement.

    Args:
        model_class: SQLAlchemy model class
        primary_keys: List of primary key column names
        instances: Data instances to upsert
        filters: Additional filters for the upsert
        update_set_override: Override values for update

    Returns:
        Tuple of (statement, parameters)
    """
    statement = sqlite.insert(model_class)
    statement = statement.on_conflict_do_update(
        index_elements=primary_keys,
        set_={
            column.name: getattr(statement.excluded, column.name)
            for column in model_class.__table__.columns
            if not column.primary_key and not column.unique
        }
        | update_set_override,
        where=and_(*filters) if filters else None,
    )
    params = [model_class(**instance.model_dump()).__dict__ for instance in instances]
    return statement, params


async def upsert_multi_mysql(
    model_class: Any,
    instances: list[Union["UpdateSchemaType", "CreateSchemaType"]],
    update_set_override: dict[str, Any],
    deleted_at_column: str,
) -> tuple[Insert, list[dict]]:
    """
    Create MySQL-specific upsert statement.

    Args:
        model_class: SQLAlchemy model class
        instances: Data instances to upsert
        update_set_override: Override values for update
        deleted_at_column: Name of the soft delete column

    Returns:
        Tuple of (statement, parameters)
    """
    statement = mysql.insert(model_class)
    statement = statement.on_duplicate_key_update(
        {
            column.name: getattr(statement.inserted, column.name)
            for column in model_class.__table__.columns
            if not column.primary_key
            and not column.unique
            and column.name != deleted_at_column
        }
        | update_set_override,
    )
    params = [model_class(**instance.model_dump()).__dict__ for instance in instances]
    return statement, params
