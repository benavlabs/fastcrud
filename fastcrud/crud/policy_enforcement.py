"""
Applying a ``PolicyConfig`` to FastCRUD's reads and writes.

These helpers turn a policy into the conditions an operation's statements carry,
refuse the data a policy doesn't let a caller write, and check written rows against
``create_check`` and ``update_check``.
"""

from collections.abc import Sequence
from typing import Any, Literal

from sqlalchemy import false, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect as sa_inspect
from sqlalchemy.sql.elements import ColumnElement, TextClause

from ..core.config.policy_configs import PolicyConfig, RowOperation
from ..core.query.cartesian import refuse_a_cartesian_product
from ..exceptions.access import AccessDenied

Condition = ColumnElement | TextClause


def conditions_in_reach(
    policy: PolicyConfig | None, model: Any, operation: RowOperation
) -> list[Condition]:
    """
    The conditions on the rows an operation reaches under a policy.

    Reads reach the rows matching ``read``. Updates and deletes reach the rows
    matching ``read`` and their own rule, as Postgres does once a statement has a
    WHERE, so they never reach a row the caller couldn't read.

    Args:
        policy: The policy, or ``None`` for no restriction.
        model: The SQLAlchemy model the operation is on.
        operation: ``"read"``, ``"update"`` or ``"delete"``.

    Returns:
        The conditions to AND with the caller's filters; ``[]`` without a policy.

    Raises:
        AccessDenied: If the policy denies the operation.
    """
    if policy is None:
        return []
    if not policy.grants(operation):
        raise AccessDenied(operation, model)
    own_conditions = policy.row_conditions(operation)
    if operation == "read":
        return own_conditions
    readable: list[Condition] = (
        policy.row_conditions("read") if policy.grants("read") else [false()]
    )
    return [*readable, *own_conditions]


def refuse_relationship_data(
    policy: PolicyConfig | None,
    model: Any,
    operation: Literal["create", "update"],
    data: dict[str, Any],
) -> None:
    """
    Refuse related records in a create or update under a policy.

    A policy covers its own model only, so it can't vouch for related records
    written through a relationship; those go through the related model's own
    ``FastCRUD`` and policy. A relationship set to ``None`` or ``[]`` carries no
    records and is allowed.

    Raises:
        AccessDenied: With ``reason="columns"``, naming the relationships.
    """
    if policy is None:
        return
    relationship_names = sa_inspect(model).relationships.keys()
    carrying_records = [
        name
        for name in relationship_names
        if name in data and data[name] is not None and data[name] != []
    ]
    if carrying_records:
        raise AccessDenied(operation, model, columns=carrying_records, reason="columns")


def primary_key_of(model: Any, instance: Any) -> tuple[Any, ...]:
    """The primary key values of a model instance, in the mapper's column order."""
    return tuple(sa_inspect(model).primary_key_from_instance(instance))


def primary_key_in(model: Any, keys: Sequence[tuple[Any, ...]]) -> ColumnElement:
    """A condition matching the rows whose primary key is one of ``keys``."""
    columns = sa_inspect(model).primary_key
    if len(columns) == 1:
        single_column_match: ColumnElement = columns[0].in_([key[0] for key in keys])
        return single_column_match
    return tuple_(*columns).in_(keys)


async def primary_keys_matching(
    db: AsyncSession, model: Any, conditions: Sequence[Condition]
) -> list[tuple[Any, ...]]:
    """The primary keys of the rows matching ``conditions``."""
    columns = sa_inspect(model).primary_key
    result = await db.execute(select(*columns).where(*conditions))
    return [tuple(row) for row in result]


async def row_in_reach(
    db: AsyncSession, model: Any, instance: Any, conditions: Sequence[Condition]
) -> bool:
    """Whether a loaded instance's row matches ``conditions``."""
    if not conditions:
        return True
    keys = [primary_key_of(model, instance)]
    statement = (
        select(func.count())
        .select_from(model)
        .where(primary_key_in(model, keys), *conditions)
    )
    refuse_a_cartesian_product(statement)
    return bool(await db.scalar(statement))


async def refuse_rows_failing_check(
    db: AsyncSession,
    model: Any,
    keys: Sequence[tuple[Any, ...]],
    conditions: Sequence[Condition],
    operation: Literal["create", "update"],
) -> None:
    """
    Refuse written rows that don't meet a policy's check.

    Args:
        db: The session holding the written rows, flushed.
        model: The SQLAlchemy model written.
        keys: The primary keys of the rows written.
        conditions: The check's conditions.
        operation: ``"create"`` or ``"update"``.

    Raises:
        AccessDenied: With ``reason="check"``, if any row fails the check.
    """
    if not conditions or not keys:
        return
    statement = (
        select(func.count())
        .select_from(model)
        .where(primary_key_in(model, keys), *conditions)
    )
    refuse_a_cartesian_product(statement)
    passing = await db.scalar(statement)
    if passing != len(set(keys)):
        raise AccessDenied(operation, model, reason="check")
