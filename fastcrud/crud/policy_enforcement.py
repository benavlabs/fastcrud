"""
Applying a ``PolicyConfig`` to FastCRUD's reads and writes.

These helpers turn a policy into the conditions an operation's statements carry,
refuse the data a policy doesn't let a caller write, and check written rows against
``create_check`` and ``update_check``.
"""

from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import false, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect as sa_inspect
from sqlalchemy.sql import visitors
from sqlalchemy.sql.elements import ColumnClause, ColumnElement, TextClause

from ..core.filtering.processor import where_conditions
from ..types import WhereClause

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


def refuse_columns_that_are_not_columns(
    policy: PolicyConfig | None, model: Any
) -> None:
    """
    Refuse column rules naming anything but the model's columns.

    A misspelled hidden column would leave the real one visible, so every name in
    ``hidden_columns`` and ``read_only_columns`` must be a column attribute of the
    model; relationships and hybrid properties aren't columns.

    Raises:
        ValueError: Naming the rule and the names that aren't columns.
    """
    if policy is None:
        return
    columns = set(sa_inspect(model).mapper.column_attrs.keys())
    for rule in ("hidden_columns", "read_only_columns"):
        not_columns = sorted(getattr(policy, rule) - columns)
        if not_columns:
            raise ValueError(
                f"{rule} names {', '.join(not_columns)}, which "
                f"{'isn' if len(not_columns) == 1 else 'aren'}'t "
                f"{'a column' if len(not_columns) == 1 else 'columns'} of {model.__name__}"
            )


def hidden_columns_of(policy: PolicyConfig | None) -> frozenset[str]:
    """The attribute names of the columns a policy hides; none without a policy."""
    return policy.hidden_columns if policy is not None else frozenset()


def refuse_hidden_input(
    policy: PolicyConfig | None,
    model: Any,
    filters: dict[str, Any] | None = None,
    where: WhereClause = None,
    sort_columns: str | Sequence[str] | None = None,
    joined_where: Iterable[WhereClause] = (),
) -> None:
    """
    Refuse a caller's filters, sorts and conditions that use hidden columns.

    Filtering or sorting by a hidden column would reveal its values, so it's a
    read refusal even inside an update or delete. Conditions are walked into
    subqueries, and columns of the model's aliases count as the model's. Raw SQL
    (``text()``, and ``column()`` or ``literal_column()``, which aren't tied to a
    table) can't be inspected, so it's refused whenever the policy hides
    columns. Joined models' filters, named with a dot, aren't checked: the
    policy covers this model only.

    Args:
        policy: The policy, or ``None`` for no restriction.
        model: The SQLAlchemy model the policy is on.
        filters: The caller's keyword filters.
        where: The caller's ``where=`` conditions.
        sort_columns: The column or columns the caller sorts by.
        joined_where: The ``where`` of the caller's ``JoinConfig`` and ``CountConfig``.

    Raises:
        AccessDenied: With ``operation="read"`` and ``reason="columns"``.
    """
    hidden = hidden_columns_of(policy)
    if not hidden:
        return
    sorted_by = [sort_columns] if isinstance(sort_columns, str) else sort_columns or []
    named = (_fields_filtered_on(filters or {}) | set(sorted_by)) & hidden
    conditions = [
        condition
        for clause in (where, *joined_where)
        for condition in where_conditions(clause)
    ]
    named_in_conditions, has_raw_sql = _hidden_columns_in(conditions, model, hidden)
    named |= named_in_conditions
    if named:
        raise AccessDenied(
            "read",
            model,
            columns=sorted(named),
            reason="columns",
            action="filter",
        )
    if has_raw_sql:
        raise AccessDenied("read", model, reason="columns", action="filter")


def _fields_filtered_on(filters: dict[str, Any]) -> set[str]:
    """The base model's fields named by keyword filters such as ``age__gt`` or ``_or``."""
    fields: set[str] = set()
    for key, value in filters.items():
        if key == "_or" and isinstance(value, dict):
            fields |= _fields_filtered_on(value)
            continue
        field = key.rsplit("__", 1)[0] if "__" in key else key
        if "." not in field:
            fields.add(field)
    return fields


def _hidden_columns_in(
    conditions: Sequence[Condition], model: Any, hidden: frozenset[str]
) -> tuple[set[str], bool]:
    """The hidden attributes the conditions name, and whether any is raw SQL."""
    mapper = sa_inspect(model).mapper
    attribute_by_column = {
        column: name for name in hidden for column in mapper.get_property(name).columns
    }
    named: set[str] = set()
    has_raw_sql = False
    for condition in conditions:
        for element in visitors.iterate(condition):
            if _is_raw_sql(element):
                has_raw_sql = True
            elif isinstance(element, ColumnClause):
                named |= {
                    attribute_by_column[proxied]
                    for proxied in element.proxy_set
                    if proxied in attribute_by_column
                }
    return named, has_raw_sql


def _is_raw_sql(element: Any) -> bool:
    """
    Whether an element is SQL text, or a column tied to no table or written as text.

    SQLAlchemy writes ``exists()``, ``.any()``, ``.has()`` and ``count(*)`` with
    literal ``*`` and ``1`` columns of its own, which name no column, so those
    aren't raw SQL.
    """
    if isinstance(element, TextClause):
        return True
    if not isinstance(element, ColumnClause):
        return False
    if element.is_literal:
        return not _names_no_column(element.name)
    return element.table is None


def _names_no_column(literal: str) -> bool:
    """Whether a literal column's text is a ``*`` or a number, which name no column."""
    return literal == "*" or literal.isdigit()


def refuse_schema_requiring_hidden(
    policy: PolicyConfig | None, model: Any, schema: type[BaseModel] | None
) -> None:
    """
    Refuse a response schema that can't be built without a hidden column.

    Raises:
        AccessDenied: With ``operation="read"``, ``reason="columns"`` and
            ``action="return"``, naming the required hidden fields.
    """
    hidden = hidden_columns_of(policy)
    if not hidden or schema is None:
        return
    required_hidden = sorted(
        name
        for name, field in schema.model_fields.items()
        if field.is_required() and name in hidden
    )
    if required_hidden:
        raise AccessDenied(
            "read",
            model,
            columns=required_hidden,
            reason="columns",
            action="return",
        )


def refuse_unwritable_data(
    policy: PolicyConfig | None,
    model: Any,
    operation: Literal["create", "update"],
    fields_set: Iterable[str],
) -> None:
    """
    Refuse data that sets a hidden or read-only column.

    Only the fields the caller set count: a field set to ``None`` is set, and one
    left to its schema default isn't.

    Raises:
        AccessDenied: With ``reason="columns"``, naming the columns.
    """
    if policy is None:
        return
    unwritable = sorted(set(fields_set) & policy.unwritable_columns)
    if unwritable:
        raise AccessDenied(operation, model, columns=unwritable, reason="columns")


def without_hidden_columns(
    policy: PolicyConfig | None, columns: list[Any], kept: Iterable[str] = ()
) -> list[Any]:
    """
    The selected columns, without the ones the policy hides.

    Args:
        policy: The policy, or ``None`` for no restriction.
        columns: The selected columns.
        kept: Hidden columns to select anyway because FastCRUD needs them, such
            as the primary key for nesting; they're dropped from the results.
    """
    left_out = hidden_columns_of(policy) - set(kept)
    return [
        column for column in columns if getattr(column, "key", None) not in left_out
    ]


def without_hidden_fields(
    policy: PolicyConfig | None, row: dict[str, Any]
) -> dict[str, Any]:
    """A result row without the policy's hidden columns."""
    hidden = hidden_columns_of(policy)
    return {field: value for field, value in row.items() if field not in hidden}


def hidden_column_keys(policy: PolicyConfig | None, model: Any) -> frozenset[str]:
    """The table column keys of the hidden columns, as ``return_columns`` names them."""
    mapper = sa_inspect(model).mapper
    return frozenset(
        column.key
        for name in hidden_columns_of(policy)
        for column in mapper.get_property(name).columns
    )
