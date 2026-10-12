"""
Row-level policies for FastCRUD.

A ``PolicyConfig`` set on ``FastCRUD(Model, policy=...)`` decides which operations a
caller may perform on the model and which rows each one reaches, in the manner of
Postgres row-level security: rows outside the policy behave as if they weren't
there, and written rows must pass the policy's checks.
"""

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, SkipValidation
from pydantic.functional_validators import field_validator
from sqlalchemy.sql.elements import ColumnElement, TextClause

from ...types import WhereClause
from ..filtering.processor import where_conditions

RowOperation = Literal["read", "update", "delete"]
CheckedOperation = Literal["create", "update"]
PolicyRule = bool | WhereClause


class PolicyConfig(BaseModel):
    """
    What a caller may do with a model, and to which rows.

    Every operation is denied unless the policy grants it, so ``PolicyConfig()``
    denies everything. ``read``, ``update`` and ``delete`` each take ``True`` for
    every row, ``False`` to deny the operation, or SQLAlchemy conditions (as the
    methods' ``where=`` takes) for the rows matching them. ``update`` and
    ``delete`` reach only rows that match ``read`` too.

    The caller's own ``where=`` and keyword filters are AND-ed with the policy and
    can never widen it: ``where=true()`` means every row the policy reaches. Rows
    outside it behave as absent: reads leave them out, and writes raise
    ``NoResultFound`` as for a missing row. ``AccessDenied`` is raised only for a
    denied operation, a written row that fails its check, or relationship data in
    a create or update.

    The policy covers this model only. Joined models' rows aren't filtered unless
    the join's ``JoinConfig.where`` says so.

    Attributes:
        read: The rows reads reach.
        create: Whether rows may be created: ``True`` or ``False`` only, since
            there are no existing rows to reach. Conditions a created row must
            meet go in ``create_check``.
        update: The rows updates reach, among those ``read`` reaches.
        delete: The rows deletes reach, among those ``read`` reaches. A soft
            delete is a delete, governed by this rule and not by ``update``.
        create_check: Conditions a created row must meet, checked after it's
            written in the same transaction; a row that fails is rolled back.
        update_check: Conditions an updated row must meet afterwards, checked the
            same way. When omitted, it's the ``update`` condition, as in Postgres:
            an updated row must still be within ``update``'s reach.
        hidden_columns: Attribute names of columns the caller never sees: reads,
            ``create`` and ``update`` leave them out of what they return, and the
            caller may not filter or sort by them, name them in ``where=`` or a
            join's ``where``, or write them. A response schema that requires one
            raises ``AccessDenied``; an optional field is just left out. The
            policy's own conditions may use them.
        read_only_columns: Attribute names of columns the caller may see but not
            write: a create or update that sets one raises ``AccessDenied``. A
            field the caller leaves unset isn't written, so the database default
            applies. Server code that sets one, such as ``owner_id``, passes it as
            ``server_data`` to ``create`` or ``update``.

    Column rules restrict the caller's input only: the columns FastCRUD writes
    itself, such as ``updated_at`` and the soft delete columns, are written as
    usual. They cover this model's columns only, so a self-referential join shows
    the joined rows' hidden columns, as it shows rows outside ``read``; a join's
    ``JoinConfig.schema_to_select`` limits what it shows.

    Example:
        >>> PolicyConfig(
        ...     read=Post.published | (Post.owner_id == me),
        ...     create=True,
        ...     create_check=Post.owner_id == me,
        ...     update=Post.owner_id == me,
        ...     delete=False,
        ... )
    """

    read: SkipValidation[PolicyRule] = False
    create: SkipValidation[bool] = False
    update: SkipValidation[PolicyRule] = False
    delete: SkipValidation[PolicyRule] = False
    create_check: SkipValidation[WhereClause] = None
    update_check: SkipValidation[WhereClause] = None
    hidden_columns: frozenset[str] = frozenset()
    read_only_columns: frozenset[str] = frozenset()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("read", "update", "delete")
    def check_rule(cls, value, info):
        """Validate that a rule is True, False or SQLAlchemy conditions."""
        if isinstance(value, bool):
            return value
        if value is None or not _holds_conditions(value):
            raise TypeError(
                f"{info.field_name} takes True, False or SQLAlchemy conditions, "
                f"got {value!r}: use True for every row and False to deny it"
            )
        return value

    @field_validator("create")
    def check_create(cls, value):
        """Validate that create is True or False."""
        if not isinstance(value, bool):
            raise TypeError(
                "create takes True or False, since there are no existing rows to "
                "reach; put the conditions a created row must meet in create_check"
            )
        return value

    @field_validator("create_check", "update_check")
    def check_conditions(cls, value):
        """Validate that a check holds SQLAlchemy conditions."""
        where_conditions(value)
        return value

    def grants(self, operation: RowOperation | Literal["create"]) -> bool:
        """Whether the policy allows an operation on at least some rows."""
        return getattr(self, operation) is not False

    def row_conditions(
        self, operation: RowOperation
    ) -> list[ColumnElement | TextClause]:
        """
        The conditions on the rows an operation reaches by its own rule.

        Args:
            operation: ``"read"``, ``"update"`` or ``"delete"``.

        Returns:
            The rule's conditions, or ``[]`` when it reaches every row. A denied
            operation has none; check ``grants`` first.
        """
        rule = getattr(self, operation)
        if isinstance(rule, bool):
            return []
        return where_conditions(cast(WhereClause, rule))

    def check_conditions_for(
        self, operation: CheckedOperation
    ) -> list[ColumnElement | TextClause]:
        """
        The conditions a row written by an operation must meet afterwards.

        Args:
            operation: ``"create"`` or ``"update"``.

        Returns:
            ``create_check``'s conditions for a create. For an update,
            ``update_check``'s, or the ``update`` rule's when it's omitted.
        """
        if operation == "create":
            return where_conditions(cast(WhereClause, self.create_check))
        if self.update_check is None:
            return self.row_conditions("update")
        return where_conditions(cast(WhereClause, self.update_check))

    @property
    def unwritable_columns(self) -> frozenset[str]:
        """The columns the caller may not write: the hidden and read-only ones."""
        return self.hidden_columns | self.read_only_columns


def _holds_conditions(value: object) -> bool:
    """Whether a value is one SQLAlchemy condition or a non-empty sequence of them."""
    try:
        return bool(where_conditions(cast(WhereClause, value)))
    except TypeError:
        return False
