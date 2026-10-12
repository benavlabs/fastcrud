"""
The exception a policy raises when it refuses a write or a read outright.

Rows outside a policy's reach behave as if they didn't exist: reads leave them
out, and writes raise ``NoResultFound`` as for a row that isn't there.
``AccessDenied`` is for a denied operation, a written row that fails its check,
data the policy doesn't let the caller write, and an upsert that meets an existing
row outside its reach, which it can't treat as absent: inserting it would collide.
"""

from collections.abc import Sequence
from typing import Any, Literal

AccessDeniedReason = Literal["operation", "check", "columns", "row"]
ColumnAction = Literal["write", "filter", "return"]

_PAST_PARTICIPLES = {"create": "created", "update": "updated"}


class AccessDenied(Exception):
    """
    A policy refused an operation.

    Attributes:
        operation: The operation refused: ``"read"``, ``"create"``, ``"update"``
            or ``"delete"``.
        model: The SQLAlchemy model the operation was on.
        columns: The fields involved when ``reason`` is ``"columns"``.
        reason: Why the policy refused it: ``"operation"`` when the policy
            denies the operation itself, ``"check"`` when a written row fails
            ``create_check`` or ``update_check``, ``"columns"`` when the data
            names fields the caller may not write, filter on, sort by or get back,
            and ``"row"`` when an upsert meets an existing row outside the
            operation's reach, which it can neither update nor insert again.
        action: For ``reason="columns"``, what the caller tried to do with the
            columns: ``"write"`` them, ``"filter"`` or sort by them, or
            ``"return"`` them through a response schema that requires them;
            ``None`` for the other reasons.

    Example:
        >>> try:
        ...     await post_crud.update(db, {"title": "New"}, id=1)
        ... except AccessDenied as denied:
        ...     if denied.reason == "check":
        ...         ...
    """

    def __init__(
        self,
        operation: str,
        model: Any,
        columns: Sequence[str] = (),
        reason: AccessDeniedReason = "operation",
        action: ColumnAction | None = None,
    ) -> None:
        self.operation = operation
        self.model = model
        self.columns = tuple(columns)
        self.reason = reason
        self.action: ColumnAction | None = (
            (action or ("filter" if operation == "read" else "write"))
            if reason == "columns"
            else None
        )
        super().__init__(self._message())

    def _message(self) -> str:
        model_name = getattr(self.model, "__name__", str(self.model))
        if self.reason == "check":
            written = _PAST_PARTICIPLES.get(self.operation, self.operation)
            return f"the {written} {model_name} doesn't pass {self.operation}_check"
        if self.reason == "columns":
            return self._columns_message(model_name)
        if self.reason == "row":
            return f"the existing {model_name} is outside {self.operation}'s reach"
        return f"{self.operation} on {model_name} is not allowed by the policy"

    def _columns_message(self, model_name: str) -> str:
        names = ", ".join(self.columns)
        if self.action == "return":
            return f"{self.operation} on {model_name} may not return: {names}"
        if self.action == "filter" and not self.columns:
            return (
                f"{self.operation} on {model_name} may not filter by raw SQL such as "
                "text(), column() or literal_column(), which can't be checked for "
                "hidden columns"
            )
        if self.action == "filter":
            return (
                f"{self.operation} on {model_name} may not filter or sort by: {names}"
            )
        return f"{self.operation} on {model_name} may not write: {names}"
