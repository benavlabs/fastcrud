"""
The exception a policy raises when it refuses a write or a read outright.

Rows outside a policy's reach behave as if they didn't exist, so they never raise
this: reads leave them out, and writes raise ``NoResultFound`` as for a row that
isn't there. ``AccessDenied`` is for a denied operation, a written row that fails
its check, and data the policy doesn't let the caller write.
"""

from collections.abc import Sequence
from typing import Any, Literal

AccessDeniedReason = Literal["operation", "check", "columns"]

_PAST_PARTICIPLES = {"create": "created", "update": "updated"}


class AccessDenied(Exception):
    """
    A policy refused an operation.

    Attributes:
        operation: The operation refused: ``"read"``, ``"create"``, ``"update"``
            or ``"delete"``.
        model: The SQLAlchemy model the operation was on.
        columns: The fields the caller may not write, when ``reason`` is
            ``"columns"``.
        reason: Why the policy refused it: ``"operation"`` when the policy
            denies the operation itself, ``"check"`` when a written row fails
            ``create_check`` or ``update_check``, and ``"columns"`` when the data
            names fields the caller may not write.

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
    ) -> None:
        self.operation = operation
        self.model = model
        self.columns = tuple(columns)
        self.reason = reason
        super().__init__(self._message())

    def _message(self) -> str:
        model_name = getattr(self.model, "__name__", str(self.model))
        if self.reason == "check":
            written = _PAST_PARTICIPLES.get(self.operation, self.operation)
            return f"the {written} {model_name} doesn't pass {self.operation}_check"
        if self.reason == "columns":
            return (
                f"{self.operation} on {model_name} may not write: "
                f"{', '.join(self.columns)}"
            )
        return f"{self.operation} on {model_name} is not allowed by the policy"
