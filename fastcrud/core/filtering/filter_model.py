"""
Internal representation of a single filter definition.

This module provides the :class:`Filter` Pydantic model, which captures all
metadata needed to (a) generate a correct FastAPI / OpenAPI query parameter
and (b) build a SQLAlchemy WHERE clause from a runtime value.

It sits between :class:`~fastcrud.core.config.crud_configs.FilterConfig`
(surface-level dict of ``"key__op": default`` entries) and the SQLAlchemy
clauses produced by :class:`~fastcrud.core.filtering.processor.FilterProcessor`.
"""

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, SkipValidation, computed_field
from sqlalchemy import Column


class Filter(BaseModel):
    """
    Resolved metadata for a single filter entry in a ``FilterConfig``.

    Attributes:
        definition: Original key from ``FilterConfig`` (e.g. ``"name"``,
            ``"age__gte"``, ``"tier.name__eq"``).
        param_name: FastAPI-safe parameter name (dots replaced with
            underscores), used as the actual function parameter name in the
            generated dependency.
        default_value: Default value or ``Depends(...)`` callable from
            ``FilterConfig``.
        operator: The operator suffix after ``__``, e.g. ``"gte"``,
            ``"ilike"``, ``"in"``; ``None`` for bare equality filters.
        wrap_type: Wrapper type for collection operators (``list`` for
            ``in``/``not_in``/``between``); ``None`` otherwise.
        joined_model: The related model class when the filter targets a
            joined relationship (e.g. ``"tier.name"``); ``None`` for filters
            on the base model.
        column: The SQLAlchemy ``Column`` the filter ultimately applies to.
        value_type: The Python type of ``column``, derived via
            ``column.type.python_type``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    definition: str
    param_name: str
    default_value: Any
    operator: str | None
    wrap_type: type | None
    joined_model: type | None
    column: SkipValidation[Column[Any]]
    value_type: type

    @computed_field
    def type(self) -> type:
        """
        Final Python type for OpenAPI / FastAPI signature generation.

        Returns ``value_type`` for scalar filters, or ``wrap_type[value_type]``
        (e.g. ``list[int]``) for collection operators.
        """
        if self.wrap_type:
            return cast(type, self.wrap_type[self.value_type])  # type: ignore[index]
        return self.value_type
