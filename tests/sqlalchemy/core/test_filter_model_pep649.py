"""Regression test for PEP 649 / Python 3.14 type-annotation collision.

The Filter Pydantic model declares fields annotated `wrap_type: type | None`
etc., and also has a `@computed_field def type(self)` method. Under Python
3.14's deferred annotation evaluation, the string form `"type | None"`
resolves the bare `type` against the class namespace, which now contains
the property -- producing `TypeError: unsupported operand type(s) for |:
'property' and 'NoneType'`.

This test ensures the model can be constructed (which triggers the
schema build that exercises annotation resolution) on every supported
Python version.
"""

import builtins

from sqlalchemy import Column, Integer

from fastcrud.core.filtering.filter_model import Filter


def test_filter_constructs_without_pep649_type_collision():
    """Constructing Filter forces Pydantic to resolve all field annotations,
    which historically failed under Python 3.14 with a TypeError when
    ``type | None`` evaluated against the class namespace and resolved
    ``type`` to the property instead of the builtin."""
    # Re-resolve all annotations explicitly. Under Python 3.14 (PEP 649)
    # this is the path that crashed with::
    #     TypeError: unsupported operand type(s) for |:
    #         'property' and 'NoneType'
    # because the bare ``type`` in ``wrap_type: type | None`` resolved to
    # the ``@computed_field def type`` property instead of the builtin.
    Filter.model_rebuild(force=True)

    col = Column("id", Integer)
    filter_obj = Filter(
        definition="id",
        param_name="id",
        default_value=None,
        operator=None,
        wrap_type=None,
        joined_model=None,
        column=col,
        value_type=int,
    )

    # Sanity: annotation resolution succeeded and the computed field
    # still returns the scalar value_type for non-collection filters.
    assert filter_obj.type is int

    # The model's stored field annotations should reference the builtin
    # ``type``, not the @computed_field property.
    assert Filter.model_fields["wrap_type"].annotation == (builtins.type | None)
    assert Filter.model_fields["value_type"].annotation is builtins.type


def test_filter_with_wrap_type_collection():
    """``type`` computed_field returns ``wrap_type[value_type]`` for collections."""
    col = Column("id", Integer)
    filter_obj = Filter(
        definition="id__in",
        param_name="id__in",
        default_value=None,
        operator="in",
        wrap_type=list,
        joined_model=None,
        column=col,
        value_type=int,
    )
    assert filter_obj.type == builtins.list[int]
