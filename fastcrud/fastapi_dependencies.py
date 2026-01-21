"""
FastAPI-specific dependencies and utilities for FastCRUD.

This module contains FastAPI-specific functionality that was extracted from
fastcrud.core.field_management to maintain separation of concerns. These
utilities are tightly coupled to FastAPI and require FastAPI to be installed.

Functions included:
- create_auto_field_injector: Creates dynamic dependency functions for auto field resolution
- create_dynamic_filters: Creates dynamic filter functions for query parameter handling
- inject_dependencies: Wraps functions in FastAPI's Depends
- apply_model_pk: Decorator for injecting primary keys as path parameters

All functions preserve their original signatures and behavior from field_management.py.
"""

import inspect
from typing import Annotated, Callable, Any, Optional, Union, Sequence, TYPE_CHECKING
from uuid import UUID

from fastapi import Depends, Query, Path, params

if TYPE_CHECKING:
    from .core.config import CreateConfig, UpdateConfig, DeleteConfig, FilterConfig


def create_auto_field_injector(
    config: Optional[Union["CreateConfig", "UpdateConfig", "DeleteConfig"]],
) -> Callable[..., dict[str, Any]]:
    """
    Creates a dynamic dependency function that resolves auto_fields.

    Similar to create_dynamic_filters but for CreateConfig/UpdateConfig/DeleteConfig.
    Returns a function that can be used with Depends() to inject auto field values.

    Args:
        config: Configuration object containing auto_fields mapping.

    Returns:
        A dependency function that resolves auto field values.

    Example:
        >>> from datetime import datetime
        >>> from fastapi import Depends
        >>>
        >>> def get_current_user_id():
        ...     return 123
        >>>
        >>> def get_timestamp():
        ...     return datetime.now()
        >>>
        >>> config = CreateConfig(auto_fields={
        ...     "user_id": get_current_user_id,
        ...     "created_at": get_timestamp
        ... })
        >>>
        >>> injector = create_auto_field_injector(config)
        >>> # injector can now be used with FastAPI's Depends()
    """
    if config is None or not config.auto_fields:
        return lambda: {}

    def auto_fields_resolver(**kwargs: Any) -> dict[str, Any]:
        """Receives resolved dependency values and returns dict of field:value."""
        return kwargs

    params = []
    for field_name, func in config.auto_fields.items():
        params.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Depends(func),
            )
        )

    sig = inspect.Signature(params)
    setattr(auto_fields_resolver, "__signature__", sig)

    return auto_fields_resolver

def _str_to_bool(value: Union[bool, str]) -> bool:
    """
    Helper function to properly convert string to bool
    """

    if isinstance(value, bool):
        return value

    str_value = str(value).lower()
    if str_value in ("true", "1"):
        return True
    elif str_value in ("false", "0"):
        return False

    raise ValueError(f"Cannot convert '{value}' / (type {type(value)}) to bool")

def create_dynamic_filters(
    filter_config: Optional["FilterConfig"], column_types: dict[str, type]
) -> Callable[..., dict[str, Any]]:
    """
    Create dynamic filter function for handling query parameters.

    This function creates a dependency function that can parse and validate
    query parameters based on the filter configuration and column types.

    Args:
        filter_config: Configuration object defining available filters.
        column_types: Dictionary mapping column names to their Python types.

    Returns:
        A dependency function that processes filter parameters.

    Example:
        >>> filter_config = FilterConfig(filters={"name": None, "age__gte": 18})
        >>> column_types = {"name": str, "age": int}
        >>> filter_func = create_dynamic_filters(filter_config, column_types)
        >>> # filter_func can be used with FastAPI's Depends()
    """
    if filter_config is None:
        return lambda: {}

    param_to_filter_key = {}
    for original_key in filter_config.filters.keys():
        param_name = original_key.replace(".", "_")
        param_to_filter_key[param_name] = original_key

    def filters(
        **kwargs: Any,
    ) -> dict[str, Any]:
        filtered_params = {}
        for param_name, value in kwargs.items():
            if value is not None:
                original_key = param_to_filter_key.get(param_name, param_name)
                key_without_op = original_key.rsplit("__", 1)[0]
                parse_func = column_types.get(key_without_op)
                if parse_func:
                    try:
                        # Special handling for bool bool('false') = True
                        filtered_params[original_key] = _str_to_bool(value) if (parse_func is bool) else parse_func(value)
                    except (ValueError, TypeError):
                        filtered_params[original_key] = value
                else:
                    filtered_params[original_key] = value
        return filtered_params

    params = []
    for key, value in filter_config.filters.items():
        param_name = key.replace(".", "_")

        if callable(value):
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=Depends(value),
                )
            )
        else:
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=Query(value, alias=key),
                )
            )

    sig = inspect.Signature(params)
    setattr(filters, "__signature__", sig)

    return filters


def inject_dependencies(
    funcs: Optional[Sequence[Callable]] = None,
) -> Optional[Sequence[params.Depends]]:
    """
    Wraps a list of functions in FastAPI's Depends.

    Args:
        funcs: Optional sequence of callable functions to wrap.

    Returns:
        Sequence of FastAPI Depends objects, or None if no functions provided.

    Raises:
        TypeError: If any function in the sequence is not callable.

    Example:
        >>> def get_current_user():
        ...     return {"id": 1}
        >>>
        >>> def get_permissions():
        ...     return ["read", "write"]
        >>>
        >>> deps = inject_dependencies([get_current_user, get_permissions])
        >>> # Returns [Depends(get_current_user), Depends(get_permissions)]
    """
    if funcs is None:
        return None

    for func in funcs:
        if not callable(func):
            raise TypeError(
                f"All dependencies must be callable. Got {type(func)} instead."
            )

    return [Depends(func) for func in funcs]


def apply_model_pk(**pkeys: type):
    """
    This decorator injects positional arguments into a fastCRUD endpoint.
    It dynamically changes the endpoint signature and allows to use
    multiple primary keys without defining them explicitly.

    Args:
        **pkeys: Dictionary mapping primary key names to their types.

    Returns:
        Decorator function that modifies endpoint signatures.

    Example:
        >>> from uuid import UUID
        >>>
        >>> @apply_model_pk(user_id=int, session_id=UUID)
        >>> def get_user_session(user_id: int, session_id: UUID, db: Session):
        ...     # Primary keys are automatically injected as path parameters
        ...     pass
    """

    def wrapper(endpoint):
        signature = inspect.signature(endpoint)
        parameters = [
            p
            for p in signature.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        extra_positional_params = []
        for k, v in pkeys.items():
            if v == UUID:
                extra_positional_params.append(
                    inspect.Parameter(
                        name=k,
                        annotation=Annotated[UUID, Path(...)],
                        kind=inspect.Parameter.POSITIONAL_ONLY,
                    )
                )
            else:
                extra_positional_params.append(
                    inspect.Parameter(
                        name=k, annotation=v, kind=inspect.Parameter.POSITIONAL_ONLY
                    )
                )

        endpoint.__signature__ = signature.replace(
            parameters=extra_positional_params + parameters
        )
        return endpoint

    return wrapper
