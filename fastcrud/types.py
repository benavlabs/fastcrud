from typing import TypeVar, Any, Union, Generic
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel
from typing_extensions import TypedDict, NotRequired

FilterValue = Union[str, int, float, bool, datetime, Decimal, None]
FilterValueSequence = Union[
    list[FilterValue], tuple[FilterValue, ...], set[FilterValue]
]
FilterValueType = Union[FilterValue, FilterValueSequence, dict[str, FilterValue]]

ModelType = TypeVar("ModelType", bound=Any)

SelectSchemaType = TypeVar("SelectSchemaType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)
UpdateSchemaInternalType = TypeVar("UpdateSchemaInternalType", bound=BaseModel)
DeleteSchemaType = TypeVar("DeleteSchemaType", bound=BaseModel)


class GetMultiResponseDict(TypedDict):
    """Response type for get_multi when return_as_model=False.

    Note: This assumes the default multi_response_key="data".
    If using a custom multi_response_key, the actual key will differ.
    """

    data: list[dict[str, Any]]
    total_count: NotRequired[int]


class GetMultiResponseModel(TypedDict, Generic[SelectSchemaType]):
    """Response type for get_multi when return_as_model=True with schema_to_select.

    Note: This assumes the default multi_response_key="data".
    If using a custom multi_response_key, the actual key will differ.
    """

    data: list[SelectSchemaType]
    total_count: NotRequired[int]


class UpsertMultiResponseDict(TypedDict):
    """Response type for upsert_multi when return_as_model=False.

    Note: This assumes the default multi_response_key="data".
    If using a custom multi_response_key, the actual key will differ.
    """

    data: list[dict[str, Any]]


class UpsertMultiResponseModel(TypedDict, Generic[SelectSchemaType]):
    """Response type for upsert_multi when return_as_model=True with schema_to_select.

    Note: This assumes the default multi_response_key="data".
    If using a custom multi_response_key, the actual key will differ.
    """

    data: list[SelectSchemaType]
