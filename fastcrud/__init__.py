from importlib.metadata import version

__version__ = version("fastcrud")

from sqlalchemy.orm import aliased
from sqlalchemy.orm.util import AliasedClass

from .crud.fast_crud import FastCRUD
from .exceptions.access import AccessDenied
from .endpoint.endpoint_creator import EndpointCreator
from .endpoint.crud_router import crud_router
from .core import (
    JoinConfig,
    CountConfig,
    PolicyConfig,
    FilterConfig,
    CreateConfig,
    UpdateConfig,
    DeleteConfig,
    PaginatedListResponse,
    ListResponse,
    PaginatedRequestQuery,
    CursorPaginatedRequestQuery,
    paginated_response,
    compute_offset,
    FilterCallable,
)

__all__ = [
    "__version__",
    "FastCRUD",
    "EndpointCreator",
    "crud_router",
    "JoinConfig",
    "CountConfig",
    "PolicyConfig",
    "AccessDenied",
    "aliased",
    "AliasedClass",
    "FilterConfig",
    "FilterCallable",
    "CreateConfig",
    "UpdateConfig",
    "DeleteConfig",
    "PaginatedListResponse",
    "ListResponse",
    "PaginatedRequestQuery",
    "CursorPaginatedRequestQuery",
    "paginated_response",
    "compute_offset",
]
