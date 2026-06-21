# Endpoints — `crud_router` and `EndpointCreator`

How to customize the auto-generated FastAPI router.

---

## `crud_router` full parameter reference

```python
crud_router(
    session,                          # callable[..., AsyncSession] — FastAPI dependency
    model,                            # SQLAlchemy / SQLModel model class
    create_schema,                    # Pydantic schema for POST body
    update_schema,                    # Pydantic schema for PATCH body
    *,
    crud=None,                        # pre-built FastCRUD instance (else FastCRUD(model))
    delete_schema=None,               # rarely needed — delete is by path param
    select_schema=None,               # response schema for read endpoints
    path="",                          # base path prefix (e.g. "/users")
    tags=None,                        # OpenAPI tags
    include_in_schema=True,

    # Per-method dependencies (auth, rate limit, etc.)
    create_deps=[],
    read_deps=[],
    read_multi_deps=[],
    update_deps=[],
    delete_deps=[],
    db_delete_deps=[],

    # Method selection
    included_methods=None,            # whitelist: ["create", "read", "read_multi", ...]
    deleted_methods=None,             # blacklist (cannot use with included_methods)

    # Endpoint name customization
    endpoint_names=None,              # {"create": "make", "read": "fetch", ...}

    # Filters on the list endpoint
    filter_config=None,               # FilterConfig | dict
    custom_filters=None,              # {"year": year_eq, ...}

    # Soft delete
    is_deleted_column="is_deleted",
    deleted_at_column="deleted_at",
    updated_at_column="updated_at",

    # Joins / relationships
    include_relationships=False,      # True | False | ["rel1", "rel2"]
    joins_config=None,                # explicit JoinConfig list (mutually exclusive with above)
    nest_joins=True,
    default_nested_limit=None,
    include_one_to_many=False,

    # Advanced configs
    create_config=None,               # CreateConfig — auto-inject fields, exclusions
    update_config=None,               # UpdateConfig — same
    delete_config=None,               # DeleteConfig

    # For deep customization
    endpoint_creator=None,            # subclass of EndpointCreator
)
```

---

## Selecting which methods to expose

```python
crud_router(..., included_methods=["read", "read_multi"])   # read-only API
crud_router(..., deleted_methods=["db_delete"])              # all but hard-delete
```

Method names (the strings accepted by `included_methods` / `deleted_methods` / `endpoint_names`):
- `"create"` → `POST {path}`
- `"read"` → `GET {path}/{pk}`
- `"read_multi"` → `GET {path}`
- `"update"` → `PATCH {path}/{pk}`
- `"delete"` → `DELETE {path}/{pk}` (soft if configured, see [Soft delete](#soft-delete))
- `"db_delete"` → `DELETE {path}/db_delete/{pk}` (always hard delete)

`{pk}` is the actual primary-key column name (usually `id`). The `db_delete` endpoint is suffixed with `/db_delete/` to keep it distinct from the soft-delete endpoint — override via `endpoint_names={"db_delete": "hard"}` to change to `/hard/{pk}` etc.

Cannot pass both `included_methods` and `deleted_methods` — raises `ValueError`.

---

## Per-method dependencies

For auth, rate limiting, audit logging, etc.:

```python
from fastapi import Depends
from .auth import require_admin, require_authenticated

crud_router(
    ...,
    read_deps=[Depends(require_authenticated)],
    read_multi_deps=[Depends(require_authenticated)],
    create_deps=[Depends(require_admin)],
    update_deps=[Depends(require_admin)],
    delete_deps=[Depends(require_admin)],
)
```

Each `*_deps` is a list of `Depends(...)` injected into that endpoint's signature. FastAPI evaluates them as usual.

---

## Auto-inject fields on create/update

Use `CreateConfig` / `UpdateConfig` to derive fields from the request automatically (timestamps, user IDs, audit fields). This keeps them out of the user-facing schema:

```python
from datetime import datetime, UTC
from fastapi import Depends
from fastcrud import CreateConfig, UpdateConfig

def now_utc() -> datetime:
    return datetime.now(UTC)

def current_user_id(user = Depends(get_current_user)) -> int:
    return user.id

crud_router(
    ...,
    create_schema=UserCreate,          # doesn't include created_by, created_at
    create_config=CreateConfig(
        auto_fields={
            "created_by": current_user_id,
            "created_at": now_utc,
        },
        exclude_from_schema=[],
    ),
    update_config=UpdateConfig(
        auto_fields={"updated_by": current_user_id},
    ),
)
```

`auto_fields` values are FastAPI dependencies — they can take `Depends(...)` args, request headers, etc.

---

## Customizing endpoint names

By default endpoints are mounted at `/`, `/{id}`, etc. To rename:

```python
crud_router(
    ...,
    path="/users",
    endpoint_names={
        "read": "fetch",          # GET /users/fetch/{id}
        "read_multi": "list",     # GET /users/list
        "create": "register",     # POST /users/register
    },
)
```

---

## Soft delete

Configure via the column-name parameters (defaults shown):

```python
crud_router(
    ...,
    is_deleted_column="is_deleted",
    deleted_at_column="deleted_at",
)
```

When **both** columns exist on the model:
- `DELETE /{id}` sets `is_deleted=True`, `deleted_at=now()`.
- All read queries auto-filter `is_deleted=False`.
- `DELETE /db/{id}` always hard-deletes.

To temporarily query soft-deleted rows, pass `is_deleted=True` as a kwarg on the underlying `FastCRUD` method (not exposed via `crud_router` by default).

---

## `EndpointCreator` — for advanced customization

Subclass `EndpointCreator` to add new endpoints or override existing ones. The override entry point is the **public** method `add_routes_to_router(...)`:

```python
from fastapi import Depends
from fastcrud import EndpointCreator, crud_router

class MyEndpointCreator(EndpointCreator):
    def add_routes_to_router(self, **kwargs):
        super().add_routes_to_router(**kwargs)    # keep the defaults
        self.router.add_api_route(
            "/me",
            self._me_endpoint(),
            methods=["GET"],
        )

    def _me_endpoint(self):
        async def endpoint(user = Depends(get_current_user)):
            async for db in self.session():
                return await self.crud.get(db, id=user.id)
        return endpoint

router = crud_router(
    ...,
    endpoint_creator=MyEndpointCreator,   # pass the CLASS, not an instance
)
```

The built-in endpoint-builder methods are private (single underscore prefix) and return the FastAPI handler callable — override them to change behavior of an existing endpoint:

| Method                | Builds                          |
|-----------------------|---------------------------------|
| `_create_item()`      | `POST {path}`                   |
| `_read_item()`        | `GET {path}/{pk}`               |
| `_read_items()`       | `GET {path}` (paginated list)   |
| `_update_item()`      | `PATCH {path}/{pk}`             |
| `_delete_item()`      | `DELETE {path}/{pk}`            |
| `_db_delete()`        | `DELETE {path}/db_delete/{pk}`  |
| `_create_cursor_validator()` | cursor query-param validator dependency used by custom cursor endpoints |

The route-registration loop is `add_routes_to_router(...)` — that's the public seam you override to add new routes or skip default ones.

---

## Response wrapper key

Multi-item endpoints wrap results in `{"data": [...], "total_count": N}` by default. To change the wrapper key:

```python
crud = FastCRUD(User, multi_response_key="items")
crud_router(..., crud=crud)
# now: {"items": [...], "total_count": N}
```

Pass the pre-built `FastCRUD` instance via `crud=` to apply.
