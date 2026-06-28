---
name: fastcrud
description: Use when building or modifying CRUD endpoints with FastCRUD (the `fastcrud` PyPI package) in a FastAPI project — covers `FastCRUD`, `crud_router`, `EndpointCreator`, `FilterConfig`, `JoinConfig`, auto-relationship detection, the filter operator syntax (`__gte`, `__in`, `__ilike`, etc.), cursor pagination, soft delete, and how to avoid N+1 queries when fetching related data. Activate when the user mentions FastCRUD, `crud_router`, `FastCRUD()`, `FilterConfig`, `JoinConfig`, or asks how to build CRUD endpoints / generate REST APIs from SQLAlchemy or SQLModel models, even if the user doesn't name the library directly.
license: MIT
metadata:
  author: benav-labs
  package: fastcrud
---

# FastCRUD

FastCRUD generates async CRUD methods (and optionally CRUD endpoints) for SQLAlchemy 2.0 models inside a FastAPI app. The two entry points are:

- **`FastCRUD(Model)`** — the data-access class. Use this for per-row CRUD operations (single-record get / create / update / delete, paginated lists, joins with relationships). Real codebases use this in services, workers, and custom endpoints.
- **`crud_router(...)`** — returns an `APIRouter` with create/read/update/delete endpoints auto-wired. Use this only when you want generated endpoints; many apps skip it and hand-roll FastAPI routes on top of `FastCRUD` instances.

This skill covers the public API across `fastcrud >= 0.22`. SQLAlchemy is the default ORM in examples; SQLModel works the same way except where noted.

**Before reaching for FastCRUD, check the [When NOT to use FastCRUD](#when-not-to-use-fastcrud) section below.** Aggregate roll-ups, CTEs, `GROUP BY` projections, and bulk writes are SQLAlchemy's job — FastCRUD is for per-row CRUD.

---

## Canonical setup

The minimal viable pattern. Always start here, then add filters/joins/relationships as needed.

```python
# models.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase): pass

class Tier(Base):
    __tablename__ = "tier"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    tier_id = Column(Integer, ForeignKey("tier.id"))
    tier = relationship("Tier")
```

```python
# schemas.py
from pydantic import BaseModel, ConfigDict

class UserCreate(BaseModel):
    name: str
    email: str
    tier_id: int

class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    tier_id: int

class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
```

```python
# main.py
from fastapi import FastAPI
from fastcrud import crud_router
from .db import get_async_session
from .models import User
from .schemas import UserCreate, UserRead, UserUpdate

app = FastAPI()

user_router = crud_router(
    session=get_async_session,    # callable returning AsyncSession (FastAPI dependency)
    model=User,
    create_schema=UserCreate,
    update_schema=UserUpdate,
    select_schema=UserRead,        # used for read responses
    path="/users",
    tags=["users"],
)
app.include_router(user_router)
```

This produces `POST /users`, `GET /users/{id}`, `PATCH /users/{id}`, `DELETE /users/{id}`, `DELETE /users/db_delete/{id}` (hard delete), and `GET /users` (paginated list).

---

## Choosing the right method

When using `FastCRUD(Model)` directly, pick the method that matches the data shape you actually need. **Do not loop over `get_multi()` and call `get_joined()` per row** — that's the canonical N+1.

| Need                                        | Use                          |
|---------------------------------------------|------------------------------|
| Single row, no joins                        | `get(db, id=...)`            |
| Single row with related data                | `get_joined(db, id=...)`     |
| Paginated list, no joins                    | `get_multi(db, offset, limit)` |
| **Paginated list with related data**       | **`get_multi_joined(db, ...)`** |
| Cursor-paginated list (infinite scroll)     | `get_multi_by_cursor(db, ...)` |
| Insert                                      | `create(db, schema)`         |
| Insert-or-update by unique constraint       | `upsert(db, schema)` / `upsert_multi(db, list)` |
| Patch by filters                            | `update(db, schema, **kwargs)` |
| Soft delete (if configured)                 | `delete(db, **kwargs)`       |
| Hard delete (always removes row)            | `db_delete(db, **kwargs)`    |
| Count                                       | `count(db, **kwargs)`        |
| Boolean existence check                     | `exists(db, **kwargs)`       |

---

## Avoiding N+1 — the most common mistake

FastCRUD has two mechanisms to prevent N+1 when fetching related data. **Use one of them — never iterate.**

### 1. Auto-detect relationships (zero config, default for v0.21+)

Set `include_relationships=True` on `crud_router` (or `auto_detect_relationships=True` on the FastCRUD method directly). FastCRUD inspects the SQLAlchemy mapper, builds the joins, and threads them through a single query.

```python
crud_router(
    session=get_async_session,
    model=User,
    create_schema=UserCreate,
    update_schema=UserUpdate,
    select_schema=UserWithTier,           # schema includes nested tier field
    include_relationships=True,           # auto-include all relationships
    path="/users",
)
```

Or pass a list to include only specific relationships:

```python
include_relationships=["tier", "department"]   # whitelist
```

**Gotcha:** **One-to-many relationships are excluded by default** because they can return unbounded data. Opt in explicitly:

```python
include_relationships=True,
include_one_to_many=True,
default_nested_limit=10,    # cap nested rows per parent (uses SQL window function)
```

`default_nested_limit` uses `row_number() OVER (PARTITION BY ...)` at the database level — it does **not** fetch everything and slice in Python.

### 2. Manual `JoinConfig` (when you need explicit control)

For self-joins, aliases, custom join conditions, or per-join schemas:

```python
from fastcrud import FastCRUD, JoinConfig

orders_with_user = await crud.get_multi_joined(
    db=session,
    joins_config=[
        JoinConfig(
            model=User,
            join_on=Order.user_id == User.id,
            join_prefix="user_",         # avoid column collisions
            schema_to_select=UserRead,
        ),
    ],
    offset=0,
    limit=20,
)
```

For one-to-many with per-parent capping:

```python
JoinConfig(
    model=Article,
    join_on=Article.author_id == Author.id,
    relationship_type="one-to-many",
    sort_columns="created_at",
    sort_orders="desc",
    nested_limit=5,         # 5 most recent articles per author, computed in SQL
)
```

See `references/joins.md` for the full `JoinConfig` spec, polymorphic inheritance, and the auto-detection rules.

---

## `limit=None` — the second biggest footgun

`get_multi(...)` / `get_multi_joined(...)` accept `limit=None` to fetch **every matching row**. This is the single biggest cause of latency cliffs and OOMs in real-world FastCRUD code.

**Rule:** `limit=None` is only safe when a `WHERE` clause domain-bounds the result to a known-small set. Otherwise, always pass an explicit numeric `limit`.

### Safe — scoped by a WHERE clause

```python
# OK: scoped to a single project → bounded by N(clips per project)
clips = await crud_clips.get_multi(db, limit=None, project_id=project.id)

# OK: scoped to a small known set of IDs
tiers = await crud_tiers.get_multi(db, limit=None, id__in=tier_ids)
```

### Unsafe — unbounded growth

```python
# BAD: grows linearly with the user table forever
users = await crud_users.get_multi(db, limit=None)

# GOOD: HTTP endpoint with a hard ceiling
limit = min(requested_limit, MAX_PAGE_LIMIT)
users = await crud_users.get_multi(db, offset=offset, limit=limit)
```

### Pattern: sanity-check `limit=None` results

When `limit=None` is justified, log when results unexpectedly explode so drift is caught before it becomes an incident:

```python
def handle_query_sanity(items: list, threshold: int, context: str) -> None:
    if len(items) > threshold:
        logger.warning("query growth: %d > %d threshold (%s)",
                       len(items), threshold, context)

entitlements = await crud_entitlements.get_multi(db, limit=None, user_id=user.id)
handle_query_sanity(entitlements["data"], threshold=100, context="user_entitlements")
```

### Three-tier default

| Surface                                       | Default                                |
|-----------------------------------------------|----------------------------------------|
| HTTP list endpoint                            | `limit = min(requested, MAX_LIMIT)`    |
| Worker fetch of bounded child collection      | Explicit cap (`limit=RENDER_MAX_CLIPS`) |
| Bulk fan-out across a large table             | `get_multi_by_cursor(limit=BATCH)`     |

---

## When NOT to use FastCRUD

FastCRUD is the right tool for **per-row CRUD on a single model**, optionally with relationship joins. Drop to raw SQLAlchemy when the query shape is anything else. In a typical app the split is roughly 80/20 — most data access is FastCRUD's territory, but the 20% that isn't is *firmly* SQLAlchemy's.

| Query shape                                                              | Use instead                                       |
|--------------------------------------------------------------------------|---------------------------------------------------|
| Aggregate roll-up (`SUM`, conditional `COUNT FILTER WHERE`, `HAVING`)    | `select(func.sum(...), func.count().filter(...))` |
| CTEs, correlated subqueries, `EXISTS()` filters                          | `select(...).where(~exists().where(...))`         |
| `GROUP BY` with derived columns (`Project + count(clips)`)               | `select(Project, func.count(Clip.id)).group_by(Project.id)` |
| Bulk `UPDATE ... WHERE id IN (...)` with column-expression RHS           | `update(...).values(expires_at=Model.expires_at + interval)` |
| Bulk `INSERT` (more than ~10 rows)                                       | `db.execute(insert(Model), records)`              |
| Dialect-specific features (Postgres advisory locks, `array_agg`, JSONB)  | `text("SELECT pg_advisory_xact_lock(...)")` etc.  |
| Anything you'd reach for `joinedload` / `selectinload` / `contains_eager` for | FastCRUD `JoinConfig` — it covers this           |

**`crud_router` itself is optional.** Production codebases often use FastCRUD instances as the data-access layer behind hand-written FastAPI routes, skipping `crud_router` entirely so they can compose auth, validation, and business logic without subclassing `EndpointCreator`.

### Anti-patterns to refactor on sight

1. **Per-ID loop calling `crud.get()`** — classic N+1.
   ```python
   # BAD
   for tier_id in tier_ids:
       tier = await crud_tiers.get(db, id=tier_id)
   # GOOD — one query
   tiers = await crud_tiers.get_multi(db, id__in=tier_ids, limit=None)
   ```
2. **`db.scalar(select(Model).where(...).exists())`** — use `await crud.exists(**filters)`.
3. **`get_multi(limit=None)` then Python filtering** — push the filter into kwargs so the database does the work.
4. **`crud.get()` in a hot path without `schema_to_select`** — selects every column. Pass a minimal schema to cut payload size.
5. **Loop of `crud.create()`** — calls go one-at-a-time. For >10 rows use `crud.upsert_multi(...)` or raw `db.execute(insert(Model), records)`.

---

## Filter syntax

FastCRUD accepts filters as keyword arguments on every read/update/delete method, and via `FilterConfig` for `crud_router`'s `GET /list` endpoint.

### Operator suffix on the field name

The operator goes after a **double underscore**:

```python
await crud.get_multi(db, price__gte=10, name__ilike="%admin%", id__in=[1, 2, 3])
```

| Suffix         | SQL                | Notes                                |
|----------------|--------------------|--------------------------------------|
| (none)         | `=`                | bare field name is equality          |
| `__eq`         | `=`                | explicit form                        |
| `__ne`         | `<>`               |                                      |
| `__gt` `__gte` | `>` `>=`           |                                      |
| `__lt` `__lte` | `<` `<=`           |                                      |
| `__in`         | `IN (...)`         | value must be `list`/`tuple`/`set`   |
| `__not_in`     | `NOT IN (...)`     | same                                 |
| `__between`    | `BETWEEN a AND b`  | value must be a 2-element sequence   |
| `__like`       | `LIKE`             | case-sensitive                       |
| `__ilike`      | `ILIKE`            | case-insensitive (Postgres)          |
| `__startswith` `__endswith` `__contains` | substring match (auto-wraps with `%`) | pass the bare substring — do NOT add `%` yourself |
| `__is` `__is_not` | `IS` `IS NOT`   | for `NULL` checks                    |
| `__match`      | engine-specific full-text | depends on dialect            |

### Joined filters

Walk relationships with `.`:

```python
await crud.get_multi_joined(db, **{"tier.name__eq": "premium"})
```

Or, equivalently, build the kwargs dict literal-style. In `FilterConfig`:

```python
FilterConfig({
    "name__ilike": None,           # query string param: ?name__ilike=foo
    "price__gte": None,
    "tier.name": None,             # joined filter — auto-includes the tier relationship
})
```

### Bool coercion

`?filter=false` is correctly parsed (case-insensitive: `False`, `FALSE`, `0`, `true`, `TRUE`, `1`). No need to pre-coerce.

See `references/filters.md` for `FilterConfig` deep-dive, `Depends(...)` filters (filter by current user automatically), and custom operator registration.

---

## Return semantics — read carefully

The biggest footgun. Several methods have non-obvious defaults that changed in v0.20:

### `create()`

- Without `schema_to_select`: **returns `None`** (since v0.20.0 — was the model in earlier versions).
- With `schema_to_select`: returns a `dict`.
- With `schema_to_select` + `return_as_model=True`: returns a Pydantic instance.

```python
await crud.create(db, UserCreate(...))                                     # → None
await crud.create(db, UserCreate(...), schema_to_select=UserRead)          # → dict
await crud.create(db, UserCreate(...), schema_to_select=UserRead,
                  return_as_model=True)                                    # → UserRead
```

`return_as_model=True` **requires** `schema_to_select` (else raises `ValueError`).

### `get()` / `get_multi()` / `get_joined()` / etc.

Same pattern: `return_as_model=True` requires `schema_to_select`. Without `schema_to_select`, all columns are returned as a dict.

### `schema_to_select` propagates the subclass type (v0.22+)

If you pass a subclass override per-call, the return type narrows correctly — no manual `cast()` needed:

```python
class UserAdminRead(UserRead):
    role: str

result = await crud.get(db, id=1, schema_to_select=UserAdminRead, return_as_model=True)
# result is typed as UserAdminRead | None, not UserRead | None
```

### `update(return_columns=...)` must be a list (v0.22+)

`return_columns=True` raises `ValueError`. Pass a list:

```python
await crud.update(db, schema=UserUpdate(name="X"), return_columns=["id", "name"], id=1)
```

### `commit=False` for transactions

All write methods take `commit=False`. Use it when chaining multiple operations under a single transaction; commit once at the end with `await db.commit()`.

---

## Soft delete

Configure at the FastCRUD level:

```python
crud = FastCRUD(User, is_deleted_column="is_deleted", deleted_at_column="deleted_at")

await crud.delete(db, id=1)       # soft: sets is_deleted=True, deleted_at=now
                                  #       (requires both columns on the model)
await crud.db_delete(db, id=1)    # hard: actual DELETE — always removes the row
```

**Reads do NOT auto-filter soft-deleted rows.** This is the most common surprise — you have to filter explicitly:

```python
active = await crud.get_multi(db, is_deleted=False)
```

If you want every read to exclude soft-deleted rows automatically, wrap `FastCRUD` in a subclass that overrides `get`/`get_multi`/etc. to add the filter, or pass `is_deleted=False` as a dependency-based default via `FilterConfig` on `crud_router`.

---

## Gotchas (read these before writing code)

1. **`limit=None` fetches every matching row.** Only safe when a `WHERE` clause domain-bounds the result. See [the dedicated section](#limitnone--the-second-biggest-footgun).
2. **Async session only.** All methods require `AsyncSession`. Sync `Session` support for `count()`/`exists()` is in flight (PR #333) but not merged.
3. **One-to-many relationships excluded from auto-detect.** Add `include_one_to_many=True` and set `default_nested_limit` (or `nested_limit` per `JoinConfig`).
4. **Filter operator separator is `__` (two underscores).** `price_gte` is just a column name `price_gte`; `price__gte` is `price >= ...`.
5. **`create()` returns `None` by default** (no schema_to_select). Don't expect the model back.
6. **`return_as_model=True` without `schema_to_select` raises.** Always pair them.
7. **`update(return_columns=True)` raises** since v0.22.0. Use a list of column names.
8. **Joined filters need `.` not `__` for the relationship part.** `"tier.name__eq"`, not `"tier__name__eq"`.
9. **`session` parameter on `crud_router` is a callable** (a FastAPI dependency that yields/returns an `AsyncSession`), not a session instance.
10. **`include_relationships` and `joins_config` are mutually exclusive** on `crud_router`. Pick one.
11. **Joined-table polymorphic inheritance is supported.** v0.22.0 fixed auto-aliasing when primary and joined share a base table; v0.22.2 fixed inherited columns missing from `create()` responses.
12. **SQLModel joined-table inheritance is fragile** — redeclaring `id` in a subclass breaks the SQLModel metaclass mapping. Stick to plain SQLAlchemy if you need polymorphism.
13. **Python 3.10+ required.** Python 3.14 works since v0.22.1 (earlier versions crashed at import under PEP 649).

---

## When to drill into references

Load these on demand:

- `references/methods.md` — full signatures and overloads for every `FastCRUD` method
- `references/filters.md` — every operator, `FilterConfig`, dependency-based filters, custom operators
- `references/joins.md` — `JoinConfig` fields, auto-detection rules, `nested_limit` mechanics, polymorphism
- `references/pagination.md` — offset vs cursor pagination, `paginated_response`, `CursorPaginatedRequestQuery`
- `references/endpoints.md` — `crud_router` full signature, `EndpointCreator` subclassing, `included_methods`, per-method dependencies, soft delete, custom endpoint names

---

## SQLModel notes (when the project uses SQLModel instead)

The library works the same way; substitute the model definition:

```python
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    __tablename__ = "user"
    id: int | None = Field(default=None, primary_key=True)
    name: str
    email: str = Field(unique=True)
    tier_id: int | None = Field(default=None, foreign_key="tier.id")
```

Schemas can also be `SQLModel` (without `table=True`) instead of `BaseModel`. The CRUD layer is identical because SQLModel inherits from SQLAlchemy. One caveat:

- **Joined-table inheritance is unreliable** — see gotcha #11. For polymorphic models, drop to plain SQLAlchemy.
