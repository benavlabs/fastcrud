# Pagination

Two strategies. Pick by access pattern. Then pick your `limit` carefully — `limit=None` is the most common source of latency cliffs in real FastCRUD code.

---

## Offset pagination — `get_multi(...)`

Best for: numbered pages, "page 5 of 17"-style UIs, admin panels.

```python
result = await crud.get_multi(db, offset=0, limit=20)
# {"data": [...], "total_count": 142}
```

For the next page:

```python
result = await crud.get_multi(db, offset=20, limit=20)
```

### Helpers

```python
from fastcrud import compute_offset, paginated_response, PaginatedListResponse

offset = compute_offset(page=3, items_per_page=20)   # → 40

result = await crud.get_multi(db, offset=offset, limit=20)
return paginated_response(crud_data=result, page=3, items_per_page=20)
# {"data": [...], "total_count": 142, "has_more": true, "page": 3, "items_per_page": 20}
```

### When `crud_router` exposes this

`GET /your-path` is paginated by default. Query params: `?page=N&itemsPerPage=M` (the `itemsPerPage` alias is camelCase, matching the OpenAPI/JSON convention — the snake_case `items_per_page` form is also accepted via `populate_by_name=True`).

---

## Cursor pagination — `get_multi_by_cursor(...)`

Best for: infinite scroll, activity feeds, anywhere "page N+1" doesn't make sense, and any list where rows are inserted/deleted frequently (offset paginates wrongly under concurrent writes).

```python
page1 = await crud.get_multi_by_cursor(
    db,
    limit=20,
    sort_column="created_at",
    sort_order="desc",
)
# {"data": [...], "next_cursor": "2026-05-20T10:30:00"}

page2 = await crud.get_multi_by_cursor(
    db,
    cursor=page1["next_cursor"],
    limit=20,
    sort_column="created_at",
    sort_order="desc",
)
```

`next_cursor` is `None` on the last page.

### Cursor validation (v0.22+, router-side only)

When the cursor endpoint is exposed by `EndpointCreator` (i.e. a custom endpoint built via subclass — `get_multi_by_cursor` isn't in the default `crud_router` endpoints), the router wraps the cursor parameter with a validator that:

- Rejects out-of-range integers (int32/int64 overflow) → HTTP 400
- Rejects malformed UUIDs → HTTP 400
- Rejects unparseable datetimes → HTTP 400
- Recognises SQLModel's `GUID` type for UUID columns

The direct `FastCRUD.get_multi_by_cursor(...)` method does **not** do this validation — it just builds the filter and runs the query, so a bad cursor surfaces as a database error. If you call the method directly from your own endpoint, validate the cursor yourself (or wire it through `EndpointCreator._create_cursor_validator`).

### When `crud_router` exposes this

Cursor pagination is **not** in the default `crud_router` endpoints. Add it via `EndpointCreator` subclass or as a custom endpoint:

```python
from fastapi import APIRouter, Depends
from fastcrud import CursorPaginatedRequestQuery, FastCRUD

router = APIRouter()
crud = FastCRUD(User)

@router.get("/users/feed")
async def feed(
    query: CursorPaginatedRequestQuery = Depends(),
    db: AsyncSession = Depends(get_async_session),
):
    return await crud.get_multi_by_cursor(
        db,
        cursor=query.cursor,
        limit=query.limit,
        sort_column=query.sort_column,
        sort_order=query.sort_order,
    )
```

`CursorPaginatedRequestQuery` is a pre-built Pydantic model with `cursor`, `limit`, `sort_column`, `sort_order` fields and FastAPI-friendly defaults.

---

## Don't mix strategies

A single endpoint should be one or the other. Switching strategies mid-list (e.g., starting with offset and then switching to cursor) requires holding two queries' worth of state on the client and rarely justifies the complexity.

---

## Picking `limit`

`get_multi(...)` / `get_multi_joined(...)` accept any positive integer or `None`. `None` means "fetch everything matching the WHERE clause." This is occasionally what you want and frequently what you don't.

### Default by surface

| Surface                                       | Default                                |
|-----------------------------------------------|----------------------------------------|
| HTTP list endpoint                            | `limit = min(requested, MAX_PAGE_LIMIT)` |
| Worker fetch of bounded child collection      | Explicit cap (`limit=RENDER_MAX_CLIPS`) |
| Bulk fan-out across a large table             | `get_multi_by_cursor(limit=BATCH_SIZE)` |
| Lookup by domain-bounded WHERE clause         | `limit=None` is OK (with sanity check) |

### When `limit=None` is safe

When a WHERE clause guarantees a bounded number of rows. Examples:

```python
# OK: bounded by clips-per-project (a few hundred at most)
await crud_clips.get_multi(db, limit=None, project_id=project.id)

# OK: bounded by len(tier_ids) which is small
await crud_tiers.get_multi(db, limit=None, id__in=tier_ids)

# OK: bounded by entitlements-per-user
await crud_entitlements.get_multi(db, limit=None, user_id=user.id)
```

### When `limit=None` is unsafe

When the WHERE clause doesn't bound the result, or when "bounded" today might not be tomorrow.

```python
# BAD: unbounded; grows with the user table forever
await crud_users.get_multi(db, limit=None)

# BAD: unbounded; grows with traffic
await crud_events.get_multi(db, limit=None, event_type="page_view")

# BAD: "bounded" by status, but pending tasks pile up
await crud_tasks.get_multi(db, limit=None, status="pending")
```

### The sanity-check pattern

When `limit=None` is justified, log when results unexpectedly explode. Catches drift before it becomes an incident:

```python
import logging
logger = logging.getLogger(__name__)

def handle_query_sanity(items: list, threshold: int, context: str) -> None:
    """Used on domain-bounded queries (limit=None with a WHERE clause)
    to surface unexpected growth without breaking the request."""
    if len(items) > threshold:
        logger.warning(
            "query growth: %d items > %d threshold (%s)",
            len(items), threshold, context,
        )

result = await crud_entitlements.get_multi(db, limit=None, user_id=user.id)
handle_query_sanity(result["data"], threshold=100, context="user_entitlements")
```

Pick a threshold ~5x the expected steady-state count. Don't raise — log and continue.

### Putting it all together — HTTP list endpoint

```python
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100

async def list_users(
    offset: int = 0,
    limit: int = DEFAULT_PAGE_LIMIT,
    db: AsyncSession = Depends(get_db),
) -> dict:
    limit = min(limit, MAX_PAGE_LIMIT)   # never trust client
    return await crud_users.get_multi(db, offset=offset, limit=limit)
```

---

## Total count is not free

`get_multi(...)` runs a `SELECT COUNT(*)` in addition to the data query. On large tables behind heavy filters, this can dominate the response time. Two options:

1. **Skip the count:** call `get_multi_by_cursor` (no count returned).
2. **Pass `return_total_count=False`** to `get_multi(...)` if you don't need it.

Per [PR #146](https://github.com/benavlabs/fastcrud/issues/146), an exact `total_count` is sometimes a foot-gun on tables with billions of rows. If your dataset is that large, drop the count.
