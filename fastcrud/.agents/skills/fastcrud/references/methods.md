# FastCRUD method reference

Every method on `FastCRUD(Model)`, with signature shape, return type, and the one or two non-obvious things to know about each.

All methods are `async` and take an `AsyncSession` as `db`. Filter kwargs use the `field__op` syntax (see `filters.md`).

---

## Reads

### `get(db, schema_to_select=None, return_as_model=False, **kwargs) -> dict | Model | None`

Single row by filters. Returns `None` if not found (does not raise).

```python
user = await crud.get(db, id=1)                                    # dict
user = await crud.get(db, id=1, schema_to_select=UserRead,
                       return_as_model=True)                        # UserRead | None
```

### `get_joined(db, joins_config=None, auto_detect_relationships=False, schema_to_select=None, return_as_model=False, nest_joins=False, **kwargs)`

Single row plus joined data. Set `auto_detect_relationships=True` (or pass `joins_config=[...]`) to control which related models are joined.

```python
user = await crud.get_joined(db, auto_detect_relationships=True, nest_joins=True, id=1)
# {"id": 1, "name": "...", "tier": {"id": 2, "name": "premium"}}
```

**Default `nest_joins=False`** on the direct method — joined columns are flattened with `join_prefix`. Pass `nest_joins=True` for the nested-dict shape shown above. Note this is the opposite of `crud_router`'s default (`nest_joins=True`).

### `get_multi(db, offset=0, limit=100, schema_to_select=None, return_as_model=False, sort_columns=None, sort_orders=None, **kwargs)`

Offset-paginated list. Returns `{"data": [...], "total_count": N}` by default; the wrapper key is configurable via `multi_response_key`.

```python
result = await crud.get_multi(db, offset=0, limit=20, tier_id=1,
                              sort_columns=["created_at"], sort_orders=["desc"])
# {"data": [...], "total_count": 42}
```

**`limit=None` fetches every matching row.** Only safe when a `WHERE` clause domain-bounds the result (e.g., `project_id=...`, `id__in=[...]`). Otherwise pass an explicit numeric limit. For tables that grow unboundedly, use `get_multi_by_cursor` for batched processing. See `pagination.md` for the sanity-check pattern.

### `get_multi_joined(db, joins_config=None, auto_detect_relationships=False, ...)`

Same as `get_multi` but with joins. **This is what you use to avoid N+1.** Per-join filters live inside each `JoinConfig.filters`.

```python
result = await crud.get_multi_joined(
    db,
    auto_detect_relationships=["tier", "articles"],
    nested_limit=5,           # cap one-to-many at 5 nested rows per parent (SQL window function)
    offset=0, limit=20,
    tier_id=1,
)
```

### `get_multi_by_cursor(db, cursor=None, limit=100, sort_column="id", sort_order="asc", schema_to_select=None, ...)`

Cursor pagination — efficient for infinite-scroll. Returns `{"data": [...], "next_cursor": <value-or-null>}`.

```python
page1 = await crud.get_multi_by_cursor(db, limit=20, sort_column="created_at", sort_order="desc")
page2 = await crud.get_multi_by_cursor(db, cursor=page1["next_cursor"], limit=20,
                                       sort_column="created_at", sort_order="desc")
```

Cursor values are validated against the column type **only when the method is exposed through a router-generated endpoint** (the cursor validator is set up in `EndpointCreator`, not in `FastCRUD.get_multi_by_cursor` itself). When you build your own endpoint around the direct method, validate the incoming cursor yourself or surface the database-level error to the caller. Validation recognises plain integers, UUIDs (including SQLModel's `GUID` type), and ISO datetimes.

### `count(db, joins_config=None, **kwargs) -> int`

Row count, with optional joins. `joins_config` can include `CountConfig` for subquery-based related counts.

### `exists(db, **kwargs) -> bool`

Cheap existence check (`SELECT ... LIMIT 1`). Returns `True` if any row matches the filters, `False` otherwise.

### `select(...) -> Select`

Returns the SQLAlchemy `Select` statement without executing. Use when you need to compose further (raw `.execute()`, custom CTE, etc.).

---

## Writes

### `create(db, object, commit=True, schema_to_select=None, return_as_model=False)`

Insert one row. Returns `None` by default (since v0.20.0) — pass `schema_to_select` to get back data.

```python
await crud.create(db, UserCreate(name="X"))                                # None
created = await crud.create(db, UserCreate(name="X"),
                             schema_to_select=UserRead,
                             return_as_model=True)                          # UserRead
```

**Inherited columns:** v0.22.2 fixed a bug where columns from parent tables in joined-table inheritance were missing from the response dict. If you hit a `KeyError` on an inherited column, upgrade.

### `upsert(db, instance, schema_to_select=None, return_as_model=False)`

Insert-or-update by primary key. **Implementation is two round-trips** — issues a `get` first, then `create` or `update` depending on whether a row matched. There is no `commit` parameter and no dialect-specific `ON CONFLICT` clause — that's `upsert_multi`. Use this only for single instances where simplicity outweighs the extra round-trip.

### `upsert_multi(db, instances: list, commit=False, return_columns=None, schema_to_select=None, return_as_model=False, update_override=None, **kwargs)`

Batch upsert via dialect-specific `ON CONFLICT` (Postgres / SQLite) or `ON DUPLICATE KEY UPDATE` (MySQL). Single round-trip.

- **`commit=False` default** (different from `create`/`update`) — caller is responsible for committing.
- **`update_override={...}`** overrides the values applied on conflict (otherwise the conflicting row is updated to the values in the corresponding `instance`).
- **`return_columns=["id", "name"]`** captures the listed columns via `RETURNING`. Returns a dict shaped `{"created": [...], "updated": [...]}` keyed by which path each row took.
- **MySQL caveat:** doesn't support `RETURNING`, so passing `return_columns` / `schema_to_select` / `return_as_model` / filter kwargs raises `ValueError` on MySQL.
- Unsupported dialects raise `NotImplementedError`.

### `update(db, object, commit=True, return_columns=None, schema_to_select=None, return_as_model=False, allow_multiple=False, **kwargs)`

Patch by filters. `kwargs` select the row(s); `object` (Pydantic model or dict) carries the new values.

```python
await crud.update(db, UserUpdate(name="Y"), id=1)
```

- **`return_columns` must be a list** (`return_columns=True` raises since v0.22.0). Pass `["id", "name"]` to get those columns back as a dict.
- **`allow_multiple=False` (default)** — if the filter matches more than one row, raises. Set `True` to bulk-update.
- **`updated_at` auto-update**: if your model has an `updated_at` column (configurable name), it's set to `datetime.now(UTC)` automatically on every update.

### `delete(db, db_row=None, filters=None, allow_multiple=False, commit=True, **kwargs)`

Soft delete if the configured soft-delete columns exist on the model; otherwise hard delete. Same `allow_multiple` semantics as `update`.

Two execution paths with slightly different rules:

- **Filter path** (no `db_row` passed) — sets whichever of `is_deleted_column` / `deleted_at_column` actually exists in `model_col_names`; falls back to hard `DELETE` only when neither exists.
- **`db_row` path** (preloaded instance passed) — requires **both** columns to be present on the instance for soft delete; otherwise hard-deletes via `session.delete(db_row)`.

You almost always want the filter path. The `db_row` path exists for cases where you've already loaded the row for another reason.

### `db_delete(db, allow_multiple=False, commit=True, **kwargs)`

Always hard delete (`DELETE FROM ...`), regardless of soft-delete configuration.

---

## Schema-controlled column selection

`schema_to_select` does **column selection**, not type coercion. The query selects only the columns listed in the schema, which is faster than `SELECT *`. Combined with `return_as_model=True`, you also get validated Pydantic instances back.

```python
class UserSummary(BaseModel):
    id: int
    name: str

# Only SELECTs id, name — not email, tier_id, etc.
summary = await crud.get(db, id=1, schema_to_select=UserSummary)
# {"id": 1, "name": "..."}
```

This is the main lever for reducing payload size on read endpoints.

---

## Type-narrowing pattern (v0.22+)

If you want per-call type narrowing (e.g., admin endpoint that returns a superset schema), pass a subclass of the class-level `select_schema`. The TypeVar on the method propagates the subclass through to the return type:

```python
crud = FastCRUD(User)  # default SelectSchemaType is unset

class UserRead(BaseModel): ...
class UserAdminRead(UserRead):
    role: str
    last_login: datetime

result = await crud.get(db, id=1, schema_to_select=UserAdminRead, return_as_model=True)
# typed as UserAdminRead | None
```

No `cast()` needed — the previous version required it.
