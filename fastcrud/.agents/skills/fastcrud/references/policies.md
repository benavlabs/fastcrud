# Row-level policies (v0.23+)

A `PolicyConfig` on a `FastCRUD` instance decides which operations are allowed and which rows each one reaches, like Postgres row-level security. Everything is denied unless granted.

```python
from fastcrud import AccessDenied, FastCRUD, PolicyConfig

post_crud = FastCRUD(Post)

def policy_for(user) -> PolicyConfig:
    return PolicyConfig(
        read=Post.published | (Post.owner_id == user.id),
        create=True,
        create_check=Post.owner_id == user.id,
        update=Post.owner_id == user.id,
        delete=Post.owner_id == user.id,
        read_only_columns={"owner_id"},
        hidden_columns={"internal_notes"},
    )

crud = post_crud.with_policy(policy_for(user))
```

`post_crud` is shared and has no policy; `with_policy` returns a copy and leaves it unchanged. `update_check` is omitted, so it defaults to the `update` condition.

## Fields

| Field | Takes | Meaning |
|---|---|---|
| `read` | `True` / `False` / conditions | rows reads reach |
| `create` | `True` / `False` only | conditions on created rows go in `create_check` |
| `update`, `delete` | `True` / `False` / conditions | rows reached, **among those `read` reaches** |
| `create_check`, `update_check` | conditions | written rows must match; `update_check` defaults to `update` |
| `hidden_columns` | names | never returned, filtered, sorted, written or named in `where=` |
| `read_only_columns` | names | readable, not writable |

## Behaviour to rely on

- **Out-of-reach rows behave as absent**: reads omit them, writes raise `NoResultFound`. Only a denied operation, a failed check, a forbidden column, or an upsert meeting an out-of-reach row raises `AccessDenied`.
- **The caller can't widen the policy**: `where=` and keyword filters are AND-ed with it; `where=true()` = every row in reach; writes with no filters are still refused.
- **Checks run in a savepoint** after a flush; a failing row is rolled back without discarding the caller's earlier session work, even with `commit=False`.
- **Written rows are returned even outside `read`.**
- **Hidden columns**: `text()`, `column()` and `literal_column()` in `where=` are refused when any column is hidden; `.any()`, `.has()`, `exists()` are fine. A response schema that *requires* a hidden field raises `AccessDenied` up front.
- **Server-written values** go in `server_data=` on `create` / `update` (e.g. `server_data={"owner_id": user.id}`): they win over the caller's data and skip the column rules, but the checks still see them.
- **Base model only**: joined rows are filtered only by `JoinConfig.where`; joined columns are never hidden; relationship data in a create/update raises `AccessDenied`. Load related records through `FastCRUD(Related).with_policy(...)` with `where=Related.fk == pk`.
- **Upserts**: new keys need `create` + `create_check`; conflicting rows must be in `update` reach (else `AccessDenied(reason="row")`) and pass `update_check`.

## `AccessDenied`

Plain exception (no HTTP). Attributes: `operation`, `model`, `reason` (`"operation"`, `"check"`, `"columns"`, `"row"`), `columns`, `action` (`"write"`, `"filter"`, `"return"` for column refusals, else `None`).

`NoResultFound` means the row doesn't exist or isn't the caller's to reach; `AccessDenied.reason` tells a denied operation from a failed check:

```python
from sqlalchemy.exc import NoResultFound
from fastcrud.exceptions.http_exceptions import ForbiddenException, NotFoundException

try:
    await crud.update(db, {"title": "New"}, id=post_id)
except NoResultFound:
    raise NotFoundException()
except AccessDenied as denied:
    raise ForbiddenException(detail=str(denied))
```

## Endpoints

`crud_router(..., policy=dependency)` resolves the policy per request (return `None` for no restriction). `AccessDenied` → 403, out of reach → 404, `auto_fields` are written as `server_data`. See `endpoints.md`.

## Pitfalls

- `PolicyConfig(read=[])` raises — use `True` for every row.
- Column names must be column attributes; typos and relationships raise `ValueError` when the policy is attached.
- SQLModel: write `col(Post.owner_id) == me` so type checkers accept the condition.
