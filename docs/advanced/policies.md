# Row-Level Policies

A `PolicyConfig` decides which operations a caller may perform on a model and which rows each one reaches, in the manner of Postgres row-level security. It's set on a `FastCRUD` instance, so every read and write through that instance applies it: a caller who may only see their own posts never gets anyone else's, whether they list, count, filter, update or delete.

```python
from fastcrud import FastCRUD, PolicyConfig

post_crud = FastCRUD(Post)

async def list_my_posts(db, me):
    policy = PolicyConfig(
        read=Post.published | (Post.owner_id == me),
        create=True,
        create_check=Post.owner_id == me,
        update=Post.owner_id == me,
    )
    return await post_crud.with_policy(policy).get_multi(db)
```

Without a policy, the default, `FastCRUD` works as it always has.

## Setting a Policy

Pass a policy to the constructor, or get a copy of an existing instance under another policy with `with_policy`:

```python
post_crud = FastCRUD(Post)

restricted = post_crud.with_policy(PolicyConfig(read=Post.owner_id == me))
unrestricted = restricted.with_policy(None)
```

`with_policy` returns a new instance and never changes the original, so an instance shared across requests can hand each request its own policy. The copy keeps every other setting, such as `custom_filters` and the soft delete columns. Building a `PolicyConfig` and a `FastCRUD` per request costs tens of microseconds, mostly validating the policy's conditions, and no database work: negligible next to a query.

## `PolicyConfig`

Every operation is denied unless the policy grants it, so `PolicyConfig()` denies everything.

| Field | Takes | Meaning |
|---|---|---|
| `read` | `True`, `False` or conditions | The rows reads reach |
| `create` | `True` or `False` | Whether rows may be created |
| `update` | `True`, `False` or conditions | The rows updates reach, among those `read` reaches |
| `delete` | `True`, `False` or conditions | The rows deletes reach, among those `read` reaches |
| `create_check` | Conditions | What a created row must meet |
| `update_check` | Conditions | What an updated row must meet; defaults to the `update` condition |
| `hidden_columns` | Column names | Columns the caller never sees, uses or writes |
| `read_only_columns` | Column names | Columns the caller may see but not write |

Conditions are what the methods' `where=` takes: one SQLAlchemy condition, or a list of them that must all hold. `True` means every row and `False` denies the operation; an empty list is refused, since `True` already says "every row".

`create` takes only `True` or `False`, since there are no existing rows to reach; conditions a created row must meet go in `create_check`.

## Rows Outside the Policy

Rows outside a policy behave as if they weren't there:

- **Reads** (`select`, `get`, `exists`, `count`, `get_multi`, `get_multi_by_cursor`, `get_joined`, `get_multi_joined`, and their `total_count`) leave them out. `select()` returns a statement that already carries the `read` condition.
- **Writes** raise `NoResultFound`, as for a row that doesn't exist. `update` reaches only rows matching `read` and `update`, and `delete` and `db_delete` only rows matching `read` and `delete`, as Postgres does once a statement has a `WHERE`. The counts behind `allow_multiple` see only those rows too.

The caller's `where=` and keyword filters are AND-ed with the policy and can never widen it. `where=true()` means every row the policy reaches, and a write with no filters at all is still refused.

A soft delete is a delete: it's governed by `delete`, not `update`. `delete(db_row=...)` checks the loaded row against `delete` first.

`AccessDenied` is raised only when the policy refuses something outright: a denied operation, a failed check, a column the caller may not use, or an upsert meeting an existing row outside its reach, which it can't treat as absent because inserting it would collide. A read, update or delete that can't reach a row never raises it. See [`AccessDenied`](#accessdenied).

## Checks

`create_check` and `update_check` are conditions a written row must meet afterwards, as Postgres's `WITH CHECK`. They're checked after a flush, in a savepoint: a row that fails is rolled back and `AccessDenied` is raised, without discarding the caller's earlier work in the session, even with `commit=False`.

When `update_check` is omitted, it's the `update` condition: an updated row must still be within `update`'s reach. With `update=Post.owner_id == me`, nobody can hand a post over to someone else.

A row written outside `read` is still returned by `create` and `update`, since it's the caller's own write and the checks already let it through. Postgres would refuse the `RETURNING` there.

## Column Rules

### `hidden_columns`

Hidden columns are left out of everything reads, `create` and `update` return, including joined and nested results. The caller may not:

- filter or sort by them, with keyword filters, `sort_columns` or the cursor's `sort_column`;
- name them in `where=`, a `JoinConfig.where` or a `CountConfig.where`. Conditions are checked inside subqueries and through aliases of the model. Raw SQL, such as `text()`, `column()` or `literal_column()`, can't be checked, so it's refused whenever the policy hides columns. SQLAlchemy's own subqueries, such as `.any()`, `.has()` and `exists()`, work as usual;
- write them.

Using a hidden column raises `AccessDenied` with `operation="read"`, even inside an update or delete, since it would reveal the column's values. A response model that requires a hidden field is refused before any query; a model where it's optional just leaves it out.

The policy's own conditions may use hidden columns: `read=Post.owner_id == me` works with `owner_id` hidden. Joined reads still nest by a hidden primary key and link `nested_limit` joins by a hidden column, then drop them from the results.

### `read_only_columns`

Read-only columns may be read but not written. A `create` or `update` that sets one raises `AccessDenied`, even when it sets it to `None`. A field the caller leaves unset isn't written, so the database default applies. On SQLModel, a table model fills every field with its own default when it's built, so the model's default is written instead; the schema's default never is.

Column names are attribute names. A name that isn't a column of the model, such as a typo or a relationship, raises `ValueError` when the policy is attached.

### Server-Written Values

Column rules restrict the caller's input only. The columns FastCRUD writes itself, such as `updated_at` and the soft delete columns, are written as usual. Server code that sets a read-only or hidden column passes it as `server_data`:

```python
await post_crud.with_policy(policy).create(
    db,
    PostCreate(title="Hello"),
    server_data={"owner_id": me},
)
```

`server_data` takes column values only. They win over the caller's values for the same field and are exempt from the column rules, but they're part of the row `create_check` and `update_check` see. A hidden unique column, such as an email the caller may not see, can only be written this way.

## Upserts

`upsert` and `upsert_multi` look the keys up first, locking existing rows with `SELECT ... FOR UPDATE` where the database supports it:

- New keys need `create`, and the rows must pass `create_check`.
- Conflicting rows must be within `update`'s reach (`read` and `update`) and pass `update_check`.
- A conflicting row outside that reach raises `AccessDenied` with `reason="row"`. Treating it as absent would mean inserting it, which would collide; Postgres refuses an `ON CONFLICT DO UPDATE` whose row fails the update policy the same way.
- The column rules apply to each record and to `update_override`, and unwritable columns are left out of both the insert and the conflict update.
- `upsert_multi`'s own `where=` keeps its meaning: conflicting rows it excludes are silently skipped.

PostgreSQL and SQLite enforce the reach in `ON CONFLICT DO UPDATE ... WHERE` and check every key that wasn't written, so a conflicting row another transaction inserts after the lookup is refused too; such a row gets `create_check`. MySQL has no such `WHERE`, so the locked rows are checked before the upsert. There, `FOR UPDATE` also blocks concurrent inserts of the missing keys under `REPEATABLE READ`, MySQL's default; under `READ COMMITTED`, a row inserted in that gap is caught only by `update_check`. On MySQL, `upsert_multi` under a policy with `create_check` needs every record's primary key, since it can't return the rows it created.

## Related Records and Joins

A policy covers its own model only:

- **Joined models' rows aren't filtered** unless the join's [`JoinConfig.where`](joins.md#choosing-joined-rows-with-joinconfigwhere) says so, and their columns are never hidden. A join's `JoinConfig.schema_to_select` limits what it shows. A self-referential join shows the joined rows' hidden columns, as it shows rows outside `read`.
- **Relationship data in a create or update is refused** with `AccessDenied`, since the policy can't vouch for related records. Write them through the related model's own `FastCRUD` and policy.
- **To load a record's related records under their own policy**, use the related model's instance:

```python
comments = await FastCRUD(Comment).with_policy(comment_policy).get_multi(
    db, where=Comment.post_id == post_id
)
```

## `AccessDenied`

`AccessDenied`, exported from `fastcrud`, is a plain exception with no HTTP dependency.

| Attribute | Meaning |
|---|---|
| `operation` | `"read"`, `"create"`, `"update"` or `"delete"` |
| `model` | The model the operation was on |
| `reason` | `"operation"`: the policy denies the operation. `"check"`: a written row failed its check. `"columns"`: the caller tried to write, filter on, sort by or get back a column they may not. `"row"`: an upsert met an existing row outside its reach |
| `columns` | The columns involved, for `reason="columns"` |
| `action` | For `reason="columns"`: `"write"`, `"filter"` or `"return"`; otherwise `None` |

Its message says what was refused, for example "update on Post is not allowed by the policy", "the updated Post doesn't pass update_check", "update on Post may not write: owner_id" or "read on Post may not filter or sort by: owner_id".

## Policies in Endpoints

`crud_router` and `EndpointCreator` take `policy=`, a FastAPI dependency returning the request's policy:

```python
from fastapi import Depends
from fastcrud import CreateConfig, PolicyConfig, crud_router

def post_policy(user: User = Depends(get_current_user)) -> PolicyConfig | None:
    if user.is_admin:
        return None
    return PolicyConfig(
        read=Post.published | (Post.owner_id == user.id),
        create=True,
        update=Post.owner_id == user.id,
        delete=Post.owner_id == user.id,
        read_only_columns={"owner_id"},
    )

def current_user_id(user: User = Depends(get_current_user)) -> int:
    return user.id

router = crud_router(
    session=get_session,
    model=Post,
    create_schema=PostCreate,
    update_schema=PostUpdate,
    path="/posts",
    tags=["posts"],
    policy=post_policy,
    create_config=CreateConfig(auto_fields={"owner_id": current_user_id}),
)
```

`owner_id` is read-only for the caller, so the endpoint sets it from the current user through `auto_fields`, which are written as `server_data`. The policy dependency is resolved per request, like the `*_deps` callables, and returning `None` lifts the restriction for that request. `AccessDenied` becomes a 403 with its message as the detail, and rows outside the policy become 404s. See [Row-Level Policies in Endpoints](endpoint.md#row-level-policies) for how the generated endpoints treat `auto_fields`, unique values and read-backs.

## SQLModel

SQLModel types a model's fields as their Python values, so to a type checker `Post.owner_id == me` is a `bool`. Wrap the field in `col()` for policy conditions, as for `where=`:

```python
from sqlmodel import col

PolicyConfig(read=col(Post.owner_id) == me)
```
