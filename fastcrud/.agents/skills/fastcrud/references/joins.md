# Joins

Everything about loading related data — auto-detection, `JoinConfig`, `nested_limit`, polymorphic inheritance.

---

## Two ways to join

Pick one. They cannot be combined on the same `crud_router` call.

### Auto-detect (default, recommended)

```python
crud_router(
    ...,
    include_relationships=True,            # all relationships
    # or:
    include_relationships=["tier", "dept"], # whitelist
)
```

FastCRUD inspects the SQLAlchemy mapper, discovers all `relationship(...)` definitions, walks foreign keys bidirectionally, generates `JoinConfig` objects, and applies left joins.

**Defaults:**
- All `*-to-one` relationships are included.
- **One-to-many is excluded** unless you explicitly opt in.
- Multiple relationships to the same table get distinct aliases automatically.

### Manual `JoinConfig`

```python
from fastcrud import JoinConfig

joins = [
    JoinConfig(
        model=Tier,
        join_on=User.tier_id == Tier.id,
        join_prefix="tier_",
        schema_to_select=TierRead,
        join_type="left",          # or "inner"
    ),
]

crud_router(..., joins_config=joins)
```

Use this for: self-joins, custom join conditions, per-join schemas, per-join filters, explicit aliasing.

---

## `JoinConfig` fields

| Field               | Required | Default        | Notes                                                            |
|---------------------|----------|----------------|------------------------------------------------------------------|
| `model`             | yes      | —              | SQLAlchemy model to join                                         |
| `join_on`           | yes      | —              | SQLAlchemy expression (`A.id == B.a_id`)                         |
| `join_prefix`       | no       | `None`         | Column prefix in flat results (no effect when `nest_joins=True`) |
| `schema_to_select`  | no       | `None`         | Limit columns selected from this join                            |
| `join_type`         | no       | `"left"`       | `"left"` or `"inner"`                                            |
| `alias`             | no       | `None`         | Pass an `aliased(Model, flat=True)` for self-joins / polymorphic |
| `filters`           | no       | `None`         | Dict of filters applied to this join's table                     |
| `relationship_type` | no       | `"one-to-one"` | Set `"one-to-many"` for collections                              |
| `sort_columns`      | no       | `None`         | Sort nested items (one-to-many only)                             |
| `sort_orders`       | no       | `None`         | `"asc"` / `"desc"` matching `sort_columns`                       |
| `nested_limit`      | no       | `None`         | Cap nested items per parent (uses SQL window function)           |

---

## One-to-many and `nested_limit`

The N+1 trap for collections: naïve code fetches the parent, then loops to fetch children. **Don't.** Use `nested_limit` on `JoinConfig` (or `default_nested_limit` on `crud_router`):

```python
JoinConfig(
    model=Article,
    join_on=Article.author_id == Author.id,
    relationship_type="one-to-many",
    sort_columns="created_at",
    sort_orders="desc",
    nested_limit=10,                       # 10 most recent articles per author
)
```

FastCRUD rewrites this as a `ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY created_at DESC)` subquery and filters `rn <= 10`. **The capping happens at the database level**, not in Python — even with 10,000 authors, only 100,000 rows transit the wire (10 per author), not the unbounded collection.

For `crud_router`:

```python
crud_router(
    ...,
    include_relationships=True,
    include_one_to_many=True,
    default_nested_limit=10,
)
```

---

## Self-joins and aliases

Use an explicit alias:

```python
from sqlalchemy.orm import aliased

manager_alias = aliased(User, flat=True)

joins = [
    JoinConfig(
        model=User,
        alias=manager_alias,
        join_on=Employee.manager_id == manager_alias.id,
        join_prefix="manager_",
        schema_to_select=UserRead,
    ),
]
```

`flat=True` is important for SQL correctness when the alias is used in nested joins.

---

## Polymorphic / joined-table inheritance

SQLAlchemy joined-table inheritance works with FastCRUD. v0.22.0 fixed a long-standing bug where the join target wasn't aliased when the primary and joined models shared a base table — that's now automatic.

```python
class Entity(Base):
    __tablename__ = "entity"
    id = Column(Integer, primary_key=True)
    type = Column(String(50))
    __mapper_args__ = {"polymorphic_on": "type", "polymorphic_identity": "entity"}

class Project(Entity):
    __tablename__ = "project"
    id = Column(Integer, ForeignKey("entity.id"), primary_key=True)
    name = Column(String)
    contract_id = Column(Integer, ForeignKey("contract.id"))
    __mapper_args__ = {"polymorphic_identity": "project"}

# This Just Works since v0.22.0
crud = FastCRUD(Project)
await crud.get_joined(db, auto_detect_relationships=True, id=1)
```

**v0.22.2 also fixed inherited columns missing from `create()` responses** — earlier versions returned only the child table's columns, silently dropping anything from the parent (including the polymorphic discriminator).

### SQLModel + polymorphism = avoid

SQLModel's metaclass treats redeclared `id` fields in subclasses as shadowing the parent attribute, breaking the SQLAlchemy mapping. The model loads, but `inspect(cls)` fails with `NoInspectionAvailable`. **Use plain SQLAlchemy if you need polymorphic inheritance.**

---

## `CountConfig` — counts without joining the rows

When you want `users` plus a count of their `posts` but don't want to actually join and nest the posts:

```python
from fastcrud import CountConfig

result = await crud.get_multi_joined(
    db,
    joins_config=[],
    count_configs=[
        CountConfig(
            model=Post,
            join_on=Post.user_id == User.id,
            alias="post_count",
        ),
    ],
)
# each user dict has a "post_count" field
```

Implemented as a scalar subquery, so every primary row is returned (with `0` for parents with no children).

---

## `nest_joins` — nested dicts vs flat columns

- `nest_joins=True`: joined data is nested under the relationship name: `{"id": 1, "tier": {"name": "premium"}}`
- `nest_joins=False`: joined columns are flattened with `join_prefix`: `{"id": 1, "tier_name": "premium"}`

**The default differs depending on the entry point:**

| Entry point                                              | Default       |
|----------------------------------------------------------|---------------|
| `crud_router(...)` / `EndpointCreator.__init__(...)`     | `True` (nest) |
| `FastCRUD(Model).get_joined(...)` / `get_multi_joined(...)` (direct call) | `False` (flat) |

When calling FastCRUD methods directly, pass `nest_joins=True` explicitly if you want the nested shape — clients usually expect it. The router defaults to nesting because that's the typical API response shape; the direct method defaults to flat because it's the cheaper structure when you're consuming the rows in Python.

**One-to-many always requires `nest_joins=True`** — `nest_joins=False` with a one-to-many `JoinConfig` raises `ValueError`.
