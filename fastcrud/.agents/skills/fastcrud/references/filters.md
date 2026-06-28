# Filters

Complete reference for filtering — operator syntax, `FilterConfig` (the `crud_router` query-param surface), dependency-based filters, custom operators.

---

## Operator reference

Every operator is registered as a suffix after `__` on a field name.

| Suffix         | SQL / behaviour                                | Value shape          |
|----------------|------------------------------------------------|----------------------|
| (none)         | `=` (equality is the default)                  | scalar               |
| `__eq`         | `=`                                            | scalar               |
| `__ne`         | `<>`                                           | scalar               |
| `__gt` `__gte` | `>` `>=`                                       | scalar               |
| `__lt` `__lte` | `<` `<=`                                       | scalar               |
| `__is`         | `IS` (use for `None`/`True`/`False`)           | scalar               |
| `__is_not`     | `IS NOT`                                       | scalar               |
| `__in`         | `IN (...)`                                     | list / tuple / set   |
| `__not_in`     | `NOT IN (...)`                                 | list / tuple / set   |
| `__between`    | `BETWEEN a AND b`                              | 2-element sequence   |
| `__like`       | `LIKE` (case-sensitive)                        | string w/ `%`        |
| `__notlike`    | `NOT LIKE`                                     | string w/ `%`        |
| `__ilike`      | `ILIKE` (case-insensitive, Postgres)           | string w/ `%`        |
| `__notilike`   | `NOT ILIKE`                                    | string w/ `%`        |
| `__startswith` | `LIKE 'value%'` (auto-wrapped)                 | bare substring — do NOT add `%` yourself |
| `__endswith`   | `LIKE '%value'` (auto-wrapped)                 | bare substring                           |
| `__contains`   | `LIKE '%value%'` (auto-wrapped)                | bare substring                           |
| `__match`      | engine-specific full-text match                | depends              |
| `__or`         | OR group (see below)                           | dict                 |
| `__not`        | NOT wrapper                                    | scalar               |

### OR groups

To express `(col >= 10 OR col <= 0)`:

```python
await crud.get_multi(db, price__or={"gte": 100, "lte": 0})
```

The keys inside the dict are operator names (without the `__` prefix).

### Joined filters

For `crud_router` filter configs and the joined CRUD methods, use a **dot** to walk relationships:

```python
await crud.get_multi_joined(db, **{"tier.name__eq": "premium"})

FilterConfig({
    "tier.name__eq": None,          # query param: ?tier.name__eq=premium
    "tier.tier_score__gte": None,
})
```

Joined filters auto-include the join — you don't need to also list the relationship in `include_relationships`. They walk one level of relationship per `.`.

---

## `FilterConfig` — the `crud_router` filter surface

Defines which query parameters are exposed on the auto-generated `GET /list` endpoint, plus their default values and types.

```python
from fastcrud import FilterConfig, crud_router

filters = FilterConfig({
    "name__ilike": None,           # ?name__ilike=%admin%
    "price__gte": 0,               # default 0
    "active": True,                # default True; query param: ?active=false to override
    "tag__in": None,               # ?tag__in=a&tag__in=b
    "tier.name": None,             # joined filter; auto-joins tier
    "created_at__between": None,
})

router = crud_router(..., filter_config=filters)
```

### Types are exposed correctly in OpenAPI (v0.22+)

FastCRUD now reads the column type and renders proper `integer`/`string`/`boolean`/`array<X>` schemas instead of falling back to `any`. The FastAPI `/docs` page shows real input controls and OpenAPI codegen produces typed clients.

You don't need to do anything to opt in — it Just Works™ as long as you're on v0.22 or later.

### Bool coercion

`?active=false` (and `False`, `FALSE`, `0`) is correctly parsed as `False`. Case-insensitive. Don't pre-coerce.

---

## Dependency-based filters

Pass a FastAPI `Depends(...)` callable as the default value to filter by something extracted from the request (current user, tenant, header, etc.):

```python
from fastapi import Depends
from fastcrud import FilterConfig, crud_router

def current_tenant_id(user = Depends(get_current_user)) -> int:
    return user.tenant_id

router = crud_router(
    ...,
    filter_config=FilterConfig({
        "tenant_id": Depends(current_tenant_id),    # every list query is scoped to caller's tenant
        "name__ilike": None,
    }),
)
```

The dependency runs per request, before the query. This is the right way to enforce row-level access control.

---

## Custom operators

Register your own operator with `custom_filters` on `crud_router`:

```python
def year_eq(column):
    """Match rows where column's YEAR(...) equals the value."""
    from sqlalchemy import extract
    def make_filter(value):
        return extract("year", column) == value
    return make_filter

router = crud_router(
    ...,
    custom_filters={"year": year_eq},
    filter_config=FilterConfig({"created_at__year": None}),
)
```

Then `?created_at__year=2025` works.

For the full registration protocol see `fastcrud.core.filtering.operators.FilterCallable`. The callable signature is `(column) -> (value) -> SQLAlchemy expression`.

---

## When the filter syntax fails open

If a filter key doesn't match any column and isn't a registered operator, FastCRUD raises `ValueError`. There's no silent fall-through, so a typo in `"price__gtee"` will be caught — but only at query time, not at startup.

For startup-time validation, instantiate the `FilterConfig` early in your app and call `validate_joined_filter_path` (from `fastcrud.core.config`) against each joined key if you want to fail fast.
