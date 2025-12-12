# Advanced Filtering

The `_parse_filters` method in FastCRUD supports complex filtering operations including OR and NOT conditions.

## Basic Usage

Filters are specified as keyword arguments in the format `field_name__operator=value`:

```python
# Simple equality filter
results = await crud.get_multi(db, name="John")

# Comparison operators
results = await crud.get_multi(db, age__gt=18)
```

## OR Operations

### Single Field OR

Use the `__or` suffix to apply multiple conditions to the same field with OR logic:

```python
# Find users aged under 18 OR over 65
results = await crud.get_multi(
    db,
    age__or={
        "lt": 18,
        "gt": 65
    }
)
# Generates: WHERE age < 18 OR age > 65
```

You can also provide a list of values for the same operator to create multiple OR conditions:

```python
# Find users with names starting with Alice OR Frank OR Bob
results = await crud.get_multi(
    db,
    name__or={
        "like": ["Alice%", "Frank%", "Bob%"]
    }
)
# Generates: WHERE name LIKE 'Alice%' OR name LIKE 'Frank%' OR name LIKE 'Bob%'

# Mix list and single values in OR conditions
results = await crud.get_multi(
    db,
    name__or={
        "like": ["Alice%", "Frank%"],  # List of patterns
        "startswith": "Bob"             # Single value
    }
)
# Generates: WHERE name LIKE 'Alice%' OR name LIKE 'Frank%' OR name STARTSWITH 'Bob'
```

### Multi-Field OR

Use the special `_or` parameter to apply OR conditions across multiple different fields:

```python
# Find users with name containing 'john' OR email containing 'john'
results = await crud.get_multi(
    db,
    _or={
        "name__ilike": "%john%",
        "email__ilike": "%john%"
    }
)
# Generates: WHERE name ILIKE '%john%' OR email ILIKE '%john%'
```

This is particularly useful for implementing search functionality across multiple fields.

## NOT Operations

Use the `__not` suffix to negate multiple conditions on the same field:

```python
# Find users NOT aged 20 AND NOT between 30-40
results = await crud.get_multi(
    db,
    age__not={
        "eq": 20,
        "between": (30, 40)
    }
)
# Generates: WHERE NOT age = 20 AND NOT (age BETWEEN 30 AND 40)
```

Similar to OR operations, you can provide a list of values for the same operator in NOT conditions:

```python
# Find users whose names do NOT start with Alice, Frank, or Bob
results = await crud.get_multi(
    db,
    name__not={
        "like": ["Alice%", "Frank%", "Bob%"]
    }
)
# Generates: WHERE NOT (name LIKE 'Alice%') AND NOT (name LIKE 'Frank%') AND NOT (name LIKE 'Bob%')
```

## Supported Operators

- Comparison: `eq`, `gt`, `lt`, `gte`, `lte`, `ne`
- Null checks: `is`, `is_not`
- Text matching: `like`, `notlike`, `ilike`, `notilike`, `startswith`, `endswith`, `contains`, `match`
- Collections: `in`, `not_in`, `between`
- Logical: `or`, `not`

## Examples

```python
# Complex age filtering
results = await crud.get_multi(
    db,
    age__or={
        "between": (20, 30),
        "eq": 18
    },
    status__not={
        "in": ["inactive", "banned"]
    }
)

# Text search with OR conditions on a single field
results = await crud.get_multi(
    db,
    name__or={
        "startswith": "A",
        "endswith": "smith"
    }
)

# Search for multiple patterns with LIKE operator
results = await crud.get_multi(
    db,
    name__or={
        "like": ["Alice%", "Bob%", "Charlie%"]
    }
)

# Combine multiple operators including lists
results = await crud.get_multi(
    db,
    email__or={
        "ilike": ["%gmail.com", "%yahoo.com"],  # Multiple email domains
        "endswith": ".edu"                      # OR educational emails
    }
)

# Search across multiple fields with the same keyword
keyword = "john"
results = await crud.get_multi(
    db,
    _or={
        "name__ilike": f"%{keyword}%",
        "email__ilike": f"%{keyword}%",
        "phone__ilike": f"%{keyword}%",
        "address__ilike": f"%{keyword}%"
    }
)

# Combining multi-field OR with regular filters
results = await crud.get_multi(
    db,
    is_active=True,  # Regular filter applied to all results
    _or={
        "name__ilike": "%search term%",
        "description__ilike": "%search term%"
    }
)
```

## Error Handling

- Invalid column names raise `ValueError`
- Invalid operators are ignored
- Invalid value types for operators (e.g., non-list for `between`) raise `ValueError`

## Custom Filters

You can define custom filter operators at the instance level by passing `custom_filters` to `FastCRUD`, `EndpointCreator`, or `crud_router`. This allows you to extend or override the built-in filter operators.

### Defining Custom Filters

A custom filter is a callable that takes a SQLAlchemy column and returns a function that generates a filter condition:

```python
from sqlalchemy import Column, func
from fastcrud import FastCRUD, FilterCallable

# Custom filter: check if year matches
def year_filter(col: Column) -> FilterCallable:
    def filter_fn(value):
        return func.extract('year', col) == value
    return filter_fn

# Custom filter: case-insensitive equality
def case_insensitive_eq(col: Column) -> FilterCallable:
    def filter_fn(value):
        if isinstance(value, str):
            return func.lower(col) == func.lower(value)
        return col == value
    return filter_fn

# Custom filter: check if value is greater than doubled input
def double_gt(col: Column) -> FilterCallable:
    def filter_fn(value):
        return col > value * 2
    return filter_fn
```

### Using Custom Filters with FastCRUD

```python
from fastcrud import FastCRUD

# Create FastCRUD instance with custom filters
crud = FastCRUD(
    MyModel,
    custom_filters={
        "year": year_filter,
        "double_gt": double_gt,
    }
)

# Use custom filters in queries
results = await crud.get_multi(
    db,
    created_at__year=2024,        # Uses custom year filter
    price__double_gt=50,          # Uses custom double_gt filter (price > 100)
    status__eq="active",          # Uses built-in eq filter
)
```

### Using Custom Filters with crud_router

```python
from fastcrud import crud_router, FilterConfig

router = crud_router(
    session=get_session,
    model=MyModel,
    create_schema=CreateSchema,
    update_schema=UpdateSchema,
    custom_filters={
        "year": year_filter,
        "eq": case_insensitive_eq,  # Override built-in eq
    },
    filter_config=FilterConfig(
        created_at__year=None,
        name__eq=None,
    ),
    path="/items",
    tags=["Items"],
)
```

### Overriding Built-in Operators

Custom filters take precedence over built-in operators. This allows you to change default behavior:

```python
# Override the default 'eq' operator to be case-insensitive
crud = FastCRUD(
    UserModel,
    custom_filters={"eq": case_insensitive_eq}
)

# Now name__eq will be case-insensitive
results = await crud.get_multi(db, name__eq="alice")  # Matches "Alice", "ALICE", etc.
```

Note: Simple equality filters (`name=value`) use direct comparison and are not affected by custom 'eq' operators. Use explicit operator syntax (`name__eq=value`) to use custom operators.

### Instance-Level Isolation

Custom filters are scoped to the specific `FastCRUD` instance. Different instances can have different custom filters without affecting each other:

```python
# Instance with custom filters
crud_with_custom = FastCRUD(MyModel, custom_filters={"year": year_filter})

# Instance without custom filters
crud_standard = FastCRUD(MyModel)

# Custom filter works on first instance
results = await crud_with_custom.get_multi(db, created_at__year=2024)  # Works

# Raises error on second instance (unknown operator)
results = await crud_standard.get_multi(db, created_at__year=2024)  # ValueError
```

### FilterCallable Type

The `FilterCallable` type is exported from FastCRUD for type hints:

```python
from fastcrud import FilterCallable

def my_custom_filter(col: Column) -> FilterCallable:
    def filter_fn(value):
        return col == value
    return filter_fn
```