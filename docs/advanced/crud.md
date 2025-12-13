# Advanced Use of FastCRUD

FastCRUD offers a flexible and powerful approach to handling CRUD operations in FastAPI applications, leveraging the SQLAlchemy ORM. Beyond basic CRUD functionality, FastCRUD provides advanced features like `allow_multiple` for updates and deletes, and support for advanced filters (e.g., less than, greater than). These features enable more complex and fine-grained data manipulation and querying capabilities.

## Typing Options for `FastCRUD`

Note that when initializing `FastCRUD`, assuming you have a model like:

???+ example "Simplified `user/model.py`"

    ```python
    --8<--
    fastcrud/examples/user/model.py:imports
    fastcrud/examples/user/model.py:model_common
    --8<--
    ```

!!! WARNING

    Note that naive `datetime` such as `datetime.utcnow` is not supported by `FastCRUD` as it was [deprecated](https://github.com/python/cpython/pull/103858).
    
    Use timezone aware `datetime`, such as `datetime.now(UTC)` instead.

You could just pass it to `FastCRUD`:

```python
from fastcrud import FastCRUD

user_crud = FastCRUD(User)
```

But you also may want a more robust typing, for that purpose, you may also pass the relevant pydantic schemas in the following way:

??? example "Simplified `user/schemas.py`"

    ```python
    --8<--
    fastcrud/examples/user/schemas.py:imports
    fastcrud/examples/user/schemas.py:createschema_common


    fastcrud/examples/user/schemas.py:readschema_common


    fastcrud/examples/user/schemas.py:updateschema_common


    fastcrud/examples/user/schemas.py:deleteschema
    --8<--
    ```

```python
from .user.model import User
from .user.schemas import CreateUserSchema, ReadUserSchema, UpdateUserSchema, DeleteUserSchema

# Just pass None if you don't have one of the schemas
UserCRUD = FastCRUD[User, CreateUserSchema, UpdateUserSchema, None, DeleteUserSchema]
```

Then you can initialize `UserCRUD` like you would any `FastCRUD` instance, but with the relevant types:

```python
from .user.model import User

user_crud = UserCRUD(User)
```

## Allow Multiple Updates and Deletes

One of FastCRUD's advanced features is the ability to update or delete multiple records at once based on specified conditions. This is particularly useful for batch operations where you need to modify or remove several records that match certain criteria.

### Updating Multiple Records

To update multiple records, you can set the `allow_multiple=True` parameter in the `update` method. This allows FastCRUD to apply the update to all records matching the given filters.

```python
# Assuming setup for FastCRUD instance `item_crud` and SQLAlchemy async session `db`

# Update all items priced below $10 to a new price
await item_crud.update(
    db=db,
    object={"price": 9.99},
    allow_multiple=True,
    price__lt=10,
)
```

### Deleting Multiple Records

Similarly, you can delete multiple records by using the `allow_multiple=True` parameter in the `delete` or `db_delete` method, depending on whether you're performing a soft or hard delete.

```python
# Soft delete all items not sold in the last year
await item_crud.delete(
    db=db,
    allow_multiple=True,
    last_sold__lt=datetime.datetime.now() - datetime.timedelta(days=365),
)
```

## Advanced Filters

FastCRUD supports advanced filtering options, allowing you to query records using operators such as greater than (`__gt`), less than (`__lt`), and their inclusive counterparts (`__gte`, `__lte`). These filters can be used in any method that retrieves or operates on records, including `get`, `get_multi`, `exists`, `count`, `update`, and `delete`.

### Single parameter filters

Most filter operators require a single string or integer value.

```python
# Fetch items priced between above $5
items = await item_crud.get_multi(
    db=db,
    price__gte=5,
)
```

Currently supported single parameter filters are:

- `__gt` - greater than
- `__lt` - less than
- `__gte` - greater than or equal to
- `__lte` - less than or equal to
- `__ne` - not equal
- `__is` - used to test True, False and None identity
- `__is_not` - negation of "is"
- `__like` - SQL "like" search for specific text pattern
- `__notlike` - negation of "like"
- `__ilike` - case insensitive "like"
- `__notilike` - case insensitive "notlike"
- `__startswith` - text starts with given string
- `__endswith` - text ends with given string
- `__contains` - text contains given string
- `__match` - database-specific match expression

### Complex parameter filters

Some operators require multiple values. They must be passed as a python tuple, list or set.

```python
# Fetch items priced between $5 and $20
items = await item_crud.get_multi(
    db=db,
    price__between=(5, 20),
)
```

- `__between` - between 2 numeric values
- `__in` - included in
- `__not_in` - not included in

### OR clauses

More complex OR filters are supported. They must be passed as dictionary, where each key is a library-supported operator to be used in OR expression and values is what get's passed as the parameter.

```python
# Fetch items priced under $5 or above $20
items = await item_crud.get_multi(
    db=db,
    price__or={'lt': 5, 'gt': 20},
)
```

### AND clauses

AND clauses can be achieved by chaining multiple filters together.

```python
# Fetch items priced under $20 and over 2 years of warranty.
items = await item_crud.get_multi(
    db=db,
    price__lt=20,
    warranty_years__gt=2,
)
```

### Counting Records

```python
# Count items created in the last month
item_count = await item_crud.count(
    db=db,
    created_at__gte=datetime.datetime.now() - datetime.timedelta(days=30),
)
```

## Skipping Database Commit

For `create`, `update`, `db_delete` and `delete` methods of `FastCRUD`, you have the option of passing `commit=False` so you don't commit the operations immediately.

```python
from fastcrud import FastCRUD

from .database import session as db
from .item.model import Item

item_crud = FastCRUD(Item)

await item_crud.delete(
    db=db, 
    commit=False, 
    id=1,
)
# this will not actually delete until you run a db.commit()
```

## Returning clause in `update`

In `update` method, you can pass `return_columns` parameter containing a list of columns you want to return after the update.

```python
from fastcrud import FastCRUD

from .database import session as db
from .item.model import Item

item_crud = FastCRUD(Item)

item = await item_crud.update(
    db=db,
    object={"price": 9.99},
    return_columns=["price"],
    price__lt=10,
)
# this will return the updated price
```

You can also pass `schema_to_select` parameter and `return_as_model` to return the updated data in the form of a Pydantic schema.

```python
from fastcrud import FastCRUD

from .database import session as db
from .item.model import Base, Item
from .item.schemas import ItemSchema

item_crud = FastCRUD(Item)

item = await item_crud.update(
    db=db,
    object={"price": 9.99},
    schema_to_select=ItemSchema,
    return_as_model=True,
    price__lt=10,
)
# this will return the updated data in the form of ItemSchema
```

## Unpaginated `get_multi` and `get_multi_joined`

If you pass `None` to `limit` in `get_multi` and `get_multi_joined`, you get the whole unpaginated set of data that matches the filters. Use this with caution.

```python
from fastcrud import FastCRUD

from .database import session as db
from .item.model import Item

item_crud = FastCRUD(Item)

items = await item_crud.get_multi(db=db, limit=None)
# this will return all items in the db
```

!!! CAUTION

    Be cautious when returning all the data in your database, and you should almost never allow your user API to do this.

## Using `get_joined` and `get_multi_joined` for multiple models

To facilitate complex data relationships, `get_joined` and `get_multi_joined` can be configured to handle joins with multiple models. This is achieved using the `joins_config` parameter, where you can specify a list of `JoinConfig` instances, each representing a distinct join configuration.

Additionally, `get_multi_joined` supports counting related objects using the `counts_config` parameter with `CountConfig` instances. This allows you to efficiently get counts of related objects (like counting participants per project) without fetching the actual related data.

For detailed examples and usage patterns, see the [joins documentation](joins.md).

## Upserting multiple records using `upsert_multi`

FastCRUD provides an `upsert_multi` method to efficiently upsert multiple records in a single operation. This method is particularly useful when you need to insert new records or update existing ones based on a unique constraint.

**Return Types**:
- When `return_as_model=True` and `schema_to_select` is provided: `UpsertMultiResponseModel[SelectSchemaType]` - a TypedDict with `data: list[SelectSchemaType]`
- When `return_as_model=False`: `UpsertMultiResponseDict` - a TypedDict with `data: list[dict[str, Any]]`

**Usage Examples**:

Upsert multiple records and return as dictionaries:
```python
from fastcrud import FastCRUD

from .database import session as db
from .item.model import Item
from .item.schemas import CreateItemSchema

item_crud = FastCRUD(Item)
items = await item_crud.upsert_multi(
    db=db,
    instances=[
        CreateItemSchema(price=9.99),
    ],
)
# Returns: {"upserted": [Dict[str, Any], ...]}
```

Upsert multiple records and return as typed Pydantic models:
```python
from fastcrud import FastCRUD

from .database import session as db
from .item.model import Item
from .item.schemas import CreateItemSchema, ItemSchema

item_crud = FastCRUD(Item)
items = await item_crud.upsert_multi(
    db=db,
    instances=[
        CreateItemSchema(price=9.99),
    ],
    schema_to_select=ItemSchema,
    return_as_model=True,
)
# Returns: {"upserted": [ItemSchema, ...]}
```

### Customizing the Update Logic

To allow more granular control over the SQL `UPDATE` operation during an upsert, `upsert_multi` can accept an `update_override` argument. This allows for the specification of custom update logic using SQL expressions, like the `case` statement, to handle complex conditions.

```python
from sqlalchemy.sql import case

update_override = {
    "name": case(
        (Item.name.is_(None), stmt.excluded.name),
        else_=Item.name
    )
}

items = await item_crud.upsert_multi(
    db=db,
    instances=[
        CreateItemSchema(name="Gadget", price=15.99),
    ],
    update_override=update_override,
    schema_to_select=ItemSchema,
    return_as_model=True,
)
```

In the example above, the `name` field of the `Item` model will be updated to the new value only if the existing `name` field is `None`. Otherwise, it retains the existing `name`.

#### Example: Joining `User`, `Tier`, and `Department` Models

Consider a scenario where you want to retrieve users along with their associated tier and department information. Here's how you can achieve this using `get_multi_joined`.

Start by creating the models and schemas, followed by a description of how they're to be joined:

??? example "Models and Schemas"

    ??? example "`tier/model.py`"

        ```python
        --8<--
        fastcrud/examples/tier/model.py:imports
        fastcrud/examples/tier/model.py:model
        --8<--
        ```

    ??? example "`tier/schemas.py`"

        ```python
        --8<--
        fastcrud/examples/tier/schemas.py:imports
        fastcrud/examples/tier/schemas.py:readschema
        --8<--
        ```

    ??? example "`department/model.py`"

        ```python
        --8<--
        fastcrud/examples/department/model.py:imports
        fastcrud/examples/department/model.py:model
        --8<--
        ```

    ??? example "`department/schemas.py`"

        ```python
        --8<--
        fastcrud/examples/department/schemas.py:imports
        fastcrud/examples/department/schemas.py:readschema
        --8<--
        ```

    ??? example "`user/model.py`"

        ```python
        --8<--
        fastcrud/examples/user/model.py:imports
        fastcrud/examples/user/model.py:model
        --8<--
        ```

    ??? example "`user/schemas.py`"

        ```python
        --8<--
        fastcrud/examples/user/schemas.py:imports
        fastcrud/examples/user/schemas.py:createschema
        fastcrud/examples/user/schemas.py:readschema
        fastcrud/examples/user/schemas.py:updateschema
        fastcrud/examples/user/schemas.py:deleteschema
        --8<--
        ```

    ??? example "`story/model.py`"

        ```python
        --8<--
        fastcrud/examples/story/model.py:imports
        fastcrud/examples/story/model.py:model
        --8<--
        ```

    ??? example "`story/schemas.py`"

        ```python
        --8<--
        fastcrud/examples/story/schemas.py:imports
        fastcrud/examples/story/schemas.py:createschema
        fastcrud/examples/story/schemas.py:readschema
        fastcrud/examples/story/schemas.py:updateschema
        fastcrud/examples/story/schemas.py:deleteschema
        --8<--
        ```

    ??? example "`task/model.py`"

        ```python
        --8<--
        fastcrud/examples/task/model.py:imports
        fastcrud/examples/task/model.py:model
        --8<--
        ```

    ??? example "`task/schemas.py`"

        ```python
        --8<--
        fastcrud/examples/task/schemas.py:imports
        fastcrud/examples/task/schemas.py:createschema
        fastcrud/examples/task/schemas.py:readschema
        fastcrud/examples/task/schemas.py:updateschema
        fastcrud/examples/task/schemas.py:deleteschema
        --8<--
        ```

```python hl_lines="1 3-10 12-19" title="Join Configurations"
from fastcrud import JoinConfig

joins_config = [
    JoinConfig(
        model=Tier,
        join_on=User.tier_id == Tier.id,
        join_prefix="tier_",
        schema_to_select=ReadTierSchema,
        join_type="left",
    ),

    JoinConfig(
        model=Department,
        join_on=User.department_id == Department.id,
        join_prefix="dept_",
        schema_to_select=ReadDepartmentSchema,
        join_type="inner",
    ),
]

users = await user_crud.get_multi_joined(
    db=session,
    schema_to_select=ReadUserSchema,
    return_as_model=True,
    offset=0,
    limit=10,
    sort_columns='username',
    sort_orders='asc',
    joins_config=joins_config,
)
```

**Return Types for `get_multi_joined`**:
- When `return_as_model=True` and `schema_to_select` is provided: `GetMultiResponseModel[SelectSchemaType]` - a TypedDict with `data: list[SelectSchemaType]` and `total_count: int` (when `return_total_count=True`)
- When `return_as_model=False`: `GetMultiResponseDict` - a TypedDict with `data: list[dict[str, Any]]` and `total_count: int` (when `return_total_count=True`)

Then just pass this list to `joins_config`:

```python hl_lines="14" title="Passing to get_multi_joined"
from fastcrud import JoinConfig

joins_config = [
    ...
]

users = await user_crud.get_multi_joined(
    db=session,
    schema_to_select=ReadUserSchema,
    offset=0,
    limit=10,
    sort_columns='username',
    sort_orders='asc',
    joins_config=joins_config,
)
```

In this example, users are joined with the `Tier` and `Department` models. The `join_on` parameter specifies the condition for the join, `join_prefix` assigns a prefix to columns from the joined models (to avoid naming conflicts), and `join_type` determines whether it's a left or inner join.

!!! WARNING

    If both single join parameters and `joins_config` are used simultaneously, an error will be raised.

### Handling One-to-One and One-to-Many Joins in FastCRUD

FastCRUD provides flexibility in handling one-to-one and one-to-many relationships through its `get_joined` and `get_multi_joined` methods, along with the ability to specify how joined data should be structured using both the `relationship_type` (default `one-to-one`) and the `nest_joins` (default `False`) parameters.

#### One-to-One Joins

**One-to-one** relationships can be efficiently managed using either `get_joined` or `get_multi_joined`. The `get_joined` method is typically used when you want to fetch a single record from the database along with its associated record from another table, such as a user and their corresponding profile details. If you're retrieving multiple records, `get_multi_joined` can also be used for one-to-one joins. The parameter that deals with it, `relationship_type`, defaults to `one-on-one`.

#### One-to-Many Joins

For **one-to-many** relationships, where a single record can be associated with multiple records in another table, `get_joined` can be used with `nest_joins` set to `True`. This setup allows the primary record to include a nested list of associated records, making it suitable for scenarios such as retrieving a user and all their blog posts. Alternatively, `get_multi_joined` is also applicable here for fetching multiple primary records, each with their nested lists of related records.

!!! WARNING

    When using `nested_joins=True`, the performance will always be a bit worse than when using `nested_joins=False`. For cases where more performance is necessary, consider using `nested_joins=False` and remodeling your database.

#### One-to-One Relationships

- **`get_joined`**: Fetch a single record and its directly associated record (e.g., a user and their profile).
- **`get_multi_joined`** (with `nest_joins=False`): Retrieve multiple records, each linked to a single related record from another table (e.g., users and their profiles).

#### One-to-Many Relationships

- **`get_joined`** (with `nest_joins=True`): Retrieve a single record with all its related records nested within it (e.g., a user and all their blog posts).
- **`get_multi_joined`** (with `nest_joins=True`): Fetch multiple primary records, each with their related records nested (e.g., multiple users and all their blog posts).

For a more detailed explanation, you may check the [joins docs](joins.md#handling-one-to-one-and-one-to-many-joins-in-fastcrud).

### Using aliases

In complex query scenarios, particularly when you need to join a table to itself or perform multiple joins on the same table for different purposes, aliasing becomes crucial. Aliasing allows you to refer to the same table in different contexts with unique identifiers, avoiding conflicts and ambiguity in your queries.

For both `get_joined` and `get_multi_joined` methods, when you need to join the same model multiple times, you can utilize the `alias` parameter within your `JoinConfig` to differentiate between the joins. This parameter expects an instance of `AliasedClass`, which can be created using the `aliased` function from SQLAlchemy (also in FastCRUD for convenience).

#### Example: Joining the Same Model Multiple Times

Consider a task management application where tasks have both an owner and an assigned user, represented by the same `User` model. To fetch tasks with details of both users, we use aliases to join the `User` model twice, distinguishing between owners and assigned users.

Let's start by creating the aliases and passing them to the join configuration. Don't forget to use the alias for `join_on`:

```python hl_lines="4-5 11 15 19 23" title="Join Configurations with Aliases"
from fastcrud import FastCRUD, JoinConfig, aliased

# Create aliases for User to distinguish between the owner and the assigned user
owner_alias = aliased(User, name="owner")
assigned_user_alias = aliased(User, name="assigned_user")

# Configure joins with aliases
joins_config = [
    JoinConfig(
        model=User,
        join_on=Task.owner_id == owner_alias.id,
        join_prefix="owner_",
        schema_to_select=ReadUserSchema,
        join_type="inner",
        alias=owner_alias,  # Pass the aliased class instance
    ),
    JoinConfig(
        model=User,
        join_on=Task.assigned_user_id == assigned_user_alias.id,
        join_prefix="assigned_",
        schema_to_select=ReadUserSchema,
        join_type="inner",
        alias=assigned_user_alias,  # Pass the aliased class instance
    ),
]

# Initialize your FastCRUD instance for Task
task_crud = FastCRUD(Task)

# Fetch tasks with joined user details
tasks = await task_crud.get_multi_joined(
    db=session,
    schema_to_select=ReadTaskSchema,
    offset=0,
    limit=10,
    joins_config=joins_config,
)
```

Then just pass this `joins_config` to `get_multi_joined`:

```python hl_lines="19" title="Passing joins_config to get_multi_joined"
from fastcrud import FastCRUD, JoinConfig, aliased

...

# Configure joins with aliases
joins_config = [
    ...
]

# Initialize your FastCRUD instance for Task
task_crud = FastCRUD(Task)

# Fetch tasks with joined user details
tasks = await task_crud.get_multi_joined(
    db=session,
    schema_to_select=ReadTaskSchema,
    offset=0,
    limit=10,
    joins_config=joins_config,
)
```

In this example, `owner_alias` and `assigned_user_alias` are created from `User` to distinguish between the task's owner and the assigned user within the task management system. By using aliases, you can join the same model multiple times for different purposes in your queries, enhancing expressiveness and eliminating ambiguity.

### Many-to-Many Relationships with `get_multi_joined`

FastCRUD simplifies dealing with many-to-many relationships by allowing easy fetch operations with joined models. Here, we demonstrate using `get_multi_joined` to handle a many-to-many relationship between `Project` and `Participant` models, linked through an association table.

**Note on Handling Many-to-Many Relationships:**

When using `get_multi_joined` for many-to-many relationships, it's essential to maintain a specific order in your `joins_config`: 

1. **First**, specify the main table you're querying from.
2. **Next**, include the association table that links your main table to the other table involved in the many-to-many relationship.
3. **Finally**, specify the other table that is connected via the association table.

This order ensures that the SQL joins are structured correctly to reflect the many-to-many relationship and retrieve the desired data accurately.

!!! TIP

    Note that the first one can be the model defined in `FastCRUD(Model)`.

??? example "Models"

    ```python
    --8<--
    tests/sqlalchemy/conftest.py:model_project
    tests/sqlalchemy/conftest.py:model_participant
    tests/sqlalchemy/conftest.py:model_proj_parts_assoc
    --8<--
    ```

```python
# Fetch projects with their participants via a many-to-many relationship
joins_config = [
    JoinConfig(
        model=ProjectsParticipantsAssociation,
        join_on=Project.id == ProjectsParticipantsAssociation.project_id,
        join_prefix="pp_",
        join_type="inner",
    ),
    JoinConfig(
        model=Participant,
        join_on=ProjectsParticipantsAssociation.participant_id == Participant.id,
        join_prefix="participant_",
        join_type="inner",
    ),
]

project_crud = FastCRUD(Project)

class ReadProjectSchema(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

projects_with_participants = await project_crud.get_multi_joined(
    db=db,
    schema_to_select=ReadProjectSchema,
    joins_config=joins_config,
)
```

For a more detailed explanation, read [this part of the docs](joins.md#many-to-many-relationships-with-get_multi_joined).

## Enhanced Query Capabilities with Method Chaining

The `select` method in FastCRUD is designed for flexibility, enabling you to build complex queries through method chaining.

### The `select` Method

```python
async def select(
    db: AsyncSession,
    schema_to_select: Optional[type[BaseModel]] = None,
    sort_columns: Optional[Union[str, list[str]]] = None,
    sort_orders: Optional[Union[str, list[str]]] = None,
    **kwargs: Any
) -> Selectable
```

This method constructs a SQL Alchemy `Select` statement, offering optional column selection, filtering, and sorting. It's designed for flexibility, allowing you to chain additional SQLAlchemy methods for even more complex queries.

#### Features:

- **Column Selection**: Specify columns with a Pydantic schema.
- **Sorting**: Define one or more columns for sorting, along with their sort order.
- **Filtering**: Apply filters directly through keyword arguments.
- **Chaining**: Chain with other SQLAlchemy methods for advanced query construction.

#### Usage Example:

```python
--8<--
fastcrud/examples/mymodel/schemas.py:readschema
--8<--
stmt = await my_model_crud.select(
    schema_to_select=ReadMyModelSchema,
    sort_columns='name',
    name__like='%example%',
)
stmt = stmt.where(additional_conditions).limit(10)
results = await db.execute(stmt)
```

This example demonstrates selecting a subset of columns, applying a filter, and chaining additional conditions like `where` and `limit`. Note that `select` returns a `Select` object, allowing for further modifications before execution.

## Conclusion

The advanced features of FastCRUD, such as `allow_multiple` and support for advanced filters, empower developers to efficiently manage database records with complex conditions. By leveraging these capabilities, you can build more dynamic, robust, and scalable FastAPI applications that effectively interact with your data model.
