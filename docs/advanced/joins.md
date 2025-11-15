# Comprehensive Guide to Joins in FastCRUD

FastCRUD simplifies CRUD operations while offering capabilities for handling complex data relationships. This guide thoroughly explores the use of `JoinConfig` for executing join operations in FastCRUD methods such as `count`, `get_joined`, and `get_multi_joined`, alongside simplified join techniques for straightforward scenarios.

## Understanding `JoinConfig`

`JoinConfig` is a detailed configuration mechanism for specifying joins between models in FastCRUD queries. It contains the following key attributes:

- **`model`**: The SQLAlchemy model to join.
- **`join_on`**: The condition defining how the join connects to other models.
- **`join_prefix`**: An optional prefix for the joined columns to avoid column name conflicts.
- **`schema_to_select`**: An optional Pydantic schema for selecting specific columns from the joined model.
- **`join_type`**: The type of join (e.g., `"left"`, `"inner"`).
- **`alias`**: An optional SQLAlchemy `AliasedClass` for complex scenarios like self-referential joins or multiple joins on the same model.
- **`filters`**: An optional dictionary to apply filters directly to the joined model.
- **`relationship_type`**: Specifies the relationship type, such as `"one-to-one"` or `"one-to-many"`. Default is `"one-to-one"`.
- **`sort_columns`**: An optional column name or list of column names to sort the nested items by. Only applies to `"one-to-many"` relationships.
- **`sort_orders`**: An optional sort order (`"asc"` or `"desc"`) or list of sort orders corresponding to the columns in `sort_columns`. If not provided, defaults to `"asc"` for each column.

!!! TIP

    For `"many-to-many"`, you don't need to pass a `relationship_type`.

## Understanding `CountConfig`

`CountConfig` is a configuration mechanism for counting related objects in joined queries. This is particularly useful for many-to-many relationships where you want to include counts of related objects without actually joining the data. It contains the following key attributes:

- **`model`**: The SQLAlchemy model to count.
- **`join_on`**: The condition defining how the count query connects to the primary model.
- **`alias`**: An optional alias for the count column in the result. Defaults to `"{model.__tablename__}_count"`.
- **`filters`**: An optional dictionary to apply filters directly to the count query.

The count is implemented as a scalar subquery, which means all records from the primary model will be returned with their respective counts (including 0 for records with no related objects).

## Applying Joins in FastCRUD Methods

??? example "Models - `Tier`, `Department`, `User`, `Story`, `Task`"

    ??? example "`tier/model.py`"

        ```python
        --8<--
        fastcrud/examples/tier/model.py:imports
        fastcrud/examples/tier/model.py:model
        --8<--
        ```

    ??? example "`department/model.py`"

        ```python
        --8<--
        fastcrud/examples/department/model.py:imports
        fastcrud/examples/department/model.py:model
        --8<--
        ```

    ??? example "`user/model.py`"

        ```python
        --8<--
        fastcrud/examples/user/model.py:imports
        fastcrud/examples/user/model.py:model
        --8<--
        ```

    ??? example "`story/model.py`"

        ```python
        --8<--
        fastcrud/examples/story/model.py:imports
        fastcrud/examples/story/model.py:model
        --8<--
        ```

    ??? example "`task/model.py`"

        ```python
        --8<--
        fastcrud/examples/task/model.py:imports
        fastcrud/examples/task/model.py:model
        --8<--
        ```

### The `count` Method with Joins

The `count` method can be enhanced with join operations to perform complex aggregate queries. While `count` primarily returns the number of records matching a given condition, introducing joins allows for counting records across related models based on specific relationships and conditions.

#### Using `JoinConfig`

For join requirements, the `count` method can be invoked with join parameters passed as a list of `JoinConfig` to the `joins_config` parameter:

```python
from fastcrud import JoinConfig

task_crud = FastCRUD(Task)

# Count the number of tasks assigned to users in a specific department
task_count = await task_crud.count(
    db=db,
    joins_config=[
        JoinConfig(
            model=User,
            join_on=Task.assigned_user_id == User.id,
        ),
        JoinConfig(
            model=Department,
            join_on=User.department_id == Department.id,
            filters={"name": "Engineering"},
        ),
    ],
)
```

### Fetching Data with `get_joined` and `get_multi_joined`

These methods are essential for retrieving records from a primary model while including related data from one or more joined models. They support both simple and complex joining scenarios, including self-referential joins and many-to-many relationships.

#### Simple Joins Using Base Parameters

For simpler join requirements, FastCRUD allows specifying join parameters directly:

- **`join_model`**: The target model to join.
- **`join_on`**: The join condition.
- **`join_prefix`**: Optional prefix for columns from the joined model.
- **`join_schema_to_select`**: An optional Pydantic schema for selecting specific columns from the joined model.
- **`join_type`**: Specifies the SQL join type.
- **`alias`**: An optional SQLAlchemy `AliasedClass` for complex scenarios like self-referential joins or multiple joins on the same model.
- **`join_filters`**: Additional filters for the joined model.

#### Examples of Simple Joining

```python
# Fetch tasks with assigned user details, specifying a left join
tasks_with_users = await task_crud.get_joined(
    db=db,
    join_model=User,
    join_on=Task.assigned_user_id == User.id,
    join_type="left",
)
```

#### Getting Joined Data Nested

Note that by default, `FastCRUD` joins all the data and returns it in a single dictionary.

Let's take two of the tables from above and join them with `FastCRUD`:

```python
user_crud = FastCRUD(User)
user_tier = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_on=User.tier_id == Tier.id,
    join_prefix="tier_",
    join_type="left",
    id=1,
)
```

We'll get:

```json
{
    "id": 1,
    "name": "Example",
    "tier_id": 1,
    "tier_name": "Free"
}
```

If you want the joined data in a nested dictionary instead, you may just pass `nest_joins=True`:

```python
user_tier = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_on=User.tier_id == Tier.id,
    join_prefix="tier_",
    join_type="left",
    nest_joins=True,
    id=1,
)
```

And you will get:

```json
{
    "id": 1,
    "name": "Example",
    "tier": {
        "id": 1,
        "name": "Free"
    }
}
```

This works for both `get_joined` and `get_multi_joined`.

!!! WARNING

    Note that the final `"_"` in the passed `"tier_"` is stripped.

#### Returning Pydantic Models with `return_as_model`

By default, `get_joined` returns dictionaries containing the joined data. However, you can use the `return_as_model` parameter to get Pydantic model instances instead:

```python
# Using a schema that includes joined fields
class UserWithTier(BaseModel):
    id: int
    name: str
    tier_id: int
    tier_name: str

# Returns a dictionary (default behavior)
user_dict = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_prefix="tier_",
    schema_to_select=UserWithTier,
    return_as_model=False,  # Default
    id=1,
)
# Result: {"id": 1, "name": "Example", "tier_id": 1, "tier_name": "Free"}

# Returns a Pydantic model instance
user_model = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_prefix="tier_",
    schema_to_select=UserWithTier,
    return_as_model=True,
    id=1,
)
# Result: UserWithTier(id=1, name="Example", tier_id=1, tier_name="Free")
```

!!! NOTE "`return_as_model` Usage Notes"

    **Required Parameters**: When `return_as_model=True`, the `schema_to_select` parameter is required. FastCRUD will raise a `ValueError` if you try to use `return_as_model=True` without providing a schema.
    
    **Schema Design**: Ensure your schema includes all the fields that will be present in the flattened result, including joined fields with their prefixes.
    
    **Nested Joins Compatibility**: When using `return_as_model=True` with `nest_joins=True`, ensure that your `join_prefix` (minus trailing "_") matches the field name in your Pydantic schema. Otherwise, FastCRUD will raise a `ValueError` with clear guidance on how to fix the mismatch.
    
    **❌ This will raise an error:**
    ```python
    # Schema expects "children" field
    class ParentRead(BaseModel):
        children: list[ChildRead] = []
    
    # But join_prefix creates "child" key
    join_config = JoinConfig(
        join_prefix="child_",  # Creates "child" key
        relationship_type="one-to-many"
    )
    
    result = await crud.get_joined(
        return_as_model=True,  # Will raise ValueError
        nest_joins=True,
        joins_config=[join_config]
    )
    # Error: join_prefix 'child_' creates key 'child' which is not a field in schema ParentRead
    ```
    
    **✅ This works correctly:**
    ```python
    # Match the schema field name
    join_config = JoinConfig(
        join_prefix="children_",  # Creates "children" key to match schema
        relationship_type="one-to-many"
    )
    
    result = await crud.get_joined(
        return_as_model=True,
        nest_joins=True,
        joins_config=[join_config]
    )
    # Result: ParentRead(children=[...actual children...])
    ```

### Complex Joins Using `JoinConfig`

When dealing with more complex join conditions, such as multiple joins, self-referential joins, or needing to specify aliases and filters, `JoinConfig` instances become the norm. They offer granular control over each join's aspects, enabling precise and efficient data retrieval.

Example:

??? example "`user/schemas.py` Excerpt"

    ```python
    --8<--
    fastcrud/examples/user/schemas.py:readschema
    --8<--
    ```

```python
# Fetch users with details from related departments and tiers, using aliases for self-referential joins
from fastcrud import aliased

manager_alias = aliased(User)

users = await user_crud.get_multi_joined(
    db=db,
    schema_to_select=ReadUserSchema,
    joins_config=[
        JoinConfig(
            model=Department,
            join_on=User.department_id == Department.id,
            join_prefix="dept_",
        ),
        JoinConfig(
            model=Tier,
            join_on=User.tier_id == Tier.id,
            join_prefix="tier_",
        ),
        JoinConfig(
            model=User,
            alias=manager_alias,
            join_on=User.manager_id == manager_alias.id,
            join_prefix="manager_",
        ),
    ],
)
```

### Handling One-to-One and One-to-Many Joins in FastCRUD

FastCRUD provides flexibility in handling one-to-one and one-to-many relationships through `get_joined` and `get_multi_joined` methods, along with the ability to specify how joined data should be structured using both the `relationship_type` (default `"one-to-one"`) and the `nest_joins` (default `False`) parameters.

#### One-to-One Relationships

- **`get_joined`**: Fetch a single record and its directly associated record (e.g., a user and their profile).
- **`get_multi_joined`** (with `nest_joins=False`): Retrieve multiple records, each linked to a single related record from another table (e.g., users and their profiles).

##### Example

Let's take two of the tables from above and join them with `FastCRUD`:

```python
user_crud = FastCRUD(User)
user_tier = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_on=User.tier_id == Tier.id,
    join_prefix="tier_",
    join_type="left",
    id=1,
)
```

The result will be:

```json
{
    "id": 1,
    "name": "Example",
    "tier_id": 1,
    "tier_name": "Free"
}
```

###### One-to-One Relationship with Nested Joins

To get the joined data in a nested dictionary:

```python
user_tier = await user_crud.get_joined(
    db=db,
    join_model=Tier,
    join_on=User.tier_id == Tier.id,
    join_prefix="tier_",
    join_type="left",
    nest_joins=True,
    id=1,
)
```

The result will be:

```json
{
    "id": 1,
    "name": "Example",
    "tier": {
        "id": 1,
        "name": "Free"
    }
}
```

#### One-to-Many Relationships

- **`get_joined`** (with `nest_joins=True`): Retrieve a single record with all its related records nested within it (e.g., a user and all their blog posts).
- **`get_multi_joined`** (with `nest_joins=True`): Fetch multiple primary records, each with their related records nested (e.g., multiple users and all their blog posts).

!!! WARNING

    When using `nest_joins=True`, the performance will always be a bit worse than when using `nest_joins=False`. For cases where more performance is necessary, consider using `nest_joins=False` and remodeling your database.

##### Example

To demonstrate a one-to-many relationship, let's assume `Author` and `Article` tables:

```python
--8<--
fastcrud/examples/author/model.py:model
fastcrud/examples/article/model.py:model
--8<--
```

Fetch a user and all their posts:

```python
author_crud = FastCRUD(Author)
author_articles = await author_crud.get_joined(
    db=db,
    join_model=Article,
    join_on=Author.id == Article.author_id,
    join_prefix="article_",
    join_type="left",
    nest_joins=True,
    id=1,
)
```

The result will be:

```json
{
    "id": 1,
    "name": "Example Author",
    "articles": [
        {
            "id": 101,
            "author_id": 1,
            "title": "First Article!",
            "content": "First article content"
        },
        {
            "id": 102,
            "author_id": 1,
            "title": "Second Article?",
            "content": "Second article content"
        }
    ]
}
```

##### Sorting Nested Items in One-to-Many Relationships

FastCRUD allows you to sort nested items in one-to-many relationships using the `sort_columns` and `sort_orders` parameters in the `JoinConfig`. This is particularly useful when you want to display nested items in a specific order.

```python
from fastcrud import FastCRUD, JoinConfig

author_crud = FastCRUD(Author)

# Define join configuration with sorting
joins_config = [
    JoinConfig(
        model=Article,
        join_on=Author.id == Article.author_id,
        join_prefix="articles_",
        relationship_type="one-to-many",
        sort_columns="title",  # Sort articles by title
        sort_orders="asc"      # In ascending order
    )
]

# Fetch authors with their articles sorted by title
result = await author_crud.get_multi_joined(
    db=db,
    joins_config=joins_config,
    nest_joins=True
)
```

You can also sort by multiple columns with different sort orders:

```python
joins_config = [
    JoinConfig(
        model=Article,
        join_on=Author.id == Article.author_id,
        join_prefix="articles_",
        relationship_type="one-to-many",
        sort_columns=["published_date", "title"],  # Sort by date first, then title
        sort_orders=["desc", "asc"]               # Date descending, title ascending
    )
]
```

This will result in nested articles being sorted first by published_date in descending order, and then by title in ascending order within each date group.

#### Many-to-Many Relationships with `get_multi_joined`

FastCRUD simplifies dealing with many-to-many relationships by allowing easy fetch operations with joined models. Here, we demonstrate using `get_multi_joined` to handle a many-to-many relationship between `Project` and `Participant` models, linked through an association table.

**Note on Handling Many-to-Many Relationships:**

When using `get_multi_joined` for many-to-many relationships, it's essential to maintain a specific order in your `joins_config`:

1. **First**, specify the main table you're querying from.
2. **Next**, include the association table that links your main table to the other table involved in the many-to-many relationship.
3. **Finally**, specify the other table that is connected via the association table.

This order ensures that the SQL joins are structured correctly to reflect the many-to-many relationship and retrieve the desired data accurately.

!!! TIP

    Note that the first one can be the model defined in `FastCRUD(Model)`.

##### Scenario

Imagine a scenario where projects have multiple participants, and participants can be involved in multiple projects. This many-to-many relationship is facilitated through an association table.

##### Models

Our models include `Project`, `Participant`, and an association model `ProjectsParticipantsAssociation`:

???+ example "Models"

    ```python
    --8<--
    tests/sqlalchemy/conftest.py:model_project
    tests/sqlalchemy/conftest.py:model_participant
    tests/sqlalchemy/conftest.py:model_proj_parts_assoc
    --8<--
    ```

##### Fetching Data with `get_multi_joined`

To fetch projects along with their participants, we utilize `get_multi_joined` with appropriate `JoinConfig` settings:

```python
from fastcrud import FastCRUD, JoinConfig

# Initialize FastCRUD for the Project model
project_crud = FastCRUD(Project)

# Define join conditions and configuration
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

# Fetch projects with their participants as dictionaries
projects_with_participants = await project_crud.get_multi_joined(
    db_session,
    joins_config=joins_config,
)

# Or fetch as typed Pydantic models
from .schemas import ReadProjectSchema

projects_with_participants = await project_crud.get_multi_joined(
    db_session,
    schema_to_select=ReadProjectSchema,
    return_as_model=True,
    joins_config=joins_config,
)
```

**Return Types**:
- When `return_as_model=True` and `schema_to_select` is provided: `GetMultiResponseModel[SelectSchemaType]` (i.e., `Dict[str, Union[List[SelectSchemaType], int]]`)
- When `return_as_model=False`: `GetMultiResponseDict` (i.e., `Dict[str, Union[List[Dict[str, Any]], int]]`)

Now, `projects_with_participants['data']` will contain projects along with their participant information. The full results would look like:

```json
{
    "data": [
        {
            "id": 1,
            "name": "Project A",
            "description": "Description of Project A",
            "participants": [
                {
                    "id": 1,
                    "name": "Participant 1",
                    "role": "Developer"
                },
                {
                    "id": 2,
                    "name": "Participant 2",
                    "role": "Designer"
                }
            ]
        },
        {
            "id": 2,
            "name": "Project B",
            "description": "Description of Project B",
            "participants": [
                {
                    "id": 3,
                    "name": "Participant 3",
                    "role": "Manager"
                },
                {
                    "id": 4,
                    "name": "Participant 4",
                    "role": "Tester"
                }
            ]
        }
    ],
    "total_count": 2
}
```

### Counting Related Objects with `CountConfig`

FastCRUD provides `CountConfig` for efficiently counting related objects without fetching the actual data. This is particularly useful for many-to-many relationships or when you need to display counts alongside your main data.

#### Basic Usage

Use `CountConfig` with the `counts_config` parameter in `get_multi_joined`:

```python
from fastcrud import FastCRUD, CountConfig

# Count participants for each project
project_crud = FastCRUD(Project)

count_config = CountConfig(
    model=Participant,
    join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
           & (ProjectsParticipantsAssociation.project_id == Project.id),
    alias="participants_count",
)

result = await project_crud.get_multi_joined(
    db=session,
    counts_config=[count_config],
)
```

This will return data like:

```json
{
    "data": [
        {"id": 1, "name": "Project Alpha", "participants_count": 3},
        {"id": 2, "name": "Project Beta", "participants_count": 2},
        {"id": 3, "name": "Project Gamma", "participants_count": 0}
    ],
    "total_count": 3
}
```

#### Counting with Filters

Apply filters to count only specific related objects:

```python
# Count only developers for each project
count_config = CountConfig(
    model=Participant,
    join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
           & (ProjectsParticipantsAssociation.project_id == Project.id),
    alias="developers_count",
    filters={"role": "Developer"},
)

result = await project_crud.get_multi_joined(
    db=session,
    counts_config=[count_config],
)
```

#### Multiple Count Configurations

You can use multiple `CountConfig` instances to get different counts:

```python
# Count all participants and developers separately
all_participants_count = CountConfig(
    model=Participant,
    join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
           & (ProjectsParticipantsAssociation.project_id == Project.id),
    alias="all_participants_count",
)

developers_count = CountConfig(
    model=Participant,
    join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
           & (ProjectsParticipantsAssociation.project_id == Project.id),
    alias="developers_count",
    filters={"role": "Developer"},
)

result = await project_crud.get_multi_joined(
    db=session,
    counts_config=[all_participants_count, developers_count],
)
```

#### One-to-Many Relationships

`CountConfig` works with one-to-many relationships as well:

```python
# Count articles for each author
author_crud = FastCRUD(Author)

count_config = CountConfig(
    model=Article,
    join_on=Article.author_id == Author.id,
    alias="articles_count",
)

result = await author_crud.get_multi_joined(
    db=session,
    counts_config=[count_config],
)
```

#### Combining with Regular Joins

You can use `CountConfig` alongside `JoinConfig` for comprehensive data retrieval:

```python
# Get project details with participant information AND counts
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

counts_config = [
    CountConfig(
        model=Participant,
        join_on=(Participant.id == ProjectsParticipantsAssociation.participant_id)
               & (ProjectsParticipantsAssociation.project_id == Project.id),
        alias="total_participants",
    )
]

result = await project_crud.get_multi_joined(
    db=session,
    joins_config=joins_config,
    counts_config=counts_config,
    nest_joins=True,
)
```

#### Practical Tips for Advanced Joins

- **Prefixing**: Always use the `join_prefix` attribute to avoid column name collisions, especially in complex joins involving multiple models or self-referential joins.
- **Aliasing**: Utilize the `alias` attribute for disambiguating joins on the same model or for self-referential joins.
- **Filtering Joined Models**: Apply filters directly to joined models using the `filters` attribute in `JoinConfig` to refine the data set returned by the query.
- **Ordering Joins**: In many-to-many relationships or complex join scenarios, carefully sequence your `JoinConfig` entries to ensure logical and efficient SQL join construction.

## Conclusion

FastCRUD's support for join operations enhances the ability to perform complex queries across related models in FastAPI applications. By understanding and utilizing the `JoinConfig` and `CountConfig` classes within the `count`, `get_joined`, and `get_multi_joined` methods, developers can craft powerful data retrieval queries that efficiently handle both data fetching and counting operations.
