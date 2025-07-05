# Dependency-Based Filtering

FastCRUD now supports dependency-based filtering, allowing you to automatically filter query results based on values from dependencies. This is particularly useful for implementing row-level access control, where users should only see data that belongs to their organization or tenant.

## Basic Usage

You can use dependency-based filtering by passing a callable (function or dependency) as a filter value in the `FilterConfig`:

```python
from fastapi import Depends, FastAPI
from fastcrud import crud_router, FilterConfig

from .database import async_session
from .models import ExternalProviderConfig
from .schemas import ExternalProviderConfigSchema

# Define a dependency that returns the user's organization ID
async def get_auth_user():
    # Your authentication logic here
    return UserInfo(organization_id=123)

async def get_org_id(auth: UserInfo = Depends(get_auth_user)):
    return auth.organization_id

# Create a router with dependency-based filtering
app = FastAPI()
epc_router = crud_router(
    session=async_session,
    model=ExternalProviderConfig,
    create_schema=ExternalProviderConfigSchema,
    update_schema=ExternalProviderConfigSchema,
    path="/external_provider_configs",
    filter_config=FilterConfig(
        organization_id=get_org_id,  # This will be resolved at runtime
    ),
    tags=["external_provider_configs"],
)

app.include_router(epc_router)
```

In this example, the `get_org_id` dependency will be called for each request, and the returned value will be used to filter the results by `organization_id`.

## How It Works

When you provide a callable as a filter value, FastCRUD will:

1. Use FastAPI's `Depends` to inject the dependency into the endpoint
2. Call the dependency function at runtime to get the actual filter value
3. Apply the filter to the query

This means that the filter value can be dynamically determined based on the current request context, such as the authenticated user.

## Combining Static and Dynamic Filters

You can combine dependency-based filters with static filters:

```python
filter_config = FilterConfig(
    organization_id=get_org_id,  # Dynamic filter from dependency
    status="active",             # Static filter
    is_deleted=False,            # Static filter
)
```

## Advanced Usage with Nested Dependencies

You can use nested dependencies to build more complex filtering logic:

```python
async def get_auth_user():
    # Your authentication logic here
    return UserInfo(organization_id=123)

async def get_user_permissions(auth: UserInfo = Depends(get_auth_user)):
    # Get user permissions
    return UserPermissions(can_see_all=False)

async def get_org_filter(
    auth: UserInfo = Depends(get_auth_user),
    permissions: UserPermissions = Depends(get_user_permissions)
):
    # If user has special permissions, don't filter by organization
    if permissions.can_see_all:
        return None
    return auth.organization_id

filter_config = FilterConfig(
    organization_id=get_org_filter,  # This might return None for some users
)
```

In this example, the `organization_id` filter will only be applied if the user doesn't have the `can_see_all` permission.

## Limitations

- Dependency-based filters are only applied to the `read_multi` endpoint
- The dependency function must return a value that is compatible with the column type
- Complex filtering logic should be implemented in the dependency function, not in the filter configuration
