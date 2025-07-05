from fastapi import Depends

from fastcrud.endpoint.helper import FilterConfig


class UserInfo:
    def __init__(self, organization_id: int):
        self.organization_id = organization_id


async def get_auth_user():
    return UserInfo(organization_id=123)


async def get_org_id(auth: UserInfo = Depends(get_auth_user)):
    return auth.organization_id


def test_filter_config_with_callable_value():
    filter_config = FilterConfig(
        organization_id=get_org_id,
        name=None,
    )

    # Check that the callable is stored correctly
    assert callable(filter_config.filters["organization_id"])
    assert filter_config.filters["organization_id"] == get_org_id

    # Check that non-callable values are still handled correctly
    assert filter_config.filters["name"] is None


def test_filter_config_get_params_with_callable():
    filter_config = FilterConfig(
        organization_id=get_org_id,
        name="test",
    )

    params = filter_config.get_params()

    # Check that callable values are wrapped with Depends
    assert hasattr(params["organization_id"], "dependency")
    assert params["organization_id"].dependency == get_org_id

    # Check that non-callable values are still handled correctly
    assert params["name"].default == "test"
