import pytest
from pydantic import BaseModel
from fastcrud.paginated.schemas import (
    create_list_response,
    create_paginated_response,
    PaginatedRequestQuery,
)


class SampleSchema(BaseModel):
    name: str
    value: int


def test_create_list_response_default_key():
    ResponseModel = create_list_response(SampleSchema)

    # Test model creation
    assert ResponseModel.__name__ == "DynamicListResponse"

    # Test with valid data
    data = [{"name": "test", "value": 1}, {"name": "test2", "value": 2}]
    response = ResponseModel(data=data)
    assert len(response.data) == 2
    assert response.data[0].name == "test"
    assert response.data[1].value == 2

    # Test with empty list
    empty_response = ResponseModel(data=[])
    assert len(empty_response.data) == 0


def test_create_list_response_custom_key():
    ResponseModel = create_list_response(SampleSchema, response_key="items")

    # Test model creation
    assert ResponseModel.__name__ == "DynamicListResponse"

    # Test with valid data
    data = [{"name": "test", "value": 1}]
    response = ResponseModel(items=data)
    assert len(response.items) == 1
    assert response.items[0].name == "test"
    assert response.items[0].value == 1


def test_create_list_response_validation():
    ResponseModel = create_list_response(SampleSchema)

    # Test invalid data
    with pytest.raises(ValueError):
        ResponseModel(data=[{"invalid_field": "test"}])


def test_create_paginated_response_default_key():
    ResponseModel = create_paginated_response(SampleSchema)

    # Test model creation
    assert ResponseModel.__name__ == "DynamicPaginatedResponse"

    # Test with valid data
    response = ResponseModel(
        data=[{"name": "test", "value": 1}],
        total_count=1,
        has_more=False,
        page=1,
        items_per_page=10,
    )

    assert len(response.data) == 1
    assert response.data[0].name == "test"
    assert response.total_count == 1
    assert response.has_more is False
    assert response.page == 1
    assert response.items_per_page == 10


def test_create_paginated_response_custom_key():
    ResponseModel = create_paginated_response(SampleSchema, response_key="items")

    # Test with valid data
    response = ResponseModel(
        items=[{"name": "test", "value": 1}],
        total_count=1,
        has_more=False,
        page=1,
        items_per_page=10,
    )

    assert len(response.items) == 1
    assert response.items[0].name == "test"


def test_create_paginated_response_optional_fields():
    ResponseModel = create_paginated_response(SampleSchema)

    # Test with minimal required fields
    response = ResponseModel(
        data=[{"name": "test", "value": 1}], total_count=1, has_more=False
    )

    assert response.page is None
    assert response.items_per_page is None


def test_create_paginated_response_validation():
    ResponseModel = create_paginated_response(SampleSchema)

    # Test missing required fields
    with pytest.raises(ValueError):
        ResponseModel(
            data=[{"name": "test", "value": 1}],
            has_more=False,  # missing total_count
        )

    # Test invalid data structure
    with pytest.raises(ValueError):
        ResponseModel(data=[{"invalid_field": "test"}], total_count=1, has_more=False)


def test_create_paginated_response_empty_list():
    ResponseModel = create_paginated_response(SampleSchema)

    response = ResponseModel(
        data=[], total_count=0, has_more=False, page=1, items_per_page=10
    )

    assert len(response.data) == 0
    assert response.total_count == 0
    assert response.has_more is False


# Tests for PaginatedRequestQuery


def test_paginated_request_query_default_values():
    """Test PaginatedRequestQuery with default values."""
    query = PaginatedRequestQuery()

    assert query.offset is None
    assert query.limit is None
    assert query.page is None
    assert query.items_per_page is None
    assert query.sort is None


def test_paginated_request_query_with_pagination():
    """Test PaginatedRequestQuery with page-based pagination."""
    query = PaginatedRequestQuery(page=2, items_per_page=20)

    assert query.page == 2
    assert query.items_per_page == 20
    assert query.offset is None
    assert query.limit is None
    assert query.sort is None


def test_paginated_request_query_with_offset_limit():
    """Test PaginatedRequestQuery with offset/limit pagination."""
    query = PaginatedRequestQuery(offset=10, limit=50)

    assert query.offset == 10
    assert query.limit == 50
    assert query.page is None
    assert query.items_per_page is None
    assert query.sort is None


def test_paginated_request_query_with_sort():
    """Test PaginatedRequestQuery with sorting."""
    query = PaginatedRequestQuery(sort="name,-age")

    assert query.sort == "name,-age"
    assert query.page is None
    assert query.items_per_page is None
    assert query.offset is None
    assert query.limit is None


def test_paginated_request_query_with_all_params():
    """Test PaginatedRequestQuery with all parameters."""
    query = PaginatedRequestQuery(
        offset=0, limit=100, page=1, items_per_page=10, sort="id"
    )

    assert query.offset == 0
    assert query.limit == 100
    assert query.page == 1
    assert query.items_per_page == 10
    assert query.sort == "id"


def test_paginated_request_query_alias_items_per_page():
    """Test PaginatedRequestQuery with itemsPerPage alias."""
    # Using the alias in dict format (as it would come from query params)
    query = PaginatedRequestQuery.model_validate({"itemsPerPage": 25, "page": 1})

    assert query.items_per_page == 25
    assert query.page == 1


def test_paginated_request_query_populate_by_name():
    """Test that populate_by_name allows both field name and alias."""
    # Test with field name
    query1 = PaginatedRequestQuery(items_per_page=15)
    assert query1.items_per_page == 15

    # Test with alias
    query2 = PaginatedRequestQuery.model_validate({"itemsPerPage": 20})
    assert query2.items_per_page == 20

    # Test that both work together (alias takes precedence in Pydantic v2)
    query3 = PaginatedRequestQuery.model_validate(
        {"items_per_page": 15, "itemsPerPage": 20}
    )
    assert query3.items_per_page == 20
