import pytest
from fastcrud import FastCRUD


@pytest.mark.asyncio
async def test_parse_filters_single_condition(test_model):
    fast_crud = FastCRUD(test_model)

    filters = fast_crud._filter_processor.parse_filters(name="John Doe")
    assert len(filters) == 1
    assert str(filters[0]) == "test.name = :name_1"


@pytest.mark.asyncio
async def test_parse_filters_multiple_conditions(test_model):
    fast_crud = FastCRUD(test_model)

    filters = fast_crud._filter_processor.parse_filters(tier_id__gt=1, is_deleted=True)
    assert len(filters) == 2
    assert str(filters[0]).endswith("tier_id > :tier_id_1")
    assert str(filters[1]) == "test.is_deleted = true"


@pytest.mark.asyncio
async def test_parse_filters_or_condition(test_model):
    fast_crud = FastCRUD(test_model)

    filters = fast_crud._filter_processor.parse_filters(name__or={"gt": 1, "lt": 5})
    assert len(filters) == 1
    assert str(filters[0]) == "test.name > :name_1 OR test.name < :name_2"


@pytest.mark.asyncio
async def test_parse_filters_contained_in(test_model):
    fast_crud = FastCRUD(test_model)
    filters = fast_crud._filter_processor.parse_filters(category_id__in=[1, 2])
    assert len(filters) == 1
    assert str(filters[0]) == "test.category_id IN (__[POSTCOMPILE_category_id_1])"


@pytest.mark.asyncio
async def test_parse_filters_not_contained_in(test_model):
    fast_crud = FastCRUD(test_model)
    filters = fast_crud._filter_processor.parse_filters(category_id__not_in=[1, 2])
    assert len(filters) == 1
    assert (
        str(filters[0]) == "(test.category_id NOT IN (__[POSTCOMPILE_category_id_1]))"
    )


@pytest.mark.asyncio
async def test_parse_filters_between_condition(test_model):
    fast_crud = FastCRUD(test_model)
    filters = fast_crud._filter_processor.parse_filters(category_id__between=[1, 5])
    assert len(filters) == 1
    assert (
        str(filters[0]) == "test.category_id BETWEEN :category_id_1 AND :category_id_2"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operator", ("in", "not_in", "between"))
async def test_parse_filters_raises_exception(test_model, operator: str):
    fast_crud: FastCRUD = FastCRUD(test_model)
    with pytest.raises(ValueError) as exc:
        if operator == "in":
            fast_crud._filter_processor.parse_filters(category_id__in=1)
        elif operator == "not_in":
            fast_crud._filter_processor.parse_filters(category_id__not_in=1)
        elif operator == "between":
            fast_crud._filter_processor.parse_filters(category_id__between=1)
    assert str(exc.value) == f"<{operator}> filter must be tuple, list or set"


@pytest.mark.asyncio
async def test_parse_filters_invalid_column(test_model):
    fast_crud = FastCRUD(test_model)

    with pytest.raises(ValueError):
        fast_crud._filter_processor.parse_filters(
            invalid_column__="This does not exist"
        )


@pytest.mark.asyncio
async def test_parse_filters_with_custom_column_names(test_model_custom_columns):
    fast_crud = FastCRUD(test_model_custom_columns)

    filters = fast_crud._filter_processor.parse_filters(meta={"key": "value"})
    assert len(filters) == 1
    assert "test_custom.metadata =" in str(filters[0])

    filters = fast_crud._filter_processor.parse_filters(name__like="John%")
    assert len(filters) == 1
    assert "test_custom.display_name LIKE" in str(filters[0])

    filters = fast_crud._filter_processor.parse_filters(
        meta__contains={"key": "value"}, name="John"
    )
    assert len(filters) == 2
    filter_str = [str(f) for f in filters]
    print(filter_str)
    assert any("test_custom.metadata LIKE" in f for f in filter_str)
    assert any("test_custom.display_name =" in f for f in filter_str)
