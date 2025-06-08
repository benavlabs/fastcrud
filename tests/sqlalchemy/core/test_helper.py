from fastcrud.endpoint import helper


def test_get_python_type_returns_none_when_not_implemented(
    fake_sqlalchemy_utils_column_no_python_type,
) -> None:
    result = helper._get_python_type(fake_sqlalchemy_utils_column_no_python_type)
    assert result is None


def test_get_python_type_returns_correct_type_when_implemented(
    fake_sqlalchemy_utils_column_with_python_type,
) -> None:
    result = helper._get_python_type(fake_sqlalchemy_utils_column_with_python_type)
    assert result is not None
    assert result == fake_sqlalchemy_utils_column_with_python_type.type.python_type
