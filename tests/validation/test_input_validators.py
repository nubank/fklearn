import pandas as pd
import pytest

from fklearn.exceptions import (
    EmptyDataFrameError,
    InvalidParameterRangeError,
    InvalidParameterValueError,
    MissingColumnsError,
)
from fklearn.validation import (
    validate_columns_exist,
    validate_nonempty_df,
    validate_positive_int,
    validate_range,
    validate_value_in_set,
)


class TestValidateColumnsExist:
    def test_valid_columns_no_error(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        # Should not raise
        validate_columns_exist(df, ['a', 'b'])

    def test_missing_columns_raises_error(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with pytest.raises(MissingColumnsError) as exc_info:
            validate_columns_exist(df, ['a', 'c', 'd'])

        assert exc_info.value.missing_columns == ['c', 'd']
        assert set(exc_info.value.available_columns) == {'a', 'b'}

    def test_context_in_error_message(self):
        df = pd.DataFrame({'a': [1, 2]})
        with pytest.raises(MissingColumnsError) as exc_info:
            validate_columns_exist(df, ['b'], context='my_function')

        assert '[my_function]' in str(exc_info.value)

    def test_empty_columns_list_no_error(self):
        df = pd.DataFrame({'a': [1, 2]})
        # Should not raise for empty column list
        validate_columns_exist(df, [])

    def test_curried_usage(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        validator = validate_columns_exist(columns=['a', 'b'])
        # Should not raise
        validator(df)


class TestValidateNonemptyDf:
    def test_nonempty_df_no_error(self):
        df = pd.DataFrame({'a': [1, 2]})
        # Should not raise
        validate_nonempty_df(df)

    def test_empty_df_raises_error(self):
        df = pd.DataFrame()
        with pytest.raises(EmptyDataFrameError):
            validate_nonempty_df(df)

    def test_empty_df_with_columns_raises_error(self):
        df = pd.DataFrame({'a': [], 'b': []})
        with pytest.raises(EmptyDataFrameError):
            validate_nonempty_df(df)

    def test_context_in_error_message(self):
        df = pd.DataFrame()
        with pytest.raises(EmptyDataFrameError) as exc_info:
            validate_nonempty_df(df, context='imputer')

        assert '[imputer]' in str(exc_info.value)

    def test_curried_usage(self):
        validator = validate_nonempty_df(context='test')
        df = pd.DataFrame({'a': [1]})
        # Should not raise
        validator(df)


class TestValidateRange:
    def test_value_in_range_no_error(self):
        validate_range(5, 'param', min_value=1, max_value=10)

    def test_value_at_min_boundary_no_error(self):
        validate_range(1, 'param', min_value=1, max_value=10)

    def test_value_at_max_boundary_no_error(self):
        validate_range(10, 'param', min_value=1, max_value=10)

    def test_value_below_min_raises_error(self):
        with pytest.raises(InvalidParameterRangeError) as exc_info:
            validate_range(0, 'n_estimators', min_value=1, max_value=100)

        assert exc_info.value.param_name == 'n_estimators'
        assert exc_info.value.value == 0
        assert exc_info.value.min_value == 1

    def test_value_above_max_raises_error(self):
        with pytest.raises(InvalidParameterRangeError) as exc_info:
            validate_range(101, 'n_estimators', min_value=1, max_value=100)

        assert exc_info.value.param_name == 'n_estimators'
        assert exc_info.value.value == 101
        assert exc_info.value.max_value == 100

    def test_only_min_bound(self):
        validate_range(100, 'param', min_value=1)  # No error
        with pytest.raises(InvalidParameterRangeError):
            validate_range(0, 'param', min_value=1)

    def test_only_max_bound(self):
        validate_range(-100, 'param', max_value=10)  # No error
        with pytest.raises(InvalidParameterRangeError):
            validate_range(11, 'param', max_value=10)

    def test_curried_usage(self):
        validator = validate_range(param_name='learning_rate', min_value=0.0, max_value=1.0)
        validator(0.5)  # Should not raise
        with pytest.raises(InvalidParameterRangeError):
            validator(1.5)


class TestValidateValueInSet:
    def test_valid_value_no_error(self):
        validate_value_in_set('mean', 'strategy', ['mean', 'median', 'mode'])

    def test_invalid_value_raises_error(self):
        with pytest.raises(InvalidParameterValueError) as exc_info:
            validate_value_in_set('average', 'strategy', ['mean', 'median', 'mode'])

        assert exc_info.value.param_name == 'strategy'
        assert exc_info.value.value == 'average'
        assert 'mean' in exc_info.value.allowed_values

    def test_works_with_set(self):
        validate_value_in_set('a', 'param', {'a', 'b', 'c'})  # No error
        with pytest.raises(InvalidParameterValueError):
            validate_value_in_set('d', 'param', {'a', 'b', 'c'})

    def test_curried_usage(self):
        validator = validate_value_in_set(param_name='impute_strategy', allowed_values=['mean', 'median'])
        validator('mean')  # Should not raise
        with pytest.raises(InvalidParameterValueError):
            validator('mode')


class TestValidatePositiveInt:
    def test_positive_int_no_error(self):
        validate_positive_int(1, 'n_folds')
        validate_positive_int(100, 'n_folds')

    def test_zero_raises_error(self):
        with pytest.raises(InvalidParameterRangeError) as exc_info:
            validate_positive_int(0, 'n_folds')

        assert exc_info.value.param_name == 'n_folds'

    def test_negative_raises_error(self):
        with pytest.raises(InvalidParameterRangeError):
            validate_positive_int(-1, 'n_folds')

    def test_curried_usage(self):
        validator = validate_positive_int(param_name='batch_size')
        validator(32)  # Should not raise
        with pytest.raises(InvalidParameterRangeError):
            validator(0)
