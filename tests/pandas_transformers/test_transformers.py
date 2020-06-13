import pandas as pd
import numpy as np
import pytest

from pandas_transformers.transformers import PandasOneHotEncoder


class TestPandasOneHotEncoder:
    """
    Tests for the PandasOneHotEncoder class

    """

    @pytest.fixture
    def example_train_df(self):
        """Example training dataset."""
        return pd.DataFrame({"cat": ["a", "a", "b"], "num": [3, 4, 4]})

    @pytest.fixture
    def example_test_df(self):
        """Example testing dataset which contains a category not present in the
        training dataset (c)."""
        return pd.DataFrame({"cat": ["a", "b", "c"], "num": [3, 4, 3]})

    @pytest.fixture
    def example_test_df_diff_column(self):
        """Example testing dataset which contains a column that is not present in the
        other example datasets"""
        return pd.DataFrame({"new_col": [3]})

    @pytest.fixture
    def example_missing_values_df(self):
        """
        Example dataset with missing value
        """

        return pd.DataFrame({"cat": ["a", "a", None]})

    def test_example(self, example_train_df):
        """ Tests a simple example. """
        transformer = PandasOneHotEncoder()
        transformer.fit(example_train_df)
        transformed = transformer.transform(example_train_df)

        expected = pd.DataFrame(
            {
                "cat_a": pd.Series([1, 1, 0], dtype=np.uint8),
                "cat_b": pd.Series([0, 0, 1], dtype=np.uint8),
                "num": [3, 4, 4],
            }
        )
        # The column order shouldnt matter (therefore we sort them)
        pd.testing.assert_frame_equal(
            transformed.sort_index(axis=1), expected.sort_index(axis=1)
        )

    def test_min_frequency(self, example_train_df):
        """Example where we use min_frequency=2. This means that any category with less
        than two occurences should be ignored.
        """
        transformer = PandasOneHotEncoder(min_frequency=2)
        transformer.fit(example_train_df)
        transformed = transformer.transform(example_train_df)

        expected = pd.DataFrame(
            {"cat_a": pd.Series([1, 1, 0], dtype=np.uint8), "num": [3, 4, 4]}
        )

        # The column order shouldnt matter (therefore we sort them)
        pd.testing.assert_frame_equal(
            transformed.sort_index(axis=1), expected.sort_index(axis=1)
        )

    def test_unseen_category(self, example_train_df, example_test_df):
        """
        Example where we test the onehot encoder in the case where it encounters
        new categories during transform (in the test set) and we choose to ignore it
        """
        transformer = PandasOneHotEncoder()
        transformer.fit(example_train_df)
        transformed_test = transformer.transform(example_test_df)

        expected = pd.DataFrame(
            {
                "cat_a": pd.Series([1, 0, 0], dtype=np.uint8),
                "cat_b": pd.Series([0, 1, 0], dtype=np.uint8),
                "num": [3, 4, 3],
            }
        )

        # The column order shouldnt matter (therefore we sort them)
        pd.testing.assert_frame_equal(
            transformed_test.sort_index(axis=1), expected.sort_index(axis=1)
        )

    def test_unseen_category_error(self, example_train_df, example_test_df):
        """
        Example where we test the onehot encoder in the case where it encounters
        new categories during transform (in the test set) and we choose to raise
        an error
        """
        transformer = PandasOneHotEncoder(handle_unknown="error")
        transformer.fit(example_train_df)
        with pytest.raises(ValueError):
            transformer.transform(example_test_df)

    def test_missing_column(self, example_train_df, example_test_df_diff_column):
        """
        Test transformer when test set does not have the required columns.
        In that case, it should return a KeyError
        """
        transformer = PandasOneHotEncoder()
        transformer.fit(example_train_df)

        with pytest.raises(KeyError):
            transformer.transform(example_test_df_diff_column)

    def test_zero_count_categories(self, example_train_df):
        """
        Test transformer when categorical columns have categories that do not
        occur at all (e.g. in predefined categories)

        """
        categ = pd.Categorical(["a", "b", "c"])
        example_train_df_extra_cat = example_train_df.assign(
            cat=lambda _df: _df["cat"].astype(categ)
        )

        transformer = PandasOneHotEncoder()
        transformer.fit(example_train_df_extra_cat)
        transformed = transformer.transform(example_train_df_extra_cat)

        expected = pd.DataFrame(
            {
                "cat_a": pd.Series([1, 1, 0], dtype=np.uint8),
                "cat_b": pd.Series([0, 0, 1], dtype=np.uint8),
                "cat_c": pd.Series([0, 0, 0], dtype=np.uint8),
                "num": [3, 4, 4],
            }
        )

        # The column order shouldnt matter (therefore we sort them)
        pd.testing.assert_frame_equal(
            transformed.sort_index(axis=1), expected.sort_index(axis=1)
        )

    def test_missing_values(self, example_missing_values_df):
        """
        Tests the case where there are missing values in the data.
        """

        transformer = PandasOneHotEncoder()
        transformer.fit(example_missing_values_df)
        transformed = transformer.transform(example_missing_values_df)

        expected = pd.DataFrame({"cat_a": pd.Series([1, 1, 0], dtype=np.uint8)})

        # The column order shouldnt matter (therefore we sort them)
        pd.testing.assert_frame_equal(
            transformed.sort_index(axis=1), expected.sort_index(axis=1)
        )
