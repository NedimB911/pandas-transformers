import pandas as pd
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from pandas_transformers.transformers import PandasOneHotEncoder, PandasTfidfVectorizer


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
        categ = pd.CategoricalDtype(["a", "b", "c"])
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


class TestPandasTfidfVectorizer:
    """
    Tests for the PandasTfidfVectorizer class

    """

    @pytest.fixture
    def example_train_df(self):
        """Example training dataset."""
        return pd.DataFrame({"text": ["house", "animal", "house"], "num": [3, 4, 4]})

    @pytest.fixture
    def example_train_df_binary(self):
        """Dataset with binary labels"""
        return pd.DataFrame({"text": ["house", "animal"] * 10, "y": [0, 1] * 10})

    @pytest.fixture
    def example_missing_values_df(self):
        """
        Example dataset with missing value
        """
        return pd.DataFrame({"text": ["house", "animal", None]})

    @pytest.fixture
    def example_test_df_diff_column(self):
        """Example testing dataset which contains a column that is not present in the
        other example datasets"""
        return pd.DataFrame({"new_col": [3]})

    @pytest.fixture
    def example_series(self):
        """Example dataste in pd.Series format"""
        return pd.Series(["house", "animal", "house"])

    @pytest.fixture
    def example_np_array(self):
        """Example dataset in np.ndarray format"""
        return pd.Series(["house", "animal", "house"]).values

    def test_example(self, example_train_df):
        """ Tests a simple example. """
        transformer = PandasTfidfVectorizer(column="text")
        transformer.fit(example_train_df)
        transformed = transformer.transform(example_train_df)

        expected = pd.DataFrame(
            {
                "num": pd.Series([3, 4, 4]),
                "animal": pd.Series([0.0, 1.0, 0.0]),
                "house": pd.Series([1.0, 0.0, 1.0]),
            }
        )
        # The column order shouldnt matter (therefore we sort them)
        pd.testing.assert_frame_equal(
            transformed.sort_index(axis=1), expected.sort_index(axis=1)
        )

    def test_missing_column(self, example_train_df, example_test_df_diff_column):
        """
        Test transformer when test set does not have the required columns.
        In that case, it should return a KeyError
        """
        transformer = PandasTfidfVectorizer(column="text")
        transformer.fit(example_train_df)

        with pytest.raises(KeyError):
            transformer.transform(example_test_df_diff_column)

    def test_missing_values_fit(self, example_missing_values_df):
        """
        Tests the case where there are missing values in the training data.
        Should return a ValueError.
        """

        transformer = PandasTfidfVectorizer(column="text")
        with pytest.raises(ValueError):
            transformer.fit(example_missing_values_df)

    def test_missing_values_transform(
        self, example_train_df, example_missing_values_df
    ):
        """
        Tests the case where there are missing values in the testing data.
        Should return a ValueError.
        """

        transformer = PandasTfidfVectorizer(column="text")
        transformer.fit(example_train_df)

        with pytest.raises(ValueError):
            transformer.transform(example_missing_values_df)

    def test_series_input(self, example_series):
        """
        In case we don't give a value for the column keyword argument, the input
        should be a pandas series or np.ndarray.
        Otherwise, return a TypeError.
        """

        transformer = PandasTfidfVectorizer()
        transformer.fit(example_series)
        transformed = transformer.transform(example_series)

        expected = pd.DataFrame(
            {"animal": pd.Series([0.0, 1.0, 0.0]), "house": pd.Series([1.0, 0.0, 1.0]),}
        )

        pd.testing.assert_frame_equal(
            transformed.sort_index(axis=1), expected.sort_index(axis=1)
        )

    def test_fit_with_series_input_with_column_arg(self, example_series):
        """
        In case we do  give a value for the column keyword argument, the input
        should be a pd.DataFrame.
        Otherwise, return a TypeError.
        """

        transformer = PandasTfidfVectorizer(column="text")
        with pytest.raises(TypeError):
            transformer.fit(example_series)

    def test_fit_with_df_input_without_column_arg(self, example_train_df):
        """
        In case we give no column argument to the initalizer, the input during fit
        should be a pd.Series. Otherwise raise TypeError.

        """
        transformer = PandasTfidfVectorizer()
        with pytest.raises(TypeError):
            transformer.fit(example_train_df)

    def test_clone(self):
        """
        Test clone

        """
        transformer = PandasTfidfVectorizer(column="test", max_features=123)
        cloned = clone(transformer)

        assert transformer.column == cloned.column
        assert transformer.max_features == cloned.max_features

    def test_grid_search(self, example_train_df_binary):
        """Tests for grid search compatibility."""

        pipe = Pipeline(
            [("tfidf", PandasTfidfVectorizer()), ("model", LogisticRegression())]
        )
        param_grid = {
            "tfidf__max_features": [5, 15],
        }

        X = example_train_df_binary["text"]
        y = example_train_df_binary["y"]

        search = GridSearchCV(pipe, param_grid)
        search.fit(X, y)
