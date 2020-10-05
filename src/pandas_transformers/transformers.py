# pylint: disable=missing-function-docstring
# pylint: disable=arguments-differ
# pylint: disable=unused-argument

from typing import List, Iterable

from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np

from .utils import check_columns_exist, check_if_dataframe


def test():
    print("hello world")


class PandasOneHotEncoder(TransformerMixin):
    """
    This one-hot encoder preserves the dataframe structure.

    It works the same as pd.get_dummies(), however pd.get_dummies() cannot deal with
    unseen categories during transformation (e.g. when a category appears in the
    test set but not in the train set)

    https://github.com/pandas-dev/pandas/issues/8918
    https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html

    Parameters
    ----------
    min_frequency :
        Any category of a categorical column that appears less than 'min_frequency',
        will be ignored (no extra dummy column for that category)
    columns :
        The list of columns to one-hot-encode. The default is None, in which
        case all columns of type 'object' or 'category' will be one-hot-encoded.
    dummy_na :
        Add a column to indicate NaNs, if False NaNs are ignored (default)
    drop_first:
        Whether to get k-1 dummies out of k categorical levels by removing
        the first level.
    """

    def __init__(
        self,
        min_frequency: int = -1,
        columns: List[str] = None,
        dummy_na: bool = False,
        drop_first: bool = False,
        handle_unknown: str = "ignore",
    ):
        super().__init__()
        self.min_frequency = min_frequency
        self.dummy_na = dummy_na
        self.drop_first = drop_first

        if handle_unknown not in {"ignore", "error"}:
            raise ValueError(
                "handle_unknown must be either 'ignore' or 'error'."
                f" Got {handle_unknown}."
            )

        self.handle_unknown = handle_unknown

        if (columns is not None) & (not isinstance(columns, list)):
            raise ValueError(
                f"'columns' must be a list (of strings). Got {type(columns)}"
            )
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PandasOneHotEncoder":
        dtypes_to_encode = ["object", "category"]

        check_if_dataframe(X)
        if self.columns is None:
            self.columns = X.select_dtypes(include=dtypes_to_encode).columns.tolist()
        else:
            check_columns_exist(X, self.columns)

        self.categories_ = {}
        self.categories_unfiltered_ = {}

        for col in self.columns:
            counts = X[col].value_counts(dropna=False)
            self.categories_[col] = list(
                set(counts[counts >= self.min_frequency].index.tolist())
            )
            self.categories_unfiltered_[col] = set(counts.index.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)

        check_if_dataframe(X)
        check_columns_exist(X, self.columns)

        if self.handle_unknown == "error":
            self._check_unknown_categories_all_columns(X)

        # pylint: disable=cell-var-from-loop
        for col in self.columns:
            cat = pd.CategoricalDtype(self.categories_[col], ordered=True)
            X = X.assign(**{f"{col}": lambda _df: _df[col].astype(cat)})

        return pd.get_dummies(
            X, columns=self.columns, dummy_na=self.dummy_na, drop_first=self.drop_first
        )

    def _find_unseen_categories(self, X, col):
        """
        We check whether X has any categories that were not seen during training
        for a single column.

        NOTE: We also consider categories that we have ignored during training
        due to 'min_frequency'.

        """
        seen_categories = set(self.categories_unfiltered_[col])
        new_categories = set(X[col].value_counts(dropna=False).index)
        unseen_categories = new_categories - seen_categories
        return unseen_categories

    def _check_unknown_categories_all_columns(self, X):
        """
        We check whether X has any categories that were not seen during training
        for *ALL* columns.

        NOTE: We also consider categories that we have ignored during training
        due to 'min_frequency'.

        """

        unseen_categories_dict = {}
        for col in self.columns:
            unseen_categories = self._find_unseen_categories(X, col)

            if unseen_categories:
                unseen_categories_dict[col] = unseen_categories

        if unseen_categories_dict:
            raise ValueError(
                f"Encountered categories not seen during fitting:"
                f"{unseen_categories_dict}"
            )


class PandasTfidfVectorizer(TfidfVectorizer):
    """
    PandasTfidfVectorizer

    A pandas version for sklearn's tf-idf vectorizer. The PandasTfidfVectorizer
    converts the sparse array returned by sklearn's tf-idf vectorizer into a dense
    array and uses the feature names to convert it into a dataframe.


    # https://www.datacamp.com/community/tutorials
    # /super-multiple-inheritance-diamond-problem

    # https://stackoverflow.com/questions/3277367
    # /how-does-pythons-super-work-with-multiple-inheritance

    Parameters
    ----------
    column: str (optional)
        Column which we wish to apply the tf-idf vectorizer to. If no column is given,
        then the input dataframe should be a pd.Series or a np.ndarray.
    **kwargs
        Keyword arguments for sklearn.feature_extraction.text.TfidfVectorizer

    """

    def __init__(self, column: str = None, **kwargs):

        super().__init__(**kwargs)
        self.column = column

    # pylint: disable = arguments-differ
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the tf-idf vectorizer.

        Parameters
        ----------
        X : pd.DataFrame or 1-d Iterable[str]
        y : optional

        """
        if self.column is not None:
            # In this case the input must be a dataframe
            self._init_input_validation()
            check_if_dataframe(X)
            check_columns_exist(X, [self.column])
            raw_documents = X[self.column]
        else:
            # if no column is given, the input must be a 1-d iterable
            # (pd.Series or np array)
            self._check_if_1d_series_or_np_array(X)
            raw_documents = X

        self._check_missing(raw_documents)

        return super().fit(raw_documents, y)

    # pylint: disable = arguments-differ
    def transform(self, X: pd.DataFrame):
        """
        Transforms the input dataframe using the fitted tf-idf vectorizer.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        if self.column is not None:
            # In this case the input must be a dataframe
            check_if_dataframe(X)
            check_columns_exist(X, [self.column])
            raw_documents = X[self.column]
        else:
            # if no column is given, the input must be a 1-d iterable
            # (pd.Series or np array)
            self._check_if_1d_series_or_np_array(X)
            raw_documents = X

        self._check_missing(raw_documents)
        transformed = super().transform(raw_documents)

        transformed_df = pd.DataFrame(
            transformed.toarray(), columns=self.get_feature_names(), index=X.index
        )

        if self.column is not None:
            transformed_df = pd.concat(
                [X.drop(columns=[self.column]), transformed_df], axis=1
            )

        return transformed_df

    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        fit_transform


        Parameters
        ----------
        X : pd.DataFrame
        y : optional

        Returns
        -------
        pd.DataFrame
        """

        self.fit(X, y)
        return self.transform(X)

    def _init_input_validation(self):
        """
        Validates the __init__() inputs.
        """

        if not isinstance(self.column, str):
            raise TypeError(
                f"'column' argument should be a string. Got {type(self.column)}"
            )

    def _check_missing(self, raw_documents: Iterable[str]):
        """
        Checks whether the raw_documents have any missing values. If so, return
        a ValueError.

        Parameters
        ----------
        raw_documents : Iterable[str]

        Raises
        ------
        ValueError

        """

        if raw_documents.isnull().any():
            raise ValueError(
                f"The {self.column} column contains None's. TfidfVectorizer requires"
                " the column to not have any None's."
            )

    def _check_if_1d_series_or_np_array(self, obj):
        if isinstance(obj, (pd.Series, np.ndarray)):
            if obj.ndim == 1:
                return True
            raise TypeError(
                "Input is of the correct type (pd.Series or np.ndarray)"
                " However, not of the correct dimension. It should be 1d."
            )

        raise TypeError(
            f"If no column is given, the input should be either a pd.Series or a "
            f"np.ndarray. Got {type(obj)} instead."
        )

    def _get_param_names(self):
        tfidf_param_names = TfidfVectorizer._get_param_names()
        current_param_names = super()._get_param_names()
        param_names = tfidf_param_names + current_param_names
        return param_names
