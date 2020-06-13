import pandas as pd


def check_if_dataframe(obj):
    """Checks if given object is a pandas DataFrame."""
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"Provided object ({type(obj).__name__}) is not a pandas.DataFrame"
        )


def check_columns_exist(df, columns):
    """Checks if given columns are in the dataframe."""
    missing_columns = set(columns).difference(df.columns)
    if missing_columns:
        raise KeyError(f"{list(missing_columns)} column(s) are not in given DataFrame")
