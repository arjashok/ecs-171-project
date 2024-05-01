"""
    @brief Utility for the initial exploration of a dataset.
    @author Arjun Ashok
"""


# Environment Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Superficial Utility
def standardize_column_names(df: pd.DataFrame, inplace: bool=False) -> pd.DataFrame | None:
    """
        Ensures column names are in a standardized format (lower case, no space, 
        `_` for separation of words).

        @param df: dataset
        @param inplace: whether to modify the dataset or not
    """

    # auxiliary fn
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    def convert_column_name(col_name: str, pattern) -> str:
        """
            Standardizes the column name.
        """

        # transform
        col_name = col_name.strip()
        col_name = col_name.replace(" ", "_")
        col_name = pattern.sub("_", col_name)

        return col_name.lower()

    # clean columns
    if not inplace:
        df = df.copy()
    df.columns = [convert_column_name(col, pattern) for col in df.columns]

    if not inplace:
        return df


def present_feature_name(feat_name: str) -> str:
    """
        Converts from the standard format to a feature 
    """

    # TODO

def normalize_features(ds, features, how):
    """
        Standardizes all the numeric features.
        @param ds - The dataset that we are normalizing the features for
        @param features - features that need to be standardized
        @param how - method used to standardize the features
    """
    if how == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    ds.data[features] = scaler.fit_transform(ds.data[features])


# Feature-Engineering
def up_sampling(df: pd.DataFrame, target: str, inplace: bool=False) -> None | pd.DataFrame:
    """
        Upsamples to balance the data within a range of tolerance.
    """

    pass
