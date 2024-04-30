"""
    @brief Utility for the initial exploration of a dataset.
    @author Arjun Ashok
"""


# Environment Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Data Understanding Utility
def check_integrity(df: pd.DataFrame, remove_na: bool=False, inplace: bool=True) -> None | pd.DataFrame:
    """
        Checks for missing/NA/NaN/None values and suspicious distributions (i.e. 
        all one value, etc.). Prints a summary and removes if specified.

        @param df: dataset
        @param remove_na: whether to remove the missing values or not
        @param inplace: whether to conduct operations on the original data
    """

    # check inplace
    if remove_na and not inplace:
        df = df.copy()

    # tracking useless columns
    useless_cols = list()

    # iterate columns and check
    print("<Checking Data Integrity>")
    for col in df.columns:
        # report
        print("\tChecking", f"{f'<{col}>...':<25}", end="")
        num_unique = df[col].nunique()
        num_missing = df[col].isna().sum()

        if num_unique > 1 and num_missing == 0:
            print(f"\t<cleared>")
        else:
            if num_unique <= 1:
                print(f"\n\t<WARNING> :: ONLY {num_unique} unique value(s)", end="")
                useless_cols.append(col)
            if num_missing > 0:
                print(f"\n\t<WARNING> :: {num_missing}, i.e. {(num_missing / df.shape[0]) * 100:.2f}%, missing entries")
    
    # exporting
    if remove_na:
        # remove & report missing
        num_bef = df.shape[0]
        df.dropna(inplace=True)
        num_aft = df.shape[0]

        if num_aft != num_bef:
            print(f"\t<CAUTION> removed {num_aft - num_bef}, i.e.",
                  f"{(num_aft - num_bef) / num_bef * 100:.2f}%, entries from",
                  f"{num_bef} --> {num_aft}")
    
        # remove & report cols
        if len(useless_cols) > 0:
            df.drop(columns=useless_cols, inplace=True)
            print(f"\t<CAUTION> removed {len(useless_cols)} columns: {', '.join(useless_cols)}")
        

    if not inplace:
        return df


def infer_types(df: pd.DataFrame, apply_inference: bool=False, 
                inplace: bool=True) -> tuple[set[str], set[str], None | pd.DataFrame]:
    """
        Infers the column types (i.e. categorical or numeric) based on 
        distributions and uniqueness. This relies on heuristics about the 
        proportion of entries.

        @param df: dataset
        @param apply_inference: whether to specify the dtypes gathered in the df 
                                or not (doesn't really apply to ordinal beyond 
                                float --> int conversions)
        @param inplace: whether to modify the dataset obj directly
    """

    # trackers + setup
    categ_cols = list()
    numeric_cols = list()
    THRESHOLD = 0.01                            # expect at least 1/100th rows to have unique values
    RAW_THRESHOLD = int(np.log(df.shape[0]))    # expect an ln increase of unique values
    int_conv, str_conv, num_conv = 0, 0, 0

    # inplace
    if return_df := apply_inference and not inplace:
        df = df.copy()

    # check col types
    print("<Dtype Inference>")
    for col in df.columns:
        # distribution
        num_unique = df[col].nunique()
        dtype = df[col].dtype

        # inference
        if num_unique / df.shape[0] < THRESHOLD and num_unique <= RAW_THRESHOLD:
            # check conversion
            if apply_inference:
                if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, float):
                    df[col] = df[col].astype(int)
                    int_conv += 1
                else:
                    df[col] = df[col].astype(str)
                    str_conv += 1
            
            # track categorical
            categ_cols.append(col)
        else:
            # check conversion
            if apply_inference:
                if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, float)):
                    # attempt conversion if not already numeric
                    try:
                        df[col] = df[col].astype(float)
                        num_conv += 1
                    except Exception as e:
                        print(f"\t<WARNING> failed to convert {col} to numeric, {e}")

            # track numeric
            numeric_cols.append(col)

    # report + export
    print(f"\t{len(numeric_cols)} numeric vars, {len(categ_cols)} categorical vars")
    if apply_inference:
        print(f"\tEnforced {int_conv} vars to ordinal, {str_conv} vars to categorical, and {num_conv} to numeric")
    
    return numeric_cols, categ_cols, df if return_df else None


# Distributions Utility
def numerical_summary(df: pd.DataFrame) -> None:
    """
        Generates a few basic summaries about the dataset.

        @param df: pandas dataframe object
    """

    # print summaries
    print(f"<Peak Dataset>\n{df.head()}\n\n")
    print(f"<DataFrame Info>\n")
    df.info()
    print(f"\n\n<Distributions of Features>\n{df.describe()}\n\n")


def visualize_relationships(df: pd.DataFrame) -> None:
    """
        Visualizations the relationships between the different features in a 
        dataframe.

        @param df: dataset
    """

    # correlation + distributions
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop(columns=infer_binary_columns(df)).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(df.drop(columns=infer_binary_columns(df)).sample(1000), diag_kind="kde")
    plt.title("Pair Plot of Features")
    plt.show()


def infer_binary_columns(df: pd.DataFrame, features: set=None) -> list[str]:
    """
        Infers binary distributions for specified columns.

        @param df: dataset
        @param features: [optional] which features to consider
    """
    
    # determine distribution via unique values
    if features is None:
        features = set(df.columns)
    return [col for col in features if df[col].nunique() <= 2]


def visualize_distributions(df: pd.DataFrame, plot_features: list[str]=None, 
                            rows: int=None, cols: int=None, 
                            ignore_binary: bool=False) -> None:
    """
        Visualize each features distributions.

        @param df: dataset
        @param plot_features: features to plot distributions
        @param rows, cols: specify plots in the x, y direction
        @param ignore_binary: whether to plot binary (false) or not
    """

    # ensure plotted features
    if plot_features is None:
        plot_features = list(df.columns)
    if ignore_binary:
        binary_cols = set(infer_binary_columns(df))
        plot_features = list(set(plot_features) - binary_cols)

    # setup
    num_features = len(plot_features)

    # find optimal dimensions for subplots
    optimal_cols = min(4, int(np.sqrt(num_features)))
    optimal_rows = int(np.ceil(num_features / optimal_cols))

    if rows is None and cols is None:
        rows = optimal_rows
        cols = optimal_cols
    elif rows is None:
        rows = int(np.ceil(num_features / cols))
    elif cols is None:
        cols = int(np.ceil(num_features / rows))

    # generate figure
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(plot_features, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[feature], bins=20, kde=True, stat="proportion")
        plt.title(f"Distribution of \"{feature}\"")

    plt.tight_layout()
    plt.show()


def target_explorations(df: pd.DataFrame, target: str) -> dict[str, int]:
    """
        Explores the distribution of the target feature to check the necessity 
        for sampling or otherwise redistributing the data.

        @param df: dataset
        @param target: name of the target feature in the dataset
    """

    # generate counts
    target_counts = dict(df[target].value_counts())
    num_labels = len(target_counts)
    num_obs = len(df)
    PADDING = 0.10                                                  # allow for +- 10% w/o error
    
    # check distributions
    print(f"\n<Target Distribution \"{target}\">")
    for label, count in target_counts.items():
        print(f"Label `{label}` has {count} observations", end="")

        if (count > num_obs / num_labels + PADDING * num_obs) or (count < num_obs / num_labels - PADDING * num_obs):
            print(f" ==> <WARNING> outside of balanced range of observations")
        else:
            print(f" ==> within tolerance range...")
    
    # export
    return target_counts


# Testing
if __name__ == "__main__":
    df = pd.read_csv("../datasets/diabetes.csv", engine="c")
    numerical_summary(df)
    visualize_distributions(df, cols=4, ignore_binary=True)
    visualize_relationships(df)

