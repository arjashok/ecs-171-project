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


# Utility Functions
def numerical_summary(df: pd.DataFrame) -> None:
    """
        Generates a few basic summaries about the dataset.

        @param df: pandas dataframe object
    """

    # print summaries
    print(f"<Peak Dataset>\n{df.head()}\n\n")
    print(f"<DataFrame Info>\n{df.info()}\n\n")
    print(f"<Distributions of Features>\n{df.describe()}\n\n")


def visualize_relationships(df: pd.DataFrame) -> None:
    """
        Visualizations the relationships between the different features in a 
        dataframe.
    """

    # correlation + distributions
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(df[infer_binary_columns(df)].sample(1000), kind="kde", diag_kind="kde")
    plt.title("Pair Plot of Features")
    plt.show()


def infer_binary_columns(df: pd.DataFrame, features: set=None) -> list[str]:
    """
        Infers binary distributions for specified columns.
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


# Testing
if __name__ == "__main__":
    df = pd.read_csv("../datasets/diabetes.csv", engine="c")
    numerical_summary(df)
    visualize_distributions(df, cols=4, ignore_binary=True)
    visualize_relationships(df)

