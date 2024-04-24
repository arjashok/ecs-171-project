"""
    @brief Utility for the initial exploration of a dataset.
    @author Arjun Ashok
"""


# Environment Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    sns.pairplot(df)
    plt.title("Pair Plot of Features")
    plt.show()


def visualize_distributions(df: pd.DataFrame, plot_features: list[str]=None) -> None:
    """
        Visualize each features distributions.
    """

    # ensure plotted features
    if plot_features is None:
        plot_features = list(df.columns)
    num_features = len(plot_features)

    # find optimal dimensions for subplots
    optimal_rows = int(np.sqrt(num_features))
    optimal_cols = int(np.ceil(num_features / optimal_rows))

    # generate figure
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(plot_features, 1):
        print(feature)
        plt.subplot(optimal_rows, optimal_cols, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of \"{feature}\"")

    plt.tight_layout()
    plt.show()

