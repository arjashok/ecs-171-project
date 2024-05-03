"""
    @brief Utility for the feature selection via visualization, correlations, 
           Ridge regression, and more.
    @author Arjun Ashok
"""


import pandas as pd
from sklearn import linear_model
from transform_dataset import *
from explore_dataset import *


# Visualizations & Correlations Utility



# Ridge Regression Utility
def feature_selection(df: pd.DataFrame, target: str, regression_type: str="lasso", 
                      target_type: str="classification",
                      numeric_features: list[str]=None) -> dict[str, float]:
    """
        Does feature selection via regularized regression, i.e. any features 
        deemed useless will have a low to zero coefficient in the final model if 
        a regularization term is applied.

        @param df: dataset; assumed to be one-hot encoded already
        @param regression_type: can be one of `ridge` or `lasso`
        @param target_type: whether to do linear or logistic regression
    """

    # execute params
    regression_lookup = {
        "ridge": {
            "classification": linear_model.RidgeClassifier,
            "regression": linear_model.Ridge,
        },
        "lasso": {
            "classification": linear_model.LogisticRegression(penalty="l1"),
            "regression": linear_model.Lasso
        }
    }
    model = regression_lookup[regression_lookup][target_type]

    if numeric_features is None:
        numeric_features, _, _ = infer_types(df, inplace=False)

    # ensure normalized data, otherwise feature coefficients are meaningless
    df = normalize_features(df, features=numeric_features)
    X, y = split_target(df, target)

    # regression
    model.fit(df[])


