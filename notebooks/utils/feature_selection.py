"""
    @brief Utility for the feature selection via visualization, correlations, 
           Ridge regression, and more.
    @author Arjun Ashok
"""


import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.feature_selection import chi2
from utils.transform_dataset import *
from utils.explore_dataset import *


# Visualizations & Correlations Utility
def visualize_target_corr(df: pd.DataFrame, target: str, numeric_features: list[str]=None,
                          ordinal_features: list[str]=None, nominal_features: list[str]=None,
                          feature_set: list[str]=None) -> dict[str, dict[str, float]]:
    """
        Visualizes how strong the relationship of every numeric/ordinal feature 
        is to the target variable. NOTE we assume an encoded dataframe.

        @param df: dataset, should be encoded already
        @param target: target feature
        @param feature_set: what features to consider, defaults to all but target
    """

    # setup and fill args
    if feature_set is None:
        feature_set = list(df.columns)
    feature_set = set(feature_set)

    if numeric_features is None or ordinal_features is None or nominal_features is None:
        numeric_features, ordinal_features, nominal_features = infer_types(df, inplace=False)
    numeric_features = feature_set & set(numeric_features)
    ordinal_features = feature_set & set(ordinal_features)
    nominal_features = feature_set & set(nominal_features)
    
    # get correlations
    target_corr = dict(zip(
        ["numeric [pearson]", "ordinal [spearman]", "nominal [kendall-tau]"], 
        [dict() for _ in range(3)]
    ))
    target_vec = df[target]

    for col in numeric_features:
        target_corr["numeric [pearson]"][col] = pearsonr(df[col], target_vec).statistic
    for col in ordinal_features:
        target_corr["ordinal [spearman]"][col] = spearmanr(df[col], target_vec).statistic
    for col in nominal_features:
        # target_corr["nominal [chi-square]"][col] = chi2(np.reshape(df[col].to_numpy(), (-1, 1)), target_vec)[0][0]
        target_corr["nominal [kendall-tau]"][col] = kendalltau(df[col], target_vec).statistic
    
    # visualization
    def _plot_corr(corr, title_aug):
        df = pd.DataFrame({
            "features": list(corr.keys()),
            "corr": list(corr.values())
        })

        plt.figure(figsize=(8, 6))
        sns.barplot(x="features", y="corr", data=df, palette="viridis", hue="features")
        plt.title(f"Correlations with Target Feature {target} for {title_aug} Features")
        plt.xlabel("Features")
        plt.ylabel("Correlation")
        plt.xticks(rotation=45)
        plt.show()
    
    _plot_corr(target_corr["numeric [pearson]"], "Numeric")
    _plot_corr(target_corr["ordinal [spearman]"], "Ordinal")
    _plot_corr(target_corr["nominal [kendall-tau]"], "Nominal")
    return target_corr


def collinearity_check(df: pd.DataFrame, target: str, threshold: float=0.25, 
                       min_list: int=5) -> pd.DataFrame:
    """
        Checks for collinearity and returns any questionable features.

        @param df: dataset; assumes at the very least all numerically encoded
        @param target: target feature name
        @param threshold: correlation required for attention
        @param min_list: number of features to list at the minimum
    """

    # generate correlations by taking magnitude and keeping only upper triangular portion (avoid duplicates and same-same corr)
    correlations = df.drop(columns=target).corr()

    upper_tri_mask = np.triu(np.ones(correlations.shape), k=1).astype(bool)
    trunc_correlations = correlations.where(upper_tri_mask)
    magnitude_mask = trunc_correlations.abs() > threshold
    trunc_correlations = trunc_correlations[magnitude_mask]

    # selection
    min_list = max(magnitude_mask.sum().sum(), min_list)
    sorted_correlations = trunc_correlations.unstack().sort_values(ascending=False, key=abs)

    top_corr_pairs = sorted_correlations.head(min_list).reset_index()
    top_corr_pairs.columns = ["feature-1", "feature-2", "correlation"]

    # export
    pprint(top_corr_pairs)
    return top_corr_pairs


# Ridge Regression Utility
def feature_selection(df: pd.DataFrame, target: str, regression_type: str="lasso", 
                      target_type: str="classification",
                      numeric_features: list[str]=None) -> tuple[list[str], list[str], dict[str, float]]:
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
        "lasso": {
            "classification": linear_model.LogisticRegression(penalty="l1", solver="saga", max_iter=100000),
            "regression": linear_model.Lasso
        },
        "ridge": {
            "classification": linear_model.LogisticRegression(penalty="l2", max_iter=100000),
            "regression": linear_model.Ridge
        }
    }

    if numeric_features is None:
        numeric_features, _, _ = infer_types(df, inplace=False)

    # model setup
    model = regression_lookup[regression_type][target_type]

    # ensure normalized data, otherwise feature coefficients are meaningless
    df = normalize_features(df, features=list(df.columns))#, features=numeric_features)
    X, y = split_target(df, target)

    # ensure target is a binary predictor if classification is used
    if target_type == "classification":
        y = y > 0

    # regression
    model.fit(X, y)

    # showcase results & auto-prune if needed
    coefs = list(model.coef_[0])
    features = X.columns

    coefs_lookup = dict(zip(features, coefs))
    pruned_features = [feature for feature, w in zip(X.columns, coefs) if np.isclose(w, 0, atol=1e-5)]

    if len(pruned_features) == 0:
        # generate the lowest magnitude features for manual pruning
        print("<Note> couldn't find obvious features to prune, listing lowest magnitude")
        pruned_features = {k: abs(v) for k, v in coefs_lookup.items()}
        pruned_features = dict(sorted(pruned_features.items(), key=lambda p: p[1]))
        pruned_features = {k: v for i, (k, v) in enumerate(pruned_features.items()) if i < 5}

    kept_features = list(set(features) - set(pruned_features))

    print_collection(coefs_lookup)
    print_collection(pruned_features)
    print_collection(kept_features)

    return pruned_features, kept_features, coefs_lookup 


