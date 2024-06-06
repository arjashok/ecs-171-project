"""
    @brief Wrapper/pipelines for easy handling of data.
    @author Arjun Ashok
"""


# Environment Setup
from utils.dataset import *


# Wrappers
def pre_split_pipeline(path: str, target: str) -> Dataset:
    """
        Augments data as necessary prior to splitting; assumes features have 
        been selected already.

        @path: path to feature-selected data; assumes columns have been 
               standardized already
    """

    # nothing to do before split after feature selection
    return None


def post_split_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.DataFrame, y_test: pd.DataFrame, 
                        target: str, categorical_features: list[str]=None,
                        upsample: bool=False) -> tuple[np.ndarray, np.ndarray, np.array, np.array]:
    """
        Augments data post split as necessary.
    """

    # # get categ
    # if categorical_features is None:
    #     ds = Dataset("../datasets/pre_split_processed.parquet", target=target)
    #     ds.set_features(features)
    #     ds.infer_types()

    #     categorical_features = list(ds.ordinal_features) + list(ds.nominal_features)

    # normalize & upsample train
    scaler = StandardScaler()
    # X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    # train_data = pd.concat([X_train, y_train], axis=1)
    
    # if upsample:
    #     train_data = up_sampling(train_data, target=target, categorical_features=categorical_features)
    
    # X_train, y_train = train_data.drop(columns=target), train_data[target]
    
    scaler = scaler.fit(X_train[X_train.columns])
    
    if upsample:
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data = up_sampling(train_data, target=target, categorical_features=categorical_features)
        X_train, y_train = train_data.drop(columns=target), train_data[target]
    
    X_train[X_train.columns] = scaler.transform(X_train[X_train.columns])
    
    # normalize test
    # X_test = normalize_features(X_test, features=features, scaler=scaler)

    # export
    return X_train, X_test, y_train, y_test, scaler

