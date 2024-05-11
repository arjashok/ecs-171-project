"""
    @brief Dataset abstraction.
    @author Arjun Ashok
"""


# Environment Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, GridSearch

from dataclasses import dataclass, field

from utils.explore_dataset import *
from utils.transform_dataset import *


# Classes
@dataclass
class TreeClassifier:
    # user-set members
    target: str = field()                                                       # target feature
    data: pd.DataFrame = field(default=None)                                    # dataset

    # inferred members
    model: GradientBoostingClassifier = field(default=None)                     # underlying model
    X_train: np.ndarray = field(default=None)                                   # data for training/testing
    y_train: np.ndarray = field(default=None)                                   # data for training/testing
    X_test: np.ndarray = field(default=None)                                    # data for training/testing
    y_test: np.ndarray = field(default=None)                                    # data for training/testing

    # internal methods
    def __post_init__(self):
        self.data = pd.read_csv(self.path, engine="c")


    # external methods
    ## mutators
    def set_hyperparams(self, hyperparams: dict[str, int | float | str]=None) -> None:
        """
            Sets hyperparameters. Note, if not specified it automatically 
            generates via grid-search.

            @param hyperparams: hyperparams to update
        """

        # default hyperparams
        self.hyperparams = {
            "loss": "log_loss",
            "learning_rate": 0.01,
            "n_estimators": 500,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 5,
            "max_depth": 5,
            "n_iter_no_change": 5,
            "tol": 1e-4
        }

        # if update not required
        if hyperparams is None:
            self.optimize_hyperparams(with_cv=True)
            return
        
        # update
        for hp, val in hyperparams:
            # include hyperparameter even if not expected
            if hp not in self.hyperparams:
                print(f"<WARNING> unexpected hyperparameter {hp} found; included anyways with value {val}")
            self.hyperparams[hp] = val


    ## utility
    def train_model(self) -> None:
        """
            Trains the model, assuming no hyperparameter optimization.
        """

        pass


    def test_model(self) -> None:
        """
            Generates test error.
        """

        pass

    
    def predict(self) -> None:
        """
            Generates predictions for use in the test data.
        """

        pass


    def optimize_hyperparams(self, grid_search: dict[str, list[int | float | str]]=None,
                             kfold: int=None) -> pd.DataFrame:
        """
            Optimizes hyperparameters via grid search.

            @param grid_search: set of parameters to search through; defaults to 
                                an expected set of a good parameters
            @param kfold: number of folds to do cross validation with; if None, 
                          defaults to no cross validation

            @returns dataframe of the searcher results for every combination 
                     specified in the search
        """

        # setup search
        if grid_search is None:
            grid_search = {
                "loss": ["log_loss", "exponential"],
                "learning_rate": [10 ** (-i) for i in range(5)],
                "n_estimators": [100, 200, 500, 1000, 2000],
                "criterion": ["friedman_mse", "squared_error"],
                "min_samples_split": [2 * i for i in range(5)],
                "min_samples_leaf": [1, 2, 5, 10, 15, 20, 25, 100],
                "max_depth": [3, 5, 10],
                "n_iter_no_change": [5, 10],
                "tol": 1e-4
            }
        
        # conduct search
        searcher = GridSearchCV(self.model, grid_search, scoring="accuracy", 
                                refit=True, cv=kfold, verbose=1).fit(self.X_train, self.y_train)
        searcher_df = pd.DataFrame.from_dict(searcher.cv_results_)
        return searcher_df


