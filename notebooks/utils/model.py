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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import pickle
import json
import os
from dataclasses import dataclass, field

from utils.explore_dataset import *
from utils.transform_dataset import *


# Classes
@dataclass(slots=True)
class TreeClassifier:
    # user-set members
    target: str = field()                                                       # target feature
    path: str = field()                                                         # dataset path
    hyperparams: dict[str, int | float | str] = field(default=None)             # hyperparams; optional parameter

    # inferred members
    data: pd.DataFrame = field(default=None)                                    # dataset
    model: GradientBoostingClassifier = field(default=None)                     # underlying model

    # calculated members
    X_train: np.ndarray = field(default=None)                                   # data for training/testing
    y_train: np.ndarray = field(default=None)                                   # data for training/testing
    X_test: np.ndarray = field(default=None)                                    # data for training/testing
    y_test: np.ndarray = field(default=None)                                    # data for training/testing
    score: dict[str, int | float] = field(default=None)                         # scores dict for the accuracy report

    # internal methods
    def gen_split(self, test_prop: float=0.2) -> None:
        """
            Generates train/test split and a report along with it
            
            @param test_prop: proportion of dataset to reserve for testing
            @param stratify: whether to stratify or not
        """

        # split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.drop(columns=self.target),
            self.data[self.target],
            test_size=test_prop,
            random_state=42
        )

        # report
        print("<Train-Test Split Report>")
        print(f"Train: {len(self.y_train)} obs, {(self.y_train == 0).sum()} no diabetes [0], {(self.y_train == 1).sum()} pre-diabetes [1], {(self.y_train == 2).sum()} diabetes [2]")
        print(f"Test: {len(self.y_test)} obs, {(self.y_test == 0).sum()} no diabetes [0], {(self.y_test == 1).sum()} pre-diabetes [1], {(self.y_test == 2).sum()} diabetes [2]")


    def __post_init__(self):
        # data
        self.data = pd.read_parquet(self.path)
        self.gen_split()

        # model
        self.set_hyperparams(self.hyperparams, optimize=False)
        if self.model is None:
            self.model = self.model = GradientBoostingClassifier(**self.hyperparams)


    # external methods
    ## mutators
    def set_hyperparams(self, hyperparams: dict[str, int | float | str]=None, optimize: bool=False) -> None:
        """
            Sets hyperparameters. Note, if not specified it automatically 
            generates via grid-search.

            @param hyperparams: hyperparams to update
        """

        # default hyperparams
        self.hyperparams = {
            "loss": "log_loss",
            "learning_rate": 0.01,
            "n_estimators": 100,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 5,
            "max_depth": 3,
            "n_iter_no_change": 5,
            "max_features": "log2",
            "tol": 1e-4
        }

        # if no update is required
        if hyperparams is None:
            return

        # if optimization is required
        if optimize:
            self.optimize_hyperparams(with_cv=True)
            return
        
        # update
        for hp, val in hyperparams:
            # include hyperparameter even if not expected
            if hp not in self.hyperparams:
                print(f"<WARNING> unexpected hyperparameter {hp} found; included anyways with value {val}")
            self.hyperparams[hp] = val

        # update model
        self.model = GradientBoostingClassifier(**self.hyperparams)


    ## utility
    def save_model(self) -> None:
        """
            Saves model weights and hyperparams for future reloading.
        """

        # generate save path
        report_name = "-".join([f"[{metric[0]}_{perf:.4f}]" for metric, perf in self.score.items()])

        # save both
        with open(f"../models/hyperparams/tree-classifier-{self.score['a']}.json") as f:
            json.dump(self.hyperparams, f, indent=4)
        pickle.dump(self.model, open(f"../models/weights/tree-classifier-{report_name}.pickle", "wb"))

        # save lookup
        if os.path.exists(self.model_lookup_path):
            model_reports = pd.read_csv(self.model_lookup_path)
        else:
            model_reports = pd.DataFrame()
        
        add_row = pd.DataFrame({
            "model-type": "XGBoost",
            **self.score,
            "path": f"tree-classifier-{report_name}"
        })
        model_reports = pd.concat([model_reports, add_row], ignore_index=True)
        model_reports.to_csv(self.model_lookup_path, index=False)

    
    def load_model(self, scores: dict[str, int | float]=None, priority_list: list[str]=None) -> None:
        """
            Loads the model via the scoring report. Notice that either explicit 
            scores or a prioritization of the scores can be provided and then 
            the best performing model will be chosen. This is done via stable 
            sort.

            @param scores: metric to performance wanted; optional
            @param priority_list: list of metrics in order of most to least 
                                  important
        """

        # load lookup
        if not os.path.exists(self.model_lookup_path):
            print("<WARNING> no lookup found for saved models :(")
            return
        model_reports = pd.read_csv(self.model_lookup_path, engine="c")
        if model_reports.shape[0] == 0:
            print("<WARNING> found 0 entries in model lookup :(")
            return

        path = None

        # ensure at least one is filled
        if scores is None and priority_list is None:
            priority_list = [
                "f1-score",
                "accuracy",
                "precision",
                "recall"
            ]

        # get model entry
        if scores is not None:
            # narrow down
            for metric, perf in scores.items():
                model_reports = model_reports[model_reports[metric] == perf]
            
            # if any entry matches the description
            if model_reports.shape[0] != 0:
                path = model_reports["path"][0]
        
        else:
            # stable sort
            model_reports.sort_values(by=priority_list)
            path = model_reports["path"][0]
        
        # load model
        self.model = pickle.load(open(f"../models/weights/{path}.pickle", "rb"))
        self.hyperparams = json.load(open(f"../models/hyperparams/{path}.json", "r"))


    def train_model(self) -> None:
        """
            Trains the model, assuming no hyperparameter optimization.
        """

        # fit model
        self.model.fit(self.X_train, self.y_train)


    def test_model(self) -> None:
        """
            Generates test error.
        """

        # predict
        y_pred = self.predict(self.X_test)
        y_test = self.y_test

        # metrics + report
        labels = self.data[self.target].unique()
        p, r, f, s = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=labels
        )
        a = accuracy_score(y_test, y_pred)

        print("\n<Test Report>")
        print(f"Precision: [no diabetes] {p[0]}, [pre-diabetes] {p[1]}, [diabetes] {p[2]}")
        print(f"Recall: [no diabetes] {r[0]}, [pre-diabetes] {r[1]}, [diabetes] {r[2]}")
        print(f"F1-Score: [no diabetes] {f[0]}, [pre-diabetes] {f[1]}, [diabetes] {f[2]}")
        print(f"Support: [no diabetes] {s[0]}, [pre-diabetes] {s[1]}, [diabetes] {s[2]}")
        print(f"Accuracy: {a * 100:.4f}%")

        # export weights
        self.score = [{"label": label, "precision": p[label], "recall": r[label], \
                       "f1-score": f[label], "support": s[label], "accuracy": a * 100} \
                        for label in labels]
        self.save_model()


    def predict(self, X: np.ndarray) -> np.array:
        """
            Generates predictions for use in the test data.

            @param X: data to predict on
        """

        # wrap predictions
        return self.model.predict(X)


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
                "loss": ["log_loss"],
                "learning_rate": [10 ** (-i) for i in range(4)],
                "n_estimators": [100, 500],
                "criterion": ["friedman_mse"],
                "min_samples_split": [2],
                "min_samples_leaf": [2, 10],
                "max_depth": [3, 5],
                "n_iter_no_change": [5],
                "max_features": ["log2"],
                "tol": [1e-4]
            }
            # grid_search = {
            #     "loss": ["log_loss"],
            #     "learning_rate": [10 ** (-i) for i in range(1)],
            #     "n_estimators": [100],
            #     "criterion": ["friedman_mse"],
            #     "min_samples_split": [2],
            #     "min_samples_leaf": [2],
            #     "max_depth": [3],
            #     "n_iter_no_change": [5],
            #     "max_features": ["log2"],
            #     "tol": [1e-4]
            # }
        
        # conduct search
        searcher = GridSearchCV(
            self.model,
            grid_search,
            scoring="accuracy",
            refit=True,
            cv=kfold,
            verbose=3,
            n_jobs=-1
        ).fit(self.X_train, self.y_train)

        searcher_df = pd.DataFrame.from_dict(searcher.cv_results_)
        accuracy = searcher.best_score_
        self.model = searcher.best_estimator_
        self.hyperparams = searcher.best_params_

        # export params, weights
        self.test_model()
        self.save_model()


