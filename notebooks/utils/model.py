"""
    @brief Dataset abstraction.
    @author Arjun Ashok
"""


# Environment Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import pickle
import json
import os
import datetime
from dataclasses import dataclass, field
from typing import Any

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
    model_lookup_path: str = field(default="../models/model_lookup.csv")        # model lookup

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
            self.model = GradientBoostingClassifier(**self.hyperparams)


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
        mean_scores = ({
            k: sum(d[k] for d in self.score) / len(self.score) 
            for k in self.score[0]
        })
        report_name = "tree-classifier-" + ("-".join([
            f"[{metric[0]}_{perf:.4f}]" for metric, perf in mean_scores.items()
            if metric[0] not in ["l", "s"]
        ]))

        # save both
        json.dump(self.hyperparams, open(f"../models/hyperparams/{report_name}.json", "w"), indent=4)
        pickle.dump(self.model, open(f"../models/weights/{report_name}.pickle", "wb"))

        # save lookup
        if os.path.exists(self.model_lookup_path):
            model_reports = pd.read_csv(self.model_lookup_path)
        else:
            model_reports = pd.DataFrame()
        
        # add row for each label's performance
        add_rows = ([
                pd.DataFrame({
                "model-type": "XGBoost",
                **{k: [v] for k, v in score_dict.items()},
                "path": {report_name}
            })
            for score_dict in self.score
        ])
        model_reports = pd.concat([model_reports, *add_rows], ignore_index=True)
        model_reports.to_csv(self.model_lookup_path, index=False)

    
    def load_model(self, scores: dict[str, int | float]=None, priority_list: list[str]=None) -> bool:
        """
            Loads the model via the scoring report. Notice that either explicit 
            scores or a prioritization of the scores can be provided and then 
            the best performing model will be chosen. This is done via stable 
            sort.

            @param scores: metric to performance wanted; optional
            @param priority_list: list of metrics in order of most to least 
                                  important

            @returns whether successful or not
        """

        # load lookup
        if not os.path.exists(self.model_lookup_path):
            print("<WARNING> no lookup found for saved models :(")
            return False
        model_reports = pd.read_csv(self.model_lookup_path, engine="c")
        model_reports = model_reports[model_reports["model-type"] == "tree"]

        if model_reports.shape[0] == 0:
            print("<WARNING> found 0 entries in model lookup :(")
            return False

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
        if path is None:
            return False
        self.model = pickle.load(open(f"../models/weights/{path}.pickle", "rb"))
        self.hyperparams = json.load(open(f"../models/hyperparams/{path}.json", "r"))

        return True


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

        print(json.dumps(self.hyperparams, indent=4))

        # export params, weights
        if not os.path.exists("../models/grid-searches/"):
            os.makedirs("../models/grid-searches/")
            searcher_df.to_csv(f"../models/grid-searches/tree-classifier-{datetime.datetime.now().strftime('%m-%d-%y-%H')}.csv", index=False)
        
        self.test_model()
        self.save_model()


class LinearNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden, num_epochs, learning_rate, batch_size):
        """
            Initialize model based on hyperparams. This is a normal FFNN with 
            ReLU
        """
        
        # setup model arch
        super(LinearNN, self).__init__()
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden)
        ])
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.classify_fn = nn.Sigmoid() #nn.Softmax(dim=1)

        # setup params
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def forward(self, x):
        """
            Propagate information through network.
        """

        out = self.fc_input(x)
        out = self.relu(out)
        for layer in self.hidden_layers:
            out = layer(out)
            out = self.relu(out)
        out = self.fc_output(out)
        out = self.classify_fn(out)
        return out


class TabularDataset(Dataset):
    def __init__(self, X, device, y=None):
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass(slots=True)
class MLPClassifier:
    # user-set members
    target: str = field()                                                       # target feature
    path: str = field()                                                         # dataset path
    hyperparams: dict[str, int | float | str] = field(default=None)             # hyperparams; optional parameter

    # inferred members
    data: pd.DataFrame = field(default=None)                                    # dataset
    model: LinearNN = field(default=None)                                       # underlying model
    model_lookup_path: str = field(default="../models/model_lookup.csv")        # model lookup
    optimizer: Any = field(default=None)                                        # optimizer function
    loss_fn: Any = field(default=None)                                          # loss for neural network
    device: Any = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available else "cpu")
    )                                                                           # device to use; tries for GPU optimization

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
            self.model = LinearNN(**self.hyperparams).to(self.device)


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
            "input_size": self.X_train.shape[1],
            "output_size": self.y_train.nunique(),
            "hidden_size": 64,
            "num_hidden": 2,
            "num_epochs": 25,
            "batch_size": 32,
            "learning_rate": 0.01
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
        self.model = LinearNN(**self.hyperparams)


    ## utility
    def save_model(self) -> None:
        """
            Saves model weights and hyperparams for future reloading.
        """

        # generate save path
        mean_scores = ({
            k: sum(d[k] for d in self.score) / len(self.score) 
            for k in self.score[0]
        })
        report_name = "ffnn-classifier-" + ("-".join([
            f"[{metric[0]}_{perf:.4f}]" for metric, perf in mean_scores.items()
            if metric[0] not in ["l", "s"]
        ]))

        # save both
        json.dump(self.hyperparams, open(f"../models/hyperparams/{report_name}.json", "w"), indent=4)
        torch.save(self.model.state_dict(), f"../models/weights/{report_name}.pt")

        # save lookup
        if os.path.exists(self.model_lookup_path):
            model_reports = pd.read_csv(self.model_lookup_path)
        else:
            model_reports = pd.DataFrame()
        
        # add row for each label's performance
        add_rows = ([
                pd.DataFrame({
                "model-type": "FFNN",
                **{k: [v] for k, v in score_dict.items()},
                "path": {report_name}
            })
            for score_dict in self.score
        ])
        model_reports = pd.concat([model_reports, *add_rows], ignore_index=True)
        model_reports.to_csv(self.model_lookup_path, index=False)

    
    def load_model(self, scores: dict[str, int | float]=None, priority_list: list[str]=None) -> bool:
        """
            Loads the model via the scoring report. Notice that either explicit 
            scores or a prioritization of the scores can be provided and then 
            the best performing model will be chosen. This is done via stable 
            sort.

            @param scores: metric to performance wanted; optional
            @param priority_list: list of metrics in order of most to least 
                                  important

            @returns whether successful or not
        """

        # load lookup
        if not os.path.exists(self.model_lookup_path):
            print("<WARNING> no lookup found for saved models :(")
            return False
        model_reports = pd.read_csv(self.model_lookup_path, engine="c")
        model_reports = model_reports[model_reports["model-type"] == "ffnn"]

        if model_reports.shape[0] == 0:
            print("<WARNING> found 0 entries in model lookup :(")
            return False

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
        if path is None:
            return False
        
        self.model = LinearNN(**self.hyperparams)
        self.model.load_state_dict(torch.load(f"../models/weights/{path}.pt"))
        self.model.eval()
        self.hyperparams = json.load(open(f"../models/hyperparams/{path}.json", "r"))

        return True


    def train_model(self) -> None:
        """
            Trains the model, assuming no hyperparameter optimization.
        """

        # setup gradient descent
        self.optimizer = Adam(self.model.parameters(), lr=self.model.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            TabularDataset(X=self.X_train, y=self.y_train, device=self.device),
            batch_size=self.model.batch_size,
            shuffle=True
        )

        # fit model & track loss
        for epoch in range(self.model.num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # loss + backprop
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # track loss
                running_loss += loss.item()

            # Print statistics
            print(f"Epoch {epoch + 1}/{self.model.num_epochs}, Loss: {running_loss / len(train_loader)}")


    def test_model(self) -> None:
        """
            Generates test error.
        """

        # predict
        y_pred, y_conf = self.predict(self.X_test)
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


    def predict(self, X: np.ndarray) -> tuple[np.array, np.array]:
        """
            Generates predictions for use in the test data.

            @param X: data to predict on
        """

        # gen tensors
        X = torch.tensor.from_numpy(X.to_numpy()).to(self.device)

        # predict without backprop
        self.model.eval()

        with torch.no_grad():
            # forward pass
            outputs = self.model(X)

            # append predictions & the raw prediction value
            confidence, predictions = torch.max(outputs, 1, dim=1)

        # wrap predictions
        return np.array(predictions), np.array(confidence)


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
                "learning_rate": [10 ** (-i) for i in range(5)],
                "input_size": self.X_train.shape[1],
                "output_size": self.y_train.nunique(),
                "hidden_size": [32, 64, 96, 128],
                "num_hidden": [1, 2, 3],
                "num_epochs": [10, 25],
                "batch_size": [32, 64, 128]
            }
        
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

        print(json.dumps(self.hyperparams, indent=4))

        # export params, weights
        if not os.path.exists("../models/grid-searches/"):
            os.makedirs("../models/grid-searches/")
            searcher_df.to_csv(f"../models/grid-searches/tree-classifier-{datetime.datetime.now().strftime('%m-%d-%y-%H')}.csv", index=False)
        
        self.test_model()
        self.save_model()
