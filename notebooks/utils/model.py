"""
    @brief Model abstraction.
    @author Arjun Ashok
"""


# Environment Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from skorch import NeuralNetClassifier
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

import pickle
import json
import os
import datetime
import copy
from itertools import product
from dataclasses import dataclass, field
from typing import Any

from utils.explore_dataset import *
from utils.transform_dataset import *
from utils.wrappers import *


# Classes
@dataclass(slots=True)
class TreeClassifier:
    # user-set members
    target: str = field()                                                       # target feature
    path: str = field()                                                         # dataset path
    hyperparams: dict[str, int | float | str] = field(default=None)             # hyperparams; optional parameter
    upsample: bool = field(default=False)                                       # whether to correct imbalance or not

    # inferred members
    data: pd.DataFrame = field(default=None)                                    # dataset
    model: GradientBoostingClassifier = field(default=None)                     # underlying model
    model_lookup_path: str = field(default="../models/model_lookup.csv")        # model lookup
    scaler: Any = field(default=None)                                           # scaler to use with new preds

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
            stratify=self.data[self.target],
            test_size=test_prop,
            random_state=42
        )

        # make augmentations
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = post_split_pipeline(
            self.X_train, self.X_test, self.y_train, self.y_test, self.target, 
            upsample=self.upsample
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
        for hp, val in hyperparams.items():
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
                "model-type": ["XGBoost"],
                **{k: [v] for k, v in mean_scores.items() if k[0] not in ["l", "s"]},
                "path": [report_name]
            })
        ])
        model_reports = pd.concat([model_reports, *add_rows], ignore_index=True)
        model_reports.drop_duplicates(inplace=True)
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
        model_reports = model_reports[model_reports["model-type"] == "XGBoost"]

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
            model_reports.sort_values(by=priority_list, inplace=True, ascending=False, ignore_index=True)
            path = model_reports["path"][0]
        
        # load model
        if path is None:
            return False
        self.model = pickle.load(open(f"../models/weights/{path}.pickle", "rb"))
        self.hyperparams = json.load(open(f"../models/hyperparams/{path}.json", "r"))

        return True


    def prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
            Applies the necessary transformations to the dataset so we can 
            propagate a prediction through.

            @param df: observation(s) to transform

            @returns object ready for predict method
        """

        # normalize
        X = self.scaler.transform(X.values)

        # DON'T CONVERT
        return X


    def train_model(self, verbose: int=2, **kwargs) -> None:
        """
            Trains the model, assuming no hyperparameter optimization.
        """

        # fit model
        self.model = self.model.fit(self.X_train.values, self.y_train.values)

        # plotting
        if verbose > 1:
            # convert tracker to df
            train_loss = self.model.train_score_

            df = pd.DataFrame({
                "train": train_loss,
            })
            df.index = df.index + 1

            # line plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x=df.index, y="train", color="darkred", marker="*", label="train")
            plt.xlabel("Epoch")
            plt.ylabel(f"{self.hyperparams['loss'].capitalize()} Loss")
            plt.legend()
            plt.ylim(bottom=0)
            plt.title("Loss vs Epochs")
            plt.show()


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
        print(f"Macro-F1: {np.mean(f):.4f}")

        # Predicting probabilies for ROC
        y_pred_proba = self.predict_proba(self.X_test.values)

        # Calculate ROC curve and AUC for each class
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_test == label, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve for {label} (area = {roc_auc:0.2f})')

        # Plot ROC curve
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

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
        return self.model.predict(self.prepare_data(X))
    
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
            Generates prediction probabilities for the test data.
            
            @param X: data to predict on
        """

        # wrap predictions
        return self.model.predict_proba(X)


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
    def __init__(self, input_size, hidden_size, output_size, num_hidden, 
                 num_epochs, learning_rate, batch_size, classify_fn="sigmoid", dropout_rate=0.3):
        """
            Initialize model based on hyperparams. This is a normal FFNN with 
            ReLU
        """
        
        # enforce args
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_hidden
        if isinstance(dropout_rate, float):
            dropout_rate = [dropout_rate] * (num_hidden - 1)

        # setup model arch
        super(LinearNN, self).__init__()

        # input
        self.fc_input = nn.Linear(input_size, hidden_size[0])

        # hidden + dropout
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(num_hidden - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_size[i], hidden_size[i + 1])
            )
            self.hidden_layers.append(
                nn.BatchNorm1d(hidden_size[i + 1])
            )
            self.dropout_layers.append(
                nn.Dropout(dropout_rate[i])
            )
        
        # output + output functions
        self.fc_output = nn.Linear(hidden_size[-1], output_size)

        self.relu = nn.functional.relu
        self.classify_fn = nn.Sigmoid() if classify_fn == "sigmoid" else nn.Softmax(dim=1)

        # setup params
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def forward(self, x):
        """
            Propagate information through network.
        """

        # ensure dtypes
        x = x.to(torch.float32)

        # push forward
        out = self.fc_input(x)
        out = self.relu(out)

        # apply hidden & dropout
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            out = self.relu(hidden_layer(out))
            out = dropout_layer(out)

        out = self.fc_output(out)
        # out = self.classify_fn(out)       # avoid using this with CE loss
        return out


    def fit(self, X, y, device) -> Any:
        """
            Wraps a fit method for use in the gridsearch functionality.

            @param X: train features
            @param y: train ground truths
            
            @returns reference to self
        """

        # model.train #
        # setup gradient descent
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            TabularDataset(X=X, y=y, device=device),
            batch_size=self.batch_size,
            shuffle=True
        )

        # fit model & track loss
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                # forward pass
                self.optimizer.zero_grad()
                outputs = self(inputs)

                # loss + backprop
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # track loss
                running_loss += loss.item()
        
        return self

    
    def predict(self, X, device) -> Any:
        # gen tensors
        X = torch.from_numpy(X).to(device)

        # predict without backprop
        self.eval()

        with torch.no_grad():
            # forward pass
            outputs = self.classify_fn(self(X))
            outputs = nn.functional.softmax(outputs, dim=1)

            # append predictions & the raw prediction value
            confidence, predictions = torch.max(outputs, 1)

        # wrap predictions
        return np.array(predictions.cpu()), np.array(confidence.cpu())


    def test(self, X, y, device) -> Any:
        # predict
        labels = list(sorted(y.unique()))
        y_pred, y_conf = self.predict(X, device)
        y_test = y

        # metrics + report
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

        return a


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
    upsample: bool = field(default=False)                                       # whether to correct imbalance or not
    loss_balance: bool = field(default=False)                                   # whether to account for imbalance in loss-calc

    # inferred members
    data: pd.DataFrame = field(default=None)                                    # dataset
    model: LinearNN = field(default=None)                                       # underlying model
    model_lookup_path: str = field(default="../models/model_lookup.csv")        # model lookup
    optimizer: Any = field(default=None)                                        # optimizer function
    loss_fn: Any = field(default=None)                                          # loss for neural network
    device: Any = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available else "cpu")
        # If you don't have CUDA Uncomment this line as well as the line of 785
        # default=torch.device("cpu")
    )                                                                           # device to use; tries for GPU optimization
    scheduler: Any = field(default=None)                                        # learning rate scheduler
    scaler: Any = field(default=None)                                           # scaler to use with new preds

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
            stratify=self.data[self.target],
            test_size=test_prop,
            random_state=42
        )

        # make augmentations
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = post_split_pipeline(
            self.X_train, self.X_test, self.y_train, self.y_test, self.target, 
            upsample=self.upsample
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
            "learning_rate": 0.01,
            "dropout_rate": 0.3,
            "classify_fn": "sigmoid"
        }

        # if no update is required
        if hyperparams is None:
            return

        # if optimization is required
        if optimize:
            self.optimize_hyperparams(with_cv=True)
            return
        
        # update
        for hp, val in hyperparams.items():
            # include hyperparameter even if not expected
            if hp not in self.hyperparams:
                print(f"<WARNING> unexpected hyperparameter {hp} found; included anyways with value {val}")
            self.hyperparams[hp] = val

        # update model
        self.model = LinearNN(**self.hyperparams).to(self.device)


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
                "model-type": ["FFNN"],
                **{k: [v] for k, v in mean_scores.items() if k[0] not in ["l", "s"]},
                "path": [report_name]
            })
        ])
        model_reports = pd.concat([model_reports, *add_rows], ignore_index=True)
        model_reports.drop_duplicates(inplace=True)
        model_reports.sort_values(by=[
            "f1-score",
            "accuracy",
            "precision",
            "recall"
        ], inplace=True, ascending=False)
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
        model_reports = model_reports[model_reports["model-type"] == "FFNN"]

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
            model_reports.sort_values(by=priority_list, ascending=False, inplace=True, ignore_index=True)
            path = model_reports["path"][0]
            # model_reports.to_csv(self.model_lookup_path, index=False)
        
        # load model
        if path is None:
            return False
        
        self.set_hyperparams(
            json.load(open(f"../models/hyperparams/{path}.json", "r"))
        )
        self.model = LinearNN(**self.hyperparams).to(self.device)
        self.model.load_state_dict(torch.load(f"../models/weights/{path}.pt"))
        # Uncomment the code below if you do not have cuda enabled
        # self.model.load_state_dict(torch.load(f"../models/weights/{path}.pt", map_location=torch.device('cpu')))
        self.model.eval()

        return True


    def prepare_data(self, X: np.ndarray) -> pd.DataFrame:
        """
            Applies the necessary transformations to the dataset so we can 
            propagate a prediction through.

            @param df: observation(s) to transform

            @returns object ready for predict method
        """

        # normalize
        X = self.scaler.transform(X)

        # DON'T CONVERT
        return X


    def train_model(self, verbose: int=0) -> None:
        """
            Trains the model, assuming no hyperparameter optimization.
        """

        # setup gradient descent
        self.optimizer = Adam(self.model.parameters(), lr=self.model.learning_rate)
        if self.scheduler is not None:
            self.scheduler = {
                "reduce": torch.optim.lr_scheduler.ReduceLROnPlateau
            }[self.scheduler](self.optimizer)
        
        if self.loss_balance:
            loss_weights = dict(self.y_train.value_counts())
            loss_weights = [loss_weights[i] for i in range(self.y_test.max() + 1)]
            total_obs = sum(loss_weights)
            loss_weights = [(total_obs - i) / total_obs for i in loss_weights]

            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(loss_weights).to(torch.float).to(self.device)
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        losses = {k: list() for k in ["train", "test"]}
        best_loss = float("inf")
        best_epoch = 0
        train_loader = DataLoader(
            TabularDataset(X=self.X_train, y=self.y_train, device=self.device),
            batch_size=self.model.batch_size,
            shuffle=True
        )

        # fit model & track loss
        for epoch in range(self.model.num_epochs):
            # setup
            running_loss = 0.0
            self.model.train()

            for inputs, labels in tqdm(train_loader, disable=(verbose < 1)):
                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # loss + backprop
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # track loss
                running_loss += loss.item()

            # validation loss & loss tracking
            self.model.eval()
            with torch.no_grad():
                test_loss = self.loss_fn(
                    self.model(torch.from_numpy(self.X_test.to_numpy()).to(self.device)),
                    torch.from_numpy(self.y_test.to_numpy()).to(self.device)
                ).item()
            train_loss = running_loss / len(train_loader)

            losses["train"].append(train_loss)
            losses["test"].append(test_loss)

            # early stopping
            avg_loss = ((test_loss + train_loss) / 2)
            if avg_loss < best_loss:
                best_loss = ((test_loss + train_loss) / 2)
                best_model_weights = copy.deepcopy(self.model.state_dict())  
                best_epoch = epoch    
                patience = 10
            else:
                # allow for some leeway
                patience -= 1

                # update model with best weights
                if patience == 0:
                    self.model.load_state_dict(best_model_weights)
                    break

            # print stats if wanted
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{self.model.num_epochs}, Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # visualization
        if verbose > 1:
            # convert tracker to df
            df = pd.DataFrame(losses)
            df.index = df.index + 1

            # line plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x=df.index, y="train", color="darkred", marker="*", label="train")
            sns.lineplot(data=df, x=df.index, y="test", color="darkblue", marker="x", label="test")
            plt.axvline(x=best_epoch + 1, color="darkred", label="chosen-model")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.ylim(bottom=0)
            plt.title("Loss vs Epochs")
            plt.show()


    def test_model(self, save: bool=True) -> None:
        """
            Generates test error.
        """

        # predict
        y_pred, y_conf = self.predict(self.X_test)
        y_test = self.y_test

        # metrics + report
        labels = list(sorted(self.data[self.target].unique()))
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
        print(f"Macro-F1: {np.mean(f):.4f}")

        # Predicting probabilies for ROC
        y_pred_proba = self.predict_proba(self.X_test.values)
        
        # Calculate ROC curve and AUC for each class
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_test == label, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve for {label} (area = {roc_auc:0.2f})')

        # Plot ROC curve
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

        # export weights
        self.score = [{"label": label, "precision": p[label], "recall": r[label], \
                       "f1-score": f[label], "support": s[label], "accuracy": a * 100} \
                        for label in labels]
        
        if save:
            self.save_model()


    def predict(self, X: np.ndarray) -> tuple[np.array, np.array]:
        """
            Generates predictions for use in the test data.

            @param X: data to predict on
        """

        # wrap
        preds, conf = self.model.predict(self.prepare_data(X), self.device)
        return preds, conf
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
            Generates prediction probabilities for the test data.
            No builtin function from scikit learn to wrap, so doing this manually.
            
            @param X: data to predict on
        """
        # Convert to tensor
        X = torch.from_numpy(X).to(self.device)

        # predict without backprop
        self.model.eval()
        with torch.no_grad():
            # forward pass
            outputs = self.model.classify_fn(self.model(X))

        return outputs.cpu().numpy()


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
                "learning_rate": [[0.001], [0.001], [0.001], [0.0005], [0.0005, 0.0001], [0.0005, 0.0001]],
                "input_size": [self.X_train.shape[1]],
                "output_size": [self.y_train.nunique()],
                "hidden_size": [64, 128, 256, 512, 1024, 2048],
                "num_hidden": [16, 8, 4, 4, 2, 2],
                "num_epochs": [25, 25, 25, 25, 25, 25],
                "dropout_rate": [[0.25, 0.5] * 6],
                "batch_size": [[128, 256, 512] * len(6)]
            }
            # grid_search = {
            #     "learning_rate": [[0.01, 0.001], [0.01, 0.001], [0.001, 0.0005], [0.001, 0.0005], [0.001, 0.0005], [0.001, 0.0005]],
            #     "input_size": [self.X_train.shape[1]],
            #     "output_size": [self.y_train.nunique()],
            #     "hidden_size": [64, 128, 256, 512, 1024, 2048],
            #     "num_hidden": [16, 8, 4, 4, 2, 2],
            #     "num_epochs": [25, 25, 25, 25, 25, 25],
            #     "batch_size": [[32, 64, 128] for _ in range(6)]
            # }
        
        # conduct search
        print("<Grid-Search>")
                ################################################################
                # DEPRECATED :: may work if we can find a wrapper for sklearn or 
                #               fix SKORCH wrapper
                # searcher = GridSearchCV(
                #     NeuralNetClassifier(
                #         LinearNN,
                #         **self.hyperparams
                #     ),
                #     grid_search,
                #     scoring="accuracy",
                #     refit=True,
                #     cv=kfold,
                #     verbose=3,
                #     n_jobs=-1
                # ).fit(self.X_train, self.y_train)

                # searcher_df = pd.DataFrame.from_dict(searcher.cv_results_)
                # accuracy = searcher.best_score_
                # self.model = searcher.best_estimator_
                # self.hyperparams = searcher.best_params_
                ################################################################

        # trackers
        num_hidden = len(grid_search["hidden_size"])
        num_lr = [len(k) for k in grid_search["learning_rate"]]
        num_bs = [len(k) for k in grid_search["batch_size"]]
        num_dr = [len(k) for k in grid_search["dropout_rate"]]
        num_combos = sum(lr * bs for lr, bs in zip(num_lr, num_bs))
        print(f"Testing {num_combos} combinations WITHOUT cross-validation")

        max_perf = -1
        tracker_df: list[pd.DataFrame] = []

        # iterate hidden sizes assuming mirrored options
        for i, hidden_size in enumerate(grid_search["hidden_size"]):
            # setup
            input_size = grid_search["input_size"][0]
            output_size = grid_search["output_size"][0]
            num_hidden = grid_search["num_hidden"][i]
            num_epochs = grid_search["num_epochs"][i]

            # for every hidden size, we'll check learning rates
            for lr in grid_search["learning_rate"][i]:
                # for every combo of lr and hidden size, we'll check batch size
                for bs in grid_search["batch_size"][i]:
                    # for every combo of lr, hs, bs, and dropout rate
                    for dr in grid_search["dropout_rate"][i]:
                        # try combo
                        hyperparam_combo = {
                            "learning_rate": lr,
                            "input_size": input_size,
                            "output_size": output_size,
                            "hidden_size": hidden_size,
                            "num_hidden": num_hidden,
                            "num_epochs": num_epochs,
                            "batch_size": bs,
                            "dropout_rate": dr,
                            "classify_fn": "sigmoid"
                        }

                        # train & test
                        print(f"\n\n<Trying Model Architecture> {hidden_size=}, {lr=}, {bs=}, {dr=}, {num_hidden=}, {num_epochs=}. . . ", end="")
                        cur_model = LinearNN(**hyperparam_combo).to(self.device)
                        cur_model = cur_model.fit(self.X_train, self.y_train, self.device)
                        cur_perf = cur_model.test(self.X_test, self.y_test, self.device)
                        print(f"perf: {cur_perf:.4f}")

                        # tracker report
                        report_df = {
                            "model-type": "ffnn",
                            "accuracy": cur_perf,
                            **hyperparam_combo
                        }
                        report_df = pd.DataFrame({k: [v] for k, v in report_df.items()})
                        tracker_df.append(report_df)

                        # keep best
                        if max_perf < cur_perf:
                            max_perf = cur_perf
                            self.hyperparams = hyperparam_combo
                            self.model = cur_model            

        print(json.dumps(self.hyperparams, indent=4))

        # export params, weights
        if not os.path.exists("../models/grid-searches/"):
            os.makedirs("../models/grid-searches/")
        
        tracker_df = pd.concat(tracker_df, ignore_index=True)
        tracker_df.to_csv(f"../models/grid-searches/ffnn-classifier-{datetime.datetime.now().strftime('%m-%d-%y-%H')}.csv", index=False)
        
        self.test_model()
        self.save_model()


    def explain_model(self) -> dict[str, float]:
        """
            Returns a dictionary of feature importance based on some model 
            explainer's output.
        """

        # explain via shap
        raise NotImplementedError("can't explain model yet...")
        # explainer = shap.KernelExplainer(self.model.predict, self.X_train.iloc[:50, :])
        # shap_values = explainer.shap_values(self.X_train.iloc[299, :], nsamples=500)
        # shap.force_plot(explainer.expected_value, shap_values, self.X_test.iloc[299, :])


@dataclass(slots=True)
class LogClassifier:
    # user-set members
    target: str = field()                                                       # target feature
    path: str = field()                                                         # dataset path
    upsample: bool = field(default=False)                                       # whether to correct imbalance or not
    hyperparams: dict[str, int | float | str] = field(default_factory=dict)     # hyperparams; optional parameters

    # inferred members
    data: pd.DataFrame = field(default=None)                                    # dataset
    model: LogisticRegression = field(default=None)                             # underlying model
    model_lookup_path: str = field(default="../models/model_lookup.csv")        # model lookup
    scaler: Any = field(default=None)                                           # scaler to use with new preds

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
            stratify=self.data[self.target],
            test_size=test_prop,
            random_state=42
        )

        # make augmentations
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = post_split_pipeline(
            self.X_train, self.X_test, self.y_train, self.y_test, self.target, 
            upsample=self.upsample
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
        if len(self.hyperparams) == 0:
            self.hyperparams["max_iter"] = 100000
            self.hyperparams["random_state"] = 42
            self.hyperparams["penalty"] = "elasticnet"
            self.hyperparams["l1_ratio"] = 0.5
            self.hyperparams["solver"] = "saga"
            self.hyperparams["verbose"] = 1
        self.model = LogisticRegression(**self.hyperparams)


    # External Methods
    def save_model(self) -> None:
        """
            Saves model weights and hyperparams for future reloading.
        """

        # generate save path
        mean_scores = ({
            k: sum(d[k] for d in self.score) / len(self.score) 
            for k in self.score[0]
        })
        report_name = "log-classifier-" + ("-".join([
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
                "model-type": ["Logistic"],
                **{k: [v] for k, v in mean_scores.items() if k[0] not in ["l", "s"]},
                "path": [report_name]
            })
        ])
        model_reports = pd.concat([model_reports, *add_rows], ignore_index=True)
        model_reports.drop_duplicates(inplace=True)
        model_reports.sort_values(by=[
            "f1-score",
            "accuracy",
            "precision",
            "recall"
        ], inplace=True, ascending=False)
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
        model_reports = model_reports[model_reports["model-type"] == "Logistic"]

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
            model_reports.sort_values(by=priority_list, ascending=False, inplace=True, ignore_index=True)
            path = model_reports["path"][0]
            # model_reports.to_csv(self.model_lookup_path, index=False)
        
        # load model
        if path is None:
            return False
        
        self.model = pickle.load(open(f"../models/weights/{path}.pickle", "rb"))
        self.hyperparams = json.load(open(f"../models/hyperparams/{path}.json", "r"))
        return True


    def prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
            Applies the necessary transformations to the dataset so we can 
            propagate a prediction through.

            @param df: observation(s) to transform

            @returns object ready for predict method
        """

        # normalize
        X = self.scaler.transform(X.values)

        # DON'T CONVERT
        return X


    def train_model(self, verbose: int=0) -> None:
        """
            Trains the model using logistic regression
        """
        
        # train
        # sort_index = np.argsort(self.y_train)
        # self.y_train = self.y_train[sort_index]
        # self.X_train = self.X_train[sort_index]
        
        self.model.fit(self.X_train, self.y_train)
    

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
        print(f"Macro-F1: {np.mean(f):.4f}")

        # Predicting probabilies for ROC
        y_pred_proba = self.predict_proba(self.X_test.values)
        
        # Calculate ROC curve and AUC for each class
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_test == label, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve for {label} (area = {roc_auc:0.2f})')

        # Plot ROC curve
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

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

        # wrap
        X = self.prepare_data(X)
        raw_preds = self.model.predict_proba(X)
        preds = np.argmax(raw_preds, axis=1)
        confs = raw_preds[np.arange(preds.shape[0]), preds]

        return preds, confs


    def explain_model(self, **kwargs) -> dict[str, float]:
        """
            Returns a dictionary of feature importance based on some model 
            explainer's output.

            We'll only take the diabetes coefs since they mirror (balance) the 
            positive predictions.
        """

        # unpack coefs
        coefs = list(self.model.coef_[0])
        feature_names = list(self.model.feature_names_in_)

        # print report
        importance = dict(zip(feature_names, coefs))
        # print(json.dumps(importance, indent=4))
        return importance


    def patient_analysis(self, user_info: dict[str, int | float], **kwargs) -> str:
        """
            Generates a feature importance lookup and gauges what contributes 
            against the patient's risk of diabetes and what helps them.
        """

        # generate importance 
        importance = self.explain_model()
        importance_df = pd.DataFrame({
            "feature": importance.keys(),
            "weight": importance.values()
        })
        user_info_df = pd.DataFrame({
            "feature": user_info.keys(),
            "weight": [v[0] for v in user_info.values()]
        })

        # ensure the user info is scaled before explaining
        user_info_df["weight"] = self.scaler.transform(user_info_df["weight"].values.reshape(1, -1)).flatten()

        # create a lookup of importance based on feature weight and user info
        importance_df.sort_values(by="feature", ascending=False)
        user_info_df.sort_values(by="feature", ascending=False)
        print(importance_df)
        print(user_info_df)
        importance_df["weight"] = importance_df["weight"].astype(float) * user_info_df["weight"]
        importance_df.sort_values(by="weight", ascending=True)

        print(importance_df)

        # risk analysis: setup trackers
        categories = ["most-harmful", "harmful", "irrelevant", "helpful", "most-helpful"]
        thresholds = dict(zip(categories, [-0.3, -0.05, 0.05, 0.05, 0.3]))
        grouped_features = dict.fromkeys(categories)
        report = dict(zip(categories, ["" for _ in categories]))

        # group features into categories
        grouped_features["most-harmful"] =  set(importance_df[importance_df["weight"] < thresholds["most-harmful"]]["feature"])
        grouped_features["harmful"]      =  set(importance_df[importance_df["weight"] <= thresholds["harmful"]]["feature"])
        grouped_features["irrelevant"]   =  set(importance_df[abs(importance_df["weight"]) < thresholds["irrelevant"]]["feature"])
        grouped_features["helpful"]      =  set(importance_df[importance_df["weight"] >= thresholds["helpful"]]["feature"])
        grouped_features["most-helpful"] =  set(importance_df[importance_df["weight"] > thresholds["most-helpful"]]["feature"])

        # generate report
        for row in importance_df.itertuples(index=False):
            # unpack
            feature = row[0]
            weight = row[1]

            # match with category
            if feature in grouped_features["most-harmful"]:
                report["most-harmful"] += f" <-> {feature}: \tincreases risk of pre-diabetes/diabetes by {-weight * 100:.2f}%; if this is above 100, this is not a good sign :(\n"
            elif feature in grouped_features["harmful"]:
                report["harmful"] += f" - {feature}: \tincreases risk of pre-diabetes/diabetes by {-weight * 100:.2f}%\n"
            elif feature in grouped_features["irrelevant"]:
                report["irrelevant"] += f" * {feature}: \tdoesn't really apply to you in this context; this doesn't imply this behavior doesn't matter, just that for your specific health information, {feature} neither helps nor hurts you\n"
            elif feature in grouped_features["helpful"]:
                report["helpful"] += f" + {feature}: \treduces risk of pre-diebetes/diabetes by {weight * 100:.2f}%\n"
            elif feature in grouped_features["most-helpful"]:
                report["most-helpful"] += f" <+> {feature}: \treduces risk of pre-diebetes/diabetes by {weight * 100:.2f}%; if this is above 100, good job!\n"

        # check empty
        report = {k: v if v != "" else "Oops... it seems like you don't have much in this category for us to analyze. This is either a really good sign :), or a really bad one :(\n"\
                  for k, v in report.items()}

        # full written analysis
        final_report = (
f"""
According to our analysis (an linear approximation of our deep learning model), we've generated the following insights:
            
** Harmful Behaviors **
{report['most-harmful']}

We also noted that the following increase your risk for pre-diabetes/diabetes, but to a lesser degree than the previous:
{report['harmful']}


** Helpful Behaviors **
We haven't forgotten that you've of course done some things right:
{report['most-helpful']}

{report['helpful']}


** Irrelevant **
The following behaviors are irrelevant to you since you either don't participate in them, or we've gauged that it doesn't really matter for you in this context:
{report['irrelevant']}
"""
        )

        # export
        return final_report, importance
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
            Generates prediction probabilities for the test data.
            
            @param X: data to predict on
        """

        # wrap predictions
        return self.model.predict_proba(X)

