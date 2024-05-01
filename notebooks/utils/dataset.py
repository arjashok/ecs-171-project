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
from dataclasses import dataclass, field
from utils.explore_dataset import *
from utils.transform_dataset import *


# Class
@dataclass
class Dataset:
    # user-set members
    path: str = field()                                                         # path to dataset
    target: str = field(default=None)                                           # target variable (column name)
    feature_set: list[str] = field(init=False)                                  # defaults to all but target

    # inferred members
    data: pd.DataFrame = field(default=None)                                    # dataset
    numeric_features: set[str] = field(default_factory=set)                     # numeric features
    ordinal_features: set[str] = field(default_factory=set)                     # ordinal features
    nominal_features: set[str] = field(default_factory=set)                     # nominal features

    # internal methods
    def __post_init__(self):
        self.data = pd.read_csv(self.path, engine="c")


    # external methods
    ## mutators
    def set_target(self, target: str) -> None:
        """
            Set the target feature, automatically generates feature_set.

            @param target: name of target feature
        """

        # set & fill feature set
        self.target = target
        self.feature_set = list(self.data.columns)
        self.feature_set.remove(self.target)
    

    def set_features(self, feature_set: list[str]) -> None:
        """
            Sets the features if manual specification is required. Note, 
            automatically verifies the target isn't contained within the feature 
            set.

            @param feature_set: iterable containing the narrowed features
        """

        # set & confirm
        if self.target in feature_set:
            print(f"<WARNING> attempting to include target variable in feature set. Try re-setting target first.")
            exit(1)
        
        self.feature_set = feature_set


    ## wrappers
    def infer_types(self, **kwargs) -> None:
        """
            Wraps call to explore dataset utility fn of the same name. Refer to 
            underlying function documentation for key-word args.
        """

        # wrap call
        self.numeric_features, self.categorical_features, _ = infer_types(
            self.data,
            **kwargs
        )

