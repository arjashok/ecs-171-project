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


# Class
@dataclass
class Dataset:
    # members
    path: str = field()                                 # path to dataset
    target: str = field(default=None)                   # target variable (column name)
    feature_set: list[str] = field(init=False)          # defaults to all but target
    data: pd.DataFrame = field(default=None)            # dataset

    # internal methods
    def __post_init__(self):
        self.data = pd.read_csv(self.path, engine="c")


    # external methods
    def set_target(self, target: str) -> None:
        """
            Set the target feature, automatically generates feature_set.

            @param target: name of target feature
        """

        # set & fill feature set
        self.target = target
        self.feature_set = list(self.data.columns)
        self.feature_set.remove(self.target)
    



