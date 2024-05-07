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
    model: 

    # internal methods
    def __post_init__(self):
        self.data = pd.read_csv(self.path, engine="c")


    # external methods
    ## mutators

    ## utility

