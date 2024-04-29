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
    target: str = field(init=False)                     # target variable (column name)
    feature_set: list[str] = field(init=False)          # defaults to all but target
    data: pd.DataFrame = field(default=None)            # dataset

    # internal methods


    # external methods
    



