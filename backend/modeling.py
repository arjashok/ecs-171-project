"""
    @brief Abstraction for calculating the information to present back to a user 
           submitting a form request.
    @author Arjun Ashok
"""


# Environment Setup
import pandas as pd
from typing import Any

import sys
sys.path.append('../notebooks/')
from utils.model import MLPClassifier, LogClassifier, TreeClassifier
from inputvalidation import input_validation


# Utility
def generate_analysis(user_data: dict[str, int | float]) -> tuple[str, dict]:
    """
        Generates an analysis of the user's highest risk features based on SHAP
        analysis.

        @param

        @returns description as a string, dictionary of feature: importance
    """

    # wrap report using a linear approximation (log model)
    clf = LogClassifier(
        target="diabetes",
        path="../datasets/pre_split_processed.parquet"
    )
    clf.load_model()
    
    return clf.patient_analysis(user_data)


# Wrapper
def generate_prediction(user_data: dict[str, str | float | int]=None) -> tuple[str, float, str]:
    """
        Wrapper for underlying implementation of the prediction pipeline; Given 
        the json dictionary, it'll transform the data as necessary and push 
        through the optimal model, returning the prediction, the model's 
        confidence, and eventually the analysis of the highest risk factors.

        @param user_data: dictionary of feature: value
        
        @returns tuple of prediction_label, confidence (float between 0 and 1), 
                 and analysis of user's risky behavior
    """

    # input validation
    if user_data is None:
        raise ValueError(f"Invalid pipeline prediction input! No data passed in bruh")
    if not isinstance(user_data, dict):
        raise ValueError(f"Invalid pipeline prediction input! Expected dict, got {type(user_data)} instead :(")

    # load best model
    model_args = {
        "target": "diabetes",
        "path": "../datasets/pre_split_processed.parquet"
    }
    model = MLPClassifier(**model_args)
    model.load_model()

    # data transformation
    if any(not isinstance(v, list) for k, v in user_data.items()):
        converted_dict =  dict()
        for k, v in user_data.items():
            converted_dict[k] = [v] if not isinstance(v, list) else v

        user_data = converted_dict

    user_df = pd.DataFrame(user_data)
    user_prepped_data = model.prepare_data(user_df)

    # prediction + confidence
    labels = [
        "Diabetes Free",
        "Pre-Diabetes",
        "Diabetes"
    ]
    prediction, confidence = model.predict(user_prepped_data)

    # generate analysis
    analysis, feature_importance = generate_analysis(user_data)

    # return all results
    return labels[prediction[0]], confidence[0], analysis
    

if __name__ == "__main__":
    test_data = [{
        "high_bp": "1",
        "high_chol": "1",
        "chol_check": "1",
        "bmi": "25",
        "smoker": "1",
        "stroke": "0",
        "heart_disease": "0",
        "physical_activity": "1",
        "fruits": "1",
        "veggies": "1",
        "heavy_drinker": "0",
        "healthcare": "1",
        "no_doc_bc_cost": "0",
        "general_health": "good",
        "mental_health": "5",
        "physical_health": "5",
        "diff_walk": "0",
        "sex": "1",
        "age": "50",
        "education": "high school graduate",
        "income": "45000"
    }]

    for td in test_data:
        td, _ = input_validation(td)
        result = generate_prediction(td)
        print(result)
