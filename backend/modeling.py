"""
    @brief Abstraction for calculating the information to present back to a user 
           submitting a form request.
    @author Arjun Ashok
"""


# Environment Setup
from notebooks.utils.model import *
from notebooks.utils.wrappers import post_split_pipeline


# Utility
def generate_analysis(user_data: pd.DataFrame, model: Any) -> tuple[str, dict]:
    """
        Generates an analysis of the user's highest risk features based on SHAP
        analysis.

        @param

        @returns description as a string, dictionary of feature: importance
    """

    return "ur fat lol", dict()


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
    model = MLPClassifier(
        target="diabetes",
        path="../datasets/pre_split_processed.parquet"
    )
    model.load_model()

    # data transformation
    user_df = pd.DataFrame(user_data)
    user_prepped_data = model.prepare_data(user_df)

    # prediction + confidence
    labels = {
        "0": "Diabetes Free",
        "1": "Pre-Diabetes",
        "2": "Diabetes"
    }
    prediction, confidence = model.predict(user_prepped_data)

    # generate analysis
    analysis, feature_importance = generate_analysis(user_df, model)

    # return all results
    return labels[prediction[0]], confidence[0], analysis
    


