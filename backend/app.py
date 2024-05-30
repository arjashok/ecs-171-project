from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import inputvalidation as iv



test_data = {
  "high_bp": "1",
  "high_chol": "1",
  "chol_check": "1",
  "bmi": "25",
  "smoker": "0",
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
  "age": "30",
  "education": "high school graduate",
  "income": "45000"
}


app = Flask(__name__)
cors = CORS(app, resources={r"/model/*": {"origins": "http://localhost:3000"}})

@app.route("/model/predict", methods=["POST"])
@cross_origin(origin='http://localhost:3000')
def predict():
    data = request.json
    data, error = iv.input_validation(data)
    print(data)
    if(data == None):
        return jsonify({"error": error,
                        "message": "Invalid Input"})
    else:
        # Returns Prediction, Confidence, and Analysis
        return jsonify({"prediction": "dummy_prediction",
                        "confidence": "dummy_confidence",
                        "analysis": "dummy_analysis"})


if __name__ == "__main__":
    app.run(debug=True)
