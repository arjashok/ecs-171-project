from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import inputvalidation as iv

app = Flask(__name__)
cors = CORS(app, resources={r"/model/*": {"origins": "http://localhost:3000"}})

@app.route("/model/predict", methods=["POST"])
@cross_origin(origin='http://localhost:3000')
def predict():
    data = request.json
    data = iv.input_validation(data)
    if(data == None):
        return jsonify({"error": "invalid input"})
    else:
        # Returns Prediction, Confidence, and Analysis
        return jsonify({"prediction": "dummy_prediction"})


if __name__ == "__main__":
    app.run(debug=True)