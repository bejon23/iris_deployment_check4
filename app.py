import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os  # For environment variables

# Create flask app
app = Flask(__name__)
model = pickle.load(open("svm_classifier.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The flower species is {}".format(prediction))

if __name__ == "__main__":
    # Get the port number from the environment variable PORT or use 4000 as fallback
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port)
