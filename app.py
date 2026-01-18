from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/id3_climate_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    uncertainty = float(request.form["uncertainty"])

    prediction = model.predict([[uncertainty]])[0]

    result = "High Temperature" if prediction == 1 else "Low Temperature"

    return render_template(
        "index.html",
        prediction_text=f"Predicted Climate Condition: {result}"
    )

if __name__ == "__main__":
    app.run(debug=True)
