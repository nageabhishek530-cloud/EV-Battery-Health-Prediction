
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load saved models
soh_model = joblib.load("soh_model.pkl")
rul_model = joblib.load("rul_model.pkl")
rul_scaler = joblib.load("rul_scaler.pkl")

# Prediction history file
HISTORY_FILE = "prediction_history.csv"


def save_history(prediction_type, cycle, voltage, current, temperature, result):

    history_data = pd.DataFrame([{
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Type": prediction_type,
        "Cycle": cycle,
        "Voltage": voltage,
        "Current": current,
        "Temperature": temperature,
        "Prediction": result
    }])

    history_data.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not os.path.exists(HISTORY_FILE),
        index=False
    )


def get_history():

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        return history.fillna("").to_dict(orient="records")[::-1]

    return []


@app.route("/")
def home():
    return render_template(
        "index.html",
        soh_result=None,
        rul_result=None,
        history=get_history()
    )


@app.route("/predict_soh", methods=["POST"])
def predict_soh():

    cycle = float(request.form["cycle"])
    voltage = float(request.form["voltage"])
    current = float(request.form["current"])
    temperature = float(request.form["temperature"])

    data = pd.DataFrame(
        [[cycle, voltage, current, temperature]],
        columns=["Cycle", "Voltage", "Current", "Temperature"]
    )

    prediction = round(float(soh_model.predict(data)[0]), 2)

    save_history(
        "SOH",
        cycle,
        voltage,
        current,
        temperature,
        f"{prediction}%"
    )

    return render_template(
        "index.html",
        soh_result=prediction,
        rul_result=None,
        history=get_history()
    )


@app.route("/predict_rul", methods=["POST"])
def predict_rul():

    cycle = float(request.form["cycle"])
    voltage = float(request.form["voltage"])
    current = float(request.form["current"])
    temperature = float(request.form["temperature"])

    data = pd.DataFrame(
        [[cycle, voltage, current, temperature]],
        columns=["Cycle", "Voltage", "Current", "Temperature"]
    )

    scaled_data = rul_scaler.transform(data)
    prediction = max(0, round(float(rul_model.predict(scaled_data)[0])))

    save_history(
        "RUL",
        cycle,
        voltage,
        current,
        temperature,
        f"{prediction} cycles"
    )

    return render_template(
        "index.html",
        soh_result=None,
        rul_result=prediction,
        history=get_history()
    )


if __name__ == "__main__":
    app.run(debug=True)
