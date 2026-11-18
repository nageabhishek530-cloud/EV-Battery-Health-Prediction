
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset/processed/battery_data_final.csv")

# Features
features = ["Cycle", "Voltage", "Current", "Temperature"]

# ---------------- SOH MODEL ----------------

X_soh = df[features]
y_soh = df["SOH"]

soh_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

soh_model.fit(X_soh, y_soh)

joblib.dump(soh_model, "soh_model.pkl")

# ---------------- RUL MODEL ----------------

rul_df = df.dropna(subset=["RUL"])

X_rul = rul_df[features]
y_rul = rul_df["RUL"]

rul_scaler = StandardScaler()
X_rul_scaled = rul_scaler.fit_transform(X_rul)

rul_model = SVR(kernel="rbf")
rul_model.fit(X_rul_scaled, y_rul)

joblib.dump(rul_model, "rul_model.pkl")
joblib.dump(rul_scaler, "rul_scaler.pkl")

print("SOH model saved successfully!")
print("RUL model and scaler saved successfully!")
