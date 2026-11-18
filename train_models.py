

import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("dataset/processed/battery_data_final.csv")

# Features and target
X = df[["Cycle", "Voltage", "Current", "Temperature"]]
y = df["SOH"]
groups = df["Battery"]

# Test one unseen battery at a time
logo = LeaveOneGroupOut()

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100, random_state=42
    ),
    "SVR": SVR(kernel="rbf")
}

results = {name: [] for name in models}

for train_index, test_index in logo.split(X, y, groups):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    test_battery = groups.iloc[test_index].iloc[0]

    print("\nTesting battery:", test_battery)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():

        if name == "SVR":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results[name].append((mae, rmse, r2))

        print(name)
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))

print("\n========== AVERAGE RESULTS ==========")

for name, scores in results.items():
    average = pd.DataFrame(
        scores, columns=["MAE", "RMSE", "R2"]
    ).mean()

    print("\n", name)
    print(average.round(4))
