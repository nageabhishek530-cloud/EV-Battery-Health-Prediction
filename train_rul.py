import pandas as pd

df = pd.read_csv("dataset/processed/battery_data_final.csv")

# Keep only rows where RUL is available
rul_df = df.dropna(subset=["RUL"])

print("RUL dataset:")
print(rul_df.head())

print("\nTotal RUL records:", len(rul_df))
print("\nRecords per battery:")
print(rul_df.groupby("Battery").size())

# Select input features
X = rul_df[["Cycle", "Voltage", "Current", "Temperature"]]

# Select target
y = rul_df["RUL"]

print("\nInput features:")
print(X.head())

print("\nTarget RUL:")
print(y.head()) 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100, random_state=42
    ),
    "SVR": SVR(kernel="rbf")
}

print("\nModels created successfully!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

batteries = ["B0005", "B0006"]

for test_battery in batteries:

    train_battery = "B0006" if test_battery == "B0005" else "B0005"

    train_mask = rul_df["Battery"] == train_battery
    test_mask = rul_df["Battery"] == test_battery

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print("\nTesting battery:", test_battery)
    print("Training battery:", train_battery)

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

        print("\n", name)
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))