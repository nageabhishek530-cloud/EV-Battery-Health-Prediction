
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

battery_files = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]

all_data = []

for file in battery_files:
    path = os.path.join("dataset", "raw", file)
    mat = loadmat(path)

    battery_name = file.replace(".mat", "")
    cycles = mat[battery_name][0, 0]["cycle"]

    for i in range(cycles.shape[1]):
        cycle = cycles[0, i]

        if cycle["type"][0] == "discharge":
            battery_data = cycle["data"][0, 0]

            all_data.append({
                "Battery": battery_name,
                "Cycle": i + 1,
                "Capacity": float(battery_data["Capacity"][0, 0]),
                "Voltage": float(np.mean(battery_data["Voltage_measured"].flatten())),
                "Current": float(np.mean(battery_data["Current_measured"].flatten())),
                "Temperature": float(np.mean(battery_data["Temperature_measured"].flatten()))
            })

df = pd.DataFrame(all_data)

os.makedirs("dataset/processed", exist_ok=True)
df.to_csv("dataset/processed/battery_data.csv", index=False)

print("Dataset prepared successfully!")
print(df.head())
print("Total rows:", len(df))
