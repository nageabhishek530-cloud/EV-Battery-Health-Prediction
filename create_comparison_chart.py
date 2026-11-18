
import matplotlib.pyplot as plt
import numpy as np
import os

algorithms = ["Random Forest", "XGBoost", "SVR"]

mae = [2.0913, 2.3130, 3.3803]
rmse = [2.6583, 2.9815, 4.4137]

x = np.arange(len(algorithms))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, mae, width, label="MAE")
plt.bar(x + width/2, rmse, width, label="RMSE")

plt.title("SOH Prediction - Algorithm Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Error (SOH percentage points)")
plt.xticks(x, algorithms)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

os.makedirs("static", exist_ok=True)
plt.savefig("static/model_comparison.png", dpi=150)
plt.close()

print("Model comparison chart created successfully!")
