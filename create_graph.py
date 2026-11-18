
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load battery dataset
df = pd.read_csv("dataset/processed/battery_data_final.csv")

# Create graph
plt.figure(figsize=(12, 6))

for battery, group in df.groupby("Battery"):
    group = group.sort_values("Cycle")

    plt.plot(
        group["Cycle"],
        group["Capacity"],
        marker="o",
        markersize=3,
        linewidth=2,
        label=battery
    )

plt.title("EV Battery Capacity Degradation", fontsize=16)
plt.xlabel("Discharge Cycle", fontsize=12)
plt.ylabel("Battery Capacity (Ah)", fontsize=12)

plt.legend(title="Battery Cells")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save graph
os.makedirs("static", exist_ok=True)
plt.savefig("static/capacity_graph.png", dpi=150)
plt.close()

print("Battery degradation graph created successfully!")
