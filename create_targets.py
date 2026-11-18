




import pandas as pd

df = pd.read_csv("dataset/processed/battery_data_with_soh.csv")

df["RUL"] = float("nan")

for battery, group in df.groupby("Battery"):
    group = group.reset_index()
    threshold = group["Capacity"].iloc[0] * 0.70

    reached = group[group["Capacity"] <= threshold]

    if not reached.empty:
        eol_position = reached.index[0]

        for position in range(eol_position + 1):
            original_index = group.loc[position, "index"]
            df.loc[original_index, "RUL"] = eol_position - position

df.to_csv("dataset/processed/battery_data_final.csv", index=False)

print(df.groupby("Battery")["RUL"].agg(["count", "min", "max"]))
print("\nRUL corrected successfully!")
