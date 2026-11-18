





from scipy.io import loadmat

data = loadmat("dataset/raw/B0005.mat")
cycles = data["B0005"][0, 0]["cycle"]

for i in range(5):
    cycle = cycles[0, i]

    if cycle["type"][0] == "discharge":
        battery_data = cycle["data"][0, 0]
        print("Cycle:", i + 1)
        print("Capacity:", battery_data["Capacity"])
