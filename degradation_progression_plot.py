import pandas as pd
import matplotlib.pyplot as plt

csv_path = "E:/MS data/data/degradation_all.csv"
df = pd.read_csv(csv_path)

df_pivot = df.pivot(index="Image", columns="Stage", values="Degradation_Percentage")

plt.figure(figsize=(10, 5))
for img in df_pivot.index:
    plt.plot(df_pivot.columns, df_pivot.loc[img], marker='o', linestyle='-', label=img)

plt.xlabel("Time Stage")
plt.ylabel("Degradation Percentage (%)")
plt.title("Degradation Progression Over Time")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid()
plt.show()
