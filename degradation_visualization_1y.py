import pandas as pd
import matplotlib.pyplot as plt

csv_path = "E:/MS data/degradation_1y.csv"
df = pd.read_csv(csv_path)

df = df.sort_values(by="Image")

plt.figure(figsize=(10, 5))
plt.bar(df["Image"], df["Degradation_Percentage"], color="tomato", alpha=0.7)
plt.xticks(rotation=90)  
plt.xlabel("Image")
plt.ylabel("Degradation Percentage (%)")
plt.title("Degradation Comparison (1 Year vs Newly Painted)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
