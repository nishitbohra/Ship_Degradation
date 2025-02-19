import pandas as pd

csv_path = "E:/MS data/data/degradation_all.csv"
df = pd.read_csv(csv_path)

average_degradation = df.groupby("Stage")["Degradation_Percentage"].mean().reset_index()
most_degraded = df.loc[df["Degradation_Percentage"].idxmax()]

df_pivot = df.pivot(index="Image", columns="Stage", values="Degradation_Percentage")
df_pivot["Max_Diff"] = df_pivot.max(axis=1) - df_pivot.min(axis=1)
biggest_jump = df_pivot["Max_Diff"].idxmax()

report_text = f"""
hip Part Degradation Report 

1. Most Degraded Image: {most_degraded["Image"]} ({most_degraded["Stage"]})
2. Average Degradation Per Stage:
{average_degradation.to_string(index=False)}

Biggest Degradation Jump: {biggest_jump}

Conclusion:
- Degradation increases over time, with {most_degraded["Image"]} showing the highest degradation at {most_degraded["Degradation_Percentage"]:.2f}%.
- The biggest jump in degradation occurred for {biggest_jump}, indicating a critical wear phase.

Saving Report...
"""

report_path = "E:/MS data/data/degradation_report.txt"
with open(report_path, "w") as f:
    f.write(report_text)

print(report_text)
print(f"Report saved at: {report_path}")
