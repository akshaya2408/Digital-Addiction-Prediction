import pandas as pd

df = pd.read_csv("../data/processed/processed_dataset.csv")

print(df[[
    "TouchFrequency",
    "SessionInterval",
    "NightUsageRatio",
    "EngagementScore"
]].head())