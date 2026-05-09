import pandas as pd

def generate_behavior_text(row):
    parts = []

    if row['ScreenTime'] > 0.7:
        parts.append("high screen time")
    elif row['ScreenTime'] > 0.4:
        parts.append("moderate screen time")
    else:
        parts.append("low screen time")

    if row['AppSwitching'] > 0.7:
        parts.append("frequent app switching")
    elif row['AppSwitching'] > 0.4:
        parts.append("moderate app switching")
    else:
        parts.append("low app switching")

    if row['LateNightUsage'] > 0.6:
        parts.append("high late night usage")
    else:
        parts.append("controlled night usage")

    if row['SleepHours'] < 0.4:
        parts.append("poor sleep pattern")
    else:
        parts.append("healthy sleep pattern")

    if row['SocialMedia'] > 0.6:
        parts.append("heavy social media engagement")

    if row['Gaming'] > 0.6:
        parts.append("high gaming activity")

    if row['MoodScore'] < 0.4:
        parts.append("possible emotional imbalance")
    else:
        parts.append("stable mood pattern")

    sentence = "User shows " + ", ".join(parts) + "."
    return sentence


def main():
    df = pd.read_csv("../data/processed/processed_dataset.csv")

    df["BehaviorText"] = df.apply(generate_behavior_text, axis=1)

    df.to_csv("../data/processed/semantic_dataset.csv", index=False)
    print("Semantic dataset saved successfully.")
    print(df[["BehaviorText"]].head())


if __name__ == "__main__":
    main()