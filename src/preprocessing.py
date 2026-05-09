import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Load Data
def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Loaded")
    print(df.shape)
    print(df.head())
    return df


# 2. Basic Cleaning
def clean_data(df):

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    return df


# 3. Encode Categorical Variables
def encode_data(df):

    le = LabelEncoder()
    df["AddictionLevel"] = le.fit_transform(df["AddictionLevel"].astype(str))

    print("AddictionLevel mapping:")
    for original, encoded in zip(le.classes_, le.transform(le.classes_)):
        print(f"{original} -> {encoded}")

    categorical_cols = ['Gender', 'Occupation', 'AddictionLevel']

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


# 4. Feature Engineering (CORE PART)
def feature_engineering(df):

    # Touch Frequency (approximation)
    df['TouchFrequency'] = df['Unlocks'] * 3

    # Session Interval
    df['SessionInterval'] = df['ScreenTime'] / (df['Unlocks'] + 1)

    # Night Usage Ratio
    df['NightUsageRatio'] = df['LateNightUsage'] / (df['ScreenTime'] + 1)

    # Engagement Score (NEW - advanced feature)
    df['EngagementScore'] = (
        df['SocialMedia'] +
        df['Gaming'] +
        df['Streaming']
    )

    return df


# 5. Normalize Data
def normalize(df):
    target_col = "AddictionLevel"

    feature_cols = [col for col in df.columns if col != target_col]
    numeric_feature_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    df[numeric_feature_cols] = scaler.fit_transform(df[numeric_feature_cols])

    return df


# 6. Save Processed Data
def save_data(df):
    df.to_csv("../data/processed/processed_dataset.csv", index=False)
    print("Processed dataset saved!")


# MAIN PIPELINE
def main():

    df = load_data("../data/raw/gadget_addiction_dataset.csv")

    df = clean_data(df)

    df = encode_data(df)

    df = feature_engineering(df)

    df = normalize(df)

    save_data(df)


if __name__ == "__main__":
    main()