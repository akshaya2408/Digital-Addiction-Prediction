import numpy as np



def create_sequences_from_df(df, seq_length=10, target_col="AddictionLevel"):
    X = []
    y = []

    feature_cols = [col for col in df.columns if col != target_col]

    feature_data = df[feature_cols].values
    target_data = df[target_col].values

    for i in range(len(df) - seq_length):
        X.append(feature_data[i:i + seq_length])
        y.append(target_data[i + seq_length - 1])

    return np.array(X), np.array(y)