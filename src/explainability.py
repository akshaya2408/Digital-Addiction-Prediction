import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_SAVE_PATH = "../models/rf_explainer.pkl"


def main():
    print("Loading dataset...")
    df = pd.read_csv("../data/processed/semantic_dataset.csv")

    X = df.drop(columns=["BehaviorText", "AddictionLevel"])
    y = df["AddictionLevel"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Explainability model accuracy:", model.score(X_test, y_test))

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Saved explainability model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()