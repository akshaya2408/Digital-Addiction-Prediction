import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sequence import create_sequences_from_df
from lstm_model import LSTMModel
from fusion_model import FusionModel
from bert_module import DistilBERTEmbedder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Loading semantic dataset...")
    df = pd.read_csv("../data/processed/semantic_dataset.csv")

    if "BehaviorText" not in df.columns:
        raise ValueError("BehaviorText column not found in semantic_dataset.csv")

    # Keep text separately
    texts = df["BehaviorText"].tolist()

    # Numeric dataframe for sequence creation
    df_numeric = df.drop(columns=["BehaviorText"]).copy()

    # Create sequences
    SEQ_LEN = 10
    X_seq, y = create_sequences_from_df(
        df_numeric,
        seq_length=SEQ_LEN,
        target_col="AddictionLevel"
    )

    # Align text with targets
    text_targets = texts[SEQ_LEN:]

    # Convert target to int
    y = y.astype(int)

    # Basic checks
    print("X_seq shape:", X_seq.shape)
    print("y shape:", y.shape)
    print("Text targets:", len(text_targets))

    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique.tolist(), counts.tolist()))
    print("Class distribution:", class_distribution)

    if len(X_seq) != len(text_targets):
        raise ValueError("Mismatch between sequence samples and text targets")

    # Safe train-test split
    if min(counts) < 2:
        print("Using split without stratify because a class has fewer than 2 samples.")
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
            X_seq, y, text_targets, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
            X_seq, y, text_targets, test_size=0.2, random_state=42, stratify=y
        )

    # Convert sequence data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    # Load trained LSTM model
    input_size = X_train.shape[2]
    lstm_model = LSTMModel(input_size=input_size).to(DEVICE)
    lstm_model.load_state_dict(torch.load("../models/lstm_model.pth", map_location=DEVICE))
    lstm_model.eval()

    print("Extracting LSTM features...")
    with torch.no_grad():
        train_lstm_features = lstm_model(X_train, return_features=True)
        test_lstm_features = lstm_model(X_test, return_features=True)

    print("Train LSTM feature shape:", train_lstm_features.shape)
    print("Test LSTM feature shape:", test_lstm_features.shape)

    # Generate DistilBERT embeddings
    print("Generating DistilBERT embeddings...")
    embedder = DistilBERTEmbedder()

    train_bert_embeddings = embedder.encode_texts(text_train).to(DEVICE)
    test_bert_embeddings = embedder.encode_texts(text_test).to(DEVICE)

    print("Train BERT embedding shape:", train_bert_embeddings.shape)
    print("Test BERT embedding shape:", test_bert_embeddings.shape)

    # Build fusion training dataset
    train_dataset = TensorDataset(
        train_lstm_features.detach(),
        train_bert_embeddings.detach(),
        y_train
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Fusion model
    fusion_model = FusionModel(
        lstm_feature_dim=train_lstm_features.shape[1],
        bert_dim=train_bert_embeddings.shape[1],
        num_classes=len(np.unique(y))
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

    # Training
    EPOCHS = 10
    print("Training fusion model...")

    for epoch in range(EPOCHS):
        fusion_model.train()
        total_loss = 0.0

        for lstm_f, bert_f, labels in train_loader:
            optimizer.zero_grad()

            outputs = fusion_model(lstm_f, bert_f)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Evaluation
    print("\nEvaluating fusion model...")
    fusion_model.eval()

    with torch.no_grad():
        outputs = fusion_model(test_lstm_features, test_bert_embeddings)
        preds = torch.argmax(outputs, dim=1)

    acc = accuracy_score(y_test.cpu(), preds.cpu())
    print(f"\nFusion Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test.cpu(), preds.cpu(), zero_division=0))

    # Save model
    torch.save(fusion_model.state_dict(), "../models/fusion_model.pth")
    print("Fusion model saved successfully.")


if __name__ == "__main__":
    main()