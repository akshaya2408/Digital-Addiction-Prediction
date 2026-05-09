import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


from sequence import create_sequences_from_df
from lstm_model import LSTMModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Loading processed dataset...")
    df = pd.read_csv("../data/processed/processed_dataset.csv")

    SEQ_LEN = 10
    X, y = create_sequences_from_df(df, seq_length=SEQ_LEN, target_col="AddictionLevel")
    y = y.astype(int)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10

    print("Training LSTM...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1)

    acc = accuracy_score(y_test.cpu(), preds.cpu())
    print(f"\nLSTM Accuracy: {acc:.4f}\n")
    print(classification_report(y_test.cpu(), preds.cpu()))

    torch.save(model.state_dict(), "../models/lstm_model.pth")
    print("New LSTM model saved successfully.")


if __name__ == "__main__":
    main()