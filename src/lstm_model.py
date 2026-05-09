import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x, return_features=False):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        features = torch.relu(self.fc(out))

        if return_features:
            return features

        out = self.classifier(features)
        return out