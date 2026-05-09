import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, lstm_feature_dim=32, bert_dim=768, num_classes=3):
        super(FusionModel, self).__init__()

        self.fc1 = nn.Linear(lstm_feature_dim + bert_dim, 256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, lstm_features, bert_embeddings):
        x = torch.cat([lstm_features, bert_embeddings], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x