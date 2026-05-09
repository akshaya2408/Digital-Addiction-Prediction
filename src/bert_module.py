import torch
from transformers import AutoTokenizer, AutoModel


class DistilBERTEmbedder:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.model.eval()

    def encode_texts(self, texts, batch_size=32, max_length=64):
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling
            last_hidden = outputs.last_hidden_state
            batch_embeddings = last_hidden.mean(dim=1)

            embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)