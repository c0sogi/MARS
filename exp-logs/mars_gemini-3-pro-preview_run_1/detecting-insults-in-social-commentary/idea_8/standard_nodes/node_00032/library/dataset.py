import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config


class InsultDataset(Dataset):
    """
    Dataset class for Insult Detection.
    Handles tokenization of text and integration of structural features.
    Supports both hard labels (int) and soft labels (float) for distillation.
    """

    def __init__(
        self, texts, struct_features, tokenizer, max_len=Config.max_len, labels=None
    ):
        """
        Args:
            texts (list or np.array): Raw text comments.
            struct_features (np.array): Pre-computed SVD features (dense).
            tokenizer (PreTrainedTokenizer): DeBERTa tokenizer.
            max_len (int): Maximum sequence length for tokenization.
            labels (list or np.array, optional): Target labels.
                                                 Can be binary (0/1) or soft probabilities.
        """
        self.texts = texts
        self.struct_features = struct_features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Ensure text is a string to handle potential NaNs or non-string inputs
        text = str(self.texts[idx])

        # Tokenize
        # return_tensors='pt' returns batch of size 1, so we squeeze(0) to get 1D tensors
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Structural features
        # Ensure float32 for neural network compatibility
        struct_feat = torch.tensor(self.struct_features[idx], dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "struct_features": struct_feat,
        }

        if self.labels is not None:
            # Convert label to float32 to support BCEWithLogitsLoss
            # This handles both hard labels (0 -> 0.0) and soft labels (0.85)
            # seamlessly for the distillation pipeline.
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            item["label"] = label

        return item
