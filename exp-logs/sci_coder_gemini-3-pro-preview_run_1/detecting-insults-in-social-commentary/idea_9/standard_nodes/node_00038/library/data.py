import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger


class InsultDataset(Dataset):
    """
    Custom Dataset for Insult Detection.
    Handles text tokenization and structural features (SVD).
    Supports:
    1. Supervised Training (Hard Labels: 0/1)
    2. Distillation (Soft Labels: Probabilities)
    3. Inference (No Labels)
    """

    def __init__(self, texts, svd_features, labels=None, tokenizer=None, max_len=None):
        """
        Args:
            texts (list or np.array or pd.Series): Input text sequences.
            svd_features (np.array): Dense structural features (SVD).
            labels (list or np.array, optional): Targets. Can be binary (int) or soft (float).
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance.
            max_len (int): Maximum sequence length for tokenization.
        """
        self.texts = texts if isinstance(texts, (list, np.ndarray)) else texts.tolist()
        self.svd_features = svd_features
        self.labels = labels if labels is not None else []

        # Use provided tokenizer or load from Config
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        else:
            self.tokenizer = tokenizer

        self.max_len = max_len if max_len is not None else Config.max_len
        self.logger = get_logger()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Extract tensors (remove batch dimension added by tokenizer)
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        # Prepare SVD features
        svd_feats = torch.tensor(self.svd_features[idx], dtype=torch.float32)

        # Construct item dictionary
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "svd_features": svd_feats,
        }

        # Add target if available
        if len(self.labels) > 0:
            # Targets are always cast to float for BCEWithLogitsLoss
            target = torch.tensor(self.labels[idx], dtype=torch.float32)
            item["target"] = target

        return item


def get_tokenizer():
    """
    Helper function to load the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.model_name)
