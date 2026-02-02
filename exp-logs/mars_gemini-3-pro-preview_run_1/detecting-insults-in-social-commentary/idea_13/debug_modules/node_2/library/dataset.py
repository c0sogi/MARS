import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import Config


class InsultDataset(Dataset):
    """
    Dataset class for the Insult Detection task.
    Handles text tokenization and integration of structural SVD features
    for the Hybrid Transformer architecture.
    """

    def __init__(
        self, texts, svd_features, tokenizer, labels=None, max_len=Config.max_len
    ):
        """
        Args:
            texts (list or pd.Series): The text content to tokenize.
            svd_features (np.ndarray): Pre-computed SVD features corresponding to the texts.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use (DeBERTa or RoBERTa).
            labels (list or pd.Series, optional): The target labels (0 or 1). Defaults to None.
            max_len (int): Maximum sequence length for tokenization. Defaults to Config.max_len.
        """
        # Ensure texts are indexable by integer position (0 to len-1)
        if hasattr(texts, "tolist"):
            self.texts = texts.tolist()
        else:
            self.texts = list(texts)

        self.svd_features = svd_features
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Ensure labels are indexable by integer position
        if labels is not None:
            if hasattr(labels, "tolist"):
                self.labels = labels.tolist()
            else:
                self.labels = list(labels)
        else:
            self.labels = None

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.texts)

    def __getitem__(self, index):
        """
        Retrieves the sample at the given index.

        Returns:
            dict: A dictionary containing:
                - input_ids: Token indices.
                - attention_mask: Mask for padding.
                - svd_features: Dense vector from SVD.
                - labels: Target label (if available).
        """
        # Retrieve text
        text = str(self.texts[index])

        # Tokenize
        # We use encode_plus to handle padding, truncation, and attention masks
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove the batch dimension added by return_tensors="pt"
        # Shape: (max_len,)
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        # Retrieve SVD features and convert to float tensor
        # Shape: (svd_components,)
        svd_vec = torch.tensor(self.svd_features[index], dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "svd_features": svd_vec,
        }

        # Handle labels if they exist
        if self.labels is not None:
            # Target is binary (0/1). We use float32 for BCEWithLogitsLoss.
            label = torch.tensor(self.labels[index], dtype=torch.float32)
            item["labels"] = label

        return item
