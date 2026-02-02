import torch
import numpy as np
import scipy.sparse
from torch.utils.data import Dataset
from library.config import Config


class WideDataset(Dataset):
    """
    Dataset for the Wide Model (Sparse Linear Layer).
    Handles sparse TF-IDF features and sparse Multi-Hot targets.
    Converts sparse rows to dense tensors on-the-fly to save memory.
    """

    def __init__(self, features, targets=None):
        """
        Args:
            features (scipy.sparse.csr_matrix): TF-IDF features.
            targets (scipy.sparse.csr_matrix, optional): Multi-hot encoded tags.
        """
        self.features = features
        self.targets = targets

        # Ensure features are in CSR format for efficient row slicing
        if not scipy.sparse.isspmatrix_csr(self.features):
            self.features = self.features.tocsr()

        if self.targets is not None and not scipy.sparse.isspmatrix_csr(self.targets):
            self.targets = self.targets.tocsr()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Extract feature row and convert to dense tensor
        # .toarray() returns shape (1, n_features), squeeze to (n_features,)
        x = self.features[idx].toarray().squeeze(0).astype(np.float32)
        x_tensor = torch.from_numpy(x)

        if self.targets is not None:
            # Extract target row and convert to dense tensor
            y = self.targets[idx].toarray().squeeze(0).astype(np.float32)
            y_tensor = torch.from_numpy(y)
            return (x_tensor,), y_tensor

        return (x_tensor,)


class DeepDataset(Dataset):
    """
    Dataset for the Deep Model (DistilRoBERTa).
    Handles on-the-fly tokenization of text and multi-hot encoding of tags.
    """

    def __init__(
        self, texts, tags=None, tokenizer=None, tag_encoder=None, max_len=Config.MAX_LEN
    ):
        """
        Args:
            texts (list or pd.Series): Input text (Title + Body).
            tags (list or pd.Series, optional): Space-delimited tag strings.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
            tag_encoder (TagEncoder, optional): Encoder to map tags to indices.
            max_len (int): Maximum sequence length for tokenization.
        """
        # Convert pandas Series to lists/numpy arrays for efficient indexing
        self.texts = texts.values if hasattr(texts, "values") else texts
        self.tags = tags.values if hasattr(tags, "values") else tags

        self.tokenizer = tokenizer
        self.tag_encoder = tag_encoder
        self.max_len = max_len

        # Determine number of classes for vector creation
        if self.tag_encoder is not None:
            self.num_classes = len(self.tag_encoder.classes_)
        else:
            self.num_classes = Config.NUM_TOP_TAGS

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer (1, seq_len) -> (seq_len,)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # If targets are present, encode them
        if self.tags is not None:
            tag_string = str(self.tags[idx])
            target = torch.zeros(self.num_classes, dtype=torch.float32)

            if self.tag_encoder:
                # Map tags to indices
                for t in tag_string.split():
                    if t in self.tag_encoder.tag_to_idx:
                        idx_label = self.tag_encoder.tag_to_idx[t]
                        target[idx_label] = 1.0

            return input_ids, attention_mask, target

        return input_ids, attention_mask
