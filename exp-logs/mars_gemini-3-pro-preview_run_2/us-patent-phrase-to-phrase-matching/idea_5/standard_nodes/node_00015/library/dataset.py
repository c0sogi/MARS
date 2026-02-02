import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.cpc_utils import map_cpc_to_text
from library.feature_engineering import get_all_structural_features


class PearsonDataset(Dataset):
    """
    PyTorch Dataset for Phrase Similarity Task.

    Features:
    - Tokenizes text in the format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    - Includes explicit structural features (Levenshtein, Jaccard, Length Ratio).
    - Maps float scores to integer class indices (0-4) for classification.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        structural_features: np.ndarray,
        tokenizer,
        max_length: int = 128,
        mode: str = "train",
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'context_text', 'anchor', 'target', and 'score' (if train/val).
            structural_features (np.ndarray): Matrix of structural features aligned with df.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.structural_features = structural_features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Pre-extract text columns to numpy arrays for faster access
        self.contexts = df["context_text"].fillna("").astype(str).values
        self.anchors = df["anchor"].fillna("").astype(str).values
        self.targets = df["target"].fillna("").astype(str).values

        # Prepare labels for training/validation
        if self.mode != "test":
            self.scores = df["score"].values
            # Map scores 0.0, 0.25, 0.5, 0.75, 1.0 to indices 0, 1, 2, 3, 4
            self.labels = (self.scores * 4).round().astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        anchor = self.anchors[idx]
        target = self.targets[idx]

        # Construct the input sequence.
        # We want: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # By passing context as 'text' and (anchor + sep + target) as 'text_pair',
        # the tokenizer automatically handles the special tokens structure.
        sep = self.tokenizer.sep_token
        text_pair = f"{anchor} {sep} {target}"

        inputs = self.tokenizer(
            context,
            text_pair,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return standard python lists
        )

        item = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "structural_features": torch.tensor(
                self.structural_features[idx], dtype=torch.float32
            ),
        }

        if self.mode != "test":
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def get_datasets(
    tokenizer,
    max_length: int = 128,
    load_cached_data: bool = True,
    debug: bool = False,
):
    """
    Loads metadata, computes/loads features, and returns PyTorch Datasets.

    Args:
        tokenizer: HuggingFace tokenizer.
        max_length (int): Max sequence length.
        load_cached_data (bool): Whether to use cached structural features.
        debug (bool): If True, truncates data for rapid debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # 1. Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # 2. Enrich Context (Map codes to descriptions)
    train_df = map_cpc_to_text(train_df)
    val_df = map_cpc_to_text(val_df)
    test_df = map_cpc_to_text(test_df)

    # 3. Get Structural Features (Handles caching internally)
    # These functions return DataFrames aligned with the metadata files
    full_train_feats, full_val_feats, full_test_feats = get_all_structural_features(
        load_cached_data=load_cached_data
    )

    # 4. Handle Debugging (Slice data)
    if debug:
        train_df = train_df.head(100).reset_index(drop=True)
        val_df = val_df.head(50).reset_index(drop=True)
        test_df = test_df.head(50).reset_index(drop=True)

        train_feats = full_train_feats.head(100).values
        val_feats = full_val_feats.head(50).values
        test_feats = full_test_feats.head(50).values
    else:
        train_feats = full_train_feats.values
        val_feats = full_val_feats.values
        test_feats = full_test_feats.values

    # 5. Create Datasets
    train_ds = PearsonDataset(
        train_df, train_feats, tokenizer, max_length=max_length, mode="train"
    )
    val_ds = PearsonDataset(
        val_df, val_feats, tokenizer, max_length=max_length, mode="val"
    )
    test_ds = PearsonDataset(
        test_df, test_feats, tokenizer, max_length=max_length, mode="test"
    )

    return train_ds, val_ds, test_ds
