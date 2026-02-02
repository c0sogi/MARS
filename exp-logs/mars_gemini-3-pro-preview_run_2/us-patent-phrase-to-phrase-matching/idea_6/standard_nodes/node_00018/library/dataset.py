import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from library.config import CFG
from library.utils import get_cpc_texts
from library.feature_engineering import get_features_batch


class PhraseDataset(Dataset):
    """
    Dataset class for the Adversarial Hybrid DeBERTa-v3-Large Ensemble.
    Handles text preprocessing, context enrichment, and structural feature integration.
    """

    def __init__(self, df, tokenizer, mode="train", cache_name=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'anchor', 'target', 'context', and 'score' (if train).
            tokenizer (PreTrainedTokenizer): Tokenizer for the model.
            mode (str): 'train', 'val', or 'test'.
            cache_name (str, optional): Unique identifier for caching structural features.
                                        If None, defaults to the value of `mode`.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = CFG.max_len

        # Determine cache name for structural features
        self.cache_name = cache_name if cache_name is not None else mode

        # ====================================================
        # 1. Text Preprocessing & Context Enrichment
        # ====================================================
        # Load CPC context descriptions
        cpc_texts = get_cpc_texts()

        # Map context codes to full text descriptions
        # Use fillna to handle any missing mappings gracefully
        self.contexts = self.df["context"].map(cpc_texts).fillna("").astype(str).values
        self.anchors = self.df["anchor"].astype(str).values
        self.targets = self.df["target"].astype(str).values

        # ====================================================
        # 2. Structural Features (Cached)
        # ====================================================
        # Compute or load structural features (Levenshtein, Jaccard, Length Ratio)
        # The get_features_batch function handles the caching logic (parquet)
        self.structural_features = get_features_batch(
            self.anchors,
            self.targets,
            cache_name=self.cache_name,
            load_cached_data=True,
        )

        # ====================================================
        # 3. Labels
        # ====================================================
        if self.mode != "test":
            self.labels = self.df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve raw text components
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Construct Input Text
        # Format: [CLS] Context Description [SEP] Anchor [SEP] Target [SEP]
        # We explicitly insert the separator token between components.
        # The tokenizer will automatically add the start [CLS] and end [SEP].
        sep = self.tokenizer.sep_token
        text = f"{context} {sep} {anchor} {sep} {target}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Extract tensors and remove the batch dimension added by tokenizer
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Retrieve structural features for this sample
        struct_feats = torch.tensor(self.structural_features[idx], dtype=torch.float32)

        # Construct return dictionary
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "structural_features": struct_feats,
        }

        # Add label if available
        if self.mode != "test":
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            item["label"] = label

        return item
