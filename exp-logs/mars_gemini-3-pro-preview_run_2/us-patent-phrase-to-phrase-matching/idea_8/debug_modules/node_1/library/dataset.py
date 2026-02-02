import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_cpc_texts
from library.features import generate_structural_features


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for the Phrase Similarity Task.
    Handles dynamic tokenization of text inputs and retrieval of pre-computed structural features.
    """

    def __init__(self, df, tokenizer, max_length=128, inference_only=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing processed data (text + features).
            tokenizer (PreTrainedTokenizer): Transformer tokenizer (e.g., DeBERTa).
            max_length (int): Maximum sequence length for tokenization.
            inference_only (bool): If True, does not look for or return labels.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference_only = inference_only

        # Pre-extract columns to numpy arrays for efficiency in __getitem__
        self.anchors = df["anchor"].astype(str).values
        self.targets = df["target"].astype(str).values
        self.contexts = df["context_text"].astype(str).values
        self.ids = df["id"].values

        # Define the specific structural feature columns to use
        self.feat_cols = [
            "levenshtein_dist",
            "levenshtein_norm",
            "jaccard_sim",
            "len_diff",
            "len_ratio",
            "word_len_diff",
        ]

        # Ensure features are float32 for Neural Networks
        self.features = df[self.feat_cols].fillna(0.0).values.astype(np.float32)

        if not self.inference_only:
            self.labels = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Construct input text format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # The tokenizer adds [CLS] at the start and [SEP] at the end automatically.
        # We manually insert the separator token between the segments.
        sep = self.tokenizer.sep_token
        text = f"{context} {sep} {anchor} {sep} {target}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Retrieve structural features
        struct_feats = torch.tensor(self.features[idx], dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "structural_features": struct_feats,
            "id": self.ids[idx],
        }

        if not self.inference_only:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def load_and_process_data(
    split="train", load_cached_data=True, debug=False, debug_size=100
):
    """
    Loads metadata, generates/loads structural features, enriches context,
    merges everything into a single DataFrame, and handles caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, processes a small subset of data.
        debug_size (int): Number of rows to use in debug mode.

    Returns:
        pd.DataFrame: The processed DataFrame ready for PhraseDataset.
    """

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Determine cache filename based on split and debug status
    debug_suffix = "_debug" if debug else ""
    cache_filename = f"{split}_processed{debug_suffix}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Logic Flow: Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to compute from scratch
            pass

    # 2. Logic Flow: Compute from scratch

    # Identify source file from Config
    if split == "train":
        file_path = Config.TRAIN_FILE
    elif split == "val":
        file_path = Config.VAL_FILE
    elif split == "test":
        file_path = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    # Load raw metadata
    df = pd.read_csv(file_path)

    # Handle Debug Mode
    if debug:
        df = df.head(debug_size).copy()

    # A. Generate Structural Features (utilizing library caching)
    # We use a specific cache name for the features to avoid collisions between debug/full
    feat_cache_name = f"{split}{debug_suffix}"
    struct_df = generate_structural_features(
        df, cache_name=feat_cache_name, load_cached_data=load_cached_data
    )

    # Merge structural features on 'id'
    if "id" in struct_df.columns:
        # Drop duplicates in feature df to be safe
        struct_df = struct_df.drop_duplicates(subset=["id"])
        df = df.merge(struct_df, on="id", how="left")

    # B. Enrich Context (utilizing library caching)
    cpc_texts = get_cpc_texts(load_cached_data=load_cached_data)

    # Map context codes (e.g., 'A47') to descriptions
    # 'context' column contains the code. We create 'context_text'.
    df["context_text"] = df["context"].map(cpc_texts).fillna("")

    # 3. Logic Flow: Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        # If saving fails, we still return the dataframe
        pass

    return df
