import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config


def get_identity_keywords():
    """
    Returns a list of keywords associated with the identity columns
    for the purpose of stochastic identity masking.
    """
    # Mapping identity columns to specific terms and common variations
    # to capture mentions effectively for masking.
    term_map = {
        "male": ["male", "man", "men", "boy", "boys"],
        "female": ["female", "woman", "women", "girl", "girls"],
        "homosexual_gay_or_lesbian": ["homosexual", "gay", "lesbian"],
        "christian": ["christian"],
        "jewish": ["jewish", "jew"],
        "muslim": ["muslim", "islam", "islamic"],
        "black": ["black"],
        "white": ["white"],
        "psychiatric_or_mental_illness": ["psychiatric", "mental", "illness"],
    }

    keywords = set()
    for col in Config.IDENTITY_COLUMNS:
        if col in term_map:
            keywords.update(term_map[col])
        else:
            keywords.add(col)

    return list(keywords)


def calculate_sample_weights(df):
    """
    Assigns weights to training samples to prioritize hard negatives.

    Logic:
    - Default weight: Config.DEFAULT_WEIGHT
    - Hard Negatives (Non-toxic + Identity Mention): Config.IDENTITY_WEIGHT_MULTIPLIER
    """
    # Initialize weights with default value
    weights = np.ones(len(df)) * Config.DEFAULT_WEIGHT

    # Ensure necessary columns exist
    if Config.TARGET_COL not in df.columns:
        return weights

    # Identify Toxic (Positive) vs Non-Toxic (Negative)
    # Competition threshold is 0.5
    is_toxic = df[Config.TARGET_COL] >= 0.5

    # Identify Identity Mentions
    # Check if any identity column has a value >= 0.5
    id_cols = [c for c in Config.IDENTITY_COLUMNS if c in df.columns]

    if id_cols:
        # Check if any identity is mentioned (value >= 0.5)
        # Using fillna(0) to treat NaNs as no mention
        has_identity = (df[id_cols].fillna(0) >= 0.5).any(axis=1)

        # Hard Negatives: Not Toxic AND Has Identity
        # These are the samples where the model is likely to be biased (false positives)
        hard_negative_mask = (~is_toxic) & has_identity

        # Apply multiplier to hard negatives
        weights[hard_negative_mask] = Config.IDENTITY_WEIGHT_MULTIPLIER

    return weights


def load_processed_train_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads training data, calculates sample weights, and handles caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, loads a small subset and skips caching.

    Returns:
        pd.DataFrame: Processed training dataframe with 'sample_weight' column.
    """
    # Debug mode: Load subset, calculate weights on fly, do not cache
    if debug:
        print(f"Debug mode: Loading {Config.DEBUG_SAMPLE_SIZE} samples from train.")
        df = pd.read_csv(Config.TRAIN_PATH, nrows=Config.DEBUG_SAMPLE_SIZE)
        weights = calculate_sample_weights(df)
        df["sample_weight"] = weights
        return df

    cache_path = os.path.join(Config.WORKING_DIR, "train_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data.")

    # 2. Process from scratch
    print("Processing training data from scratch...")
    df = pd.read_csv(Config.TRAIN_PATH)

    # Calculate and assign weights
    weights = calculate_sample_weights(df)
    df["sample_weight"] = weights

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Saved processed training data to {cache_path}")

    return df


class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=Config.MAX_LEN, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing text and labels.
            tokenizer: Transformer tokenizer (RoBERTa).
            max_len (int): Maximum sequence length.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Handle potential NaNs in text
        self.text = df[Config.TEXT_COL].fillna("").values

        # Pre-compute identity token IDs for Stochastic Identity Masking
        self.identity_token_ids = set()
        if self.mode == "train":
            keywords = get_identity_keywords()
            for word in keywords:
                # Encode word alone
                ids = tokenizer.encode(word, add_special_tokens=False)
                self.identity_token_ids.update(ids)
                # Encode word with leading space (RoBERTa uses 'Ġ' for space)
                # This ensures we catch " muslim" as well as "muslim" at start of sentence
                ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
                self.identity_token_ids.update(ids_space)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.text[index])

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        # Stochastic Identity Masking (Train only)
        if self.mode == "train":
            # Clone to avoid modifying shared storage if any
            input_ids = input_ids.clone()

            # Iterate through tokens to find identity terms
            # Note: Iterating over tensor elements in Python is generally slow,
            # but with max_len=256 it is acceptable for this task.
            for i, token_id in enumerate(input_ids):
                # Skip special tokens
                if token_id.item() in [
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id,
                ]:
                    continue

                # Check if token is an identity term
                if token_id.item() in self.identity_token_ids:
                    # Randomly mask with probability MASK_PROB
                    if torch.rand(1).item() < Config.MASK_PROB:
                        input_ids[i] = self.tokenizer.mask_token_id

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": self.df.iloc[index][Config.ID_COL],
        }

        # Add targets and weights if not in test mode
        if self.mode != "test":
            # Main Target
            target = self.df.iloc[index][Config.TARGET_COL]
            result["target"] = torch.tensor(target, dtype=torch.float)

            # Auxiliary Targets (for Multi-Task Learning)
            aux_targets = []
            for col in Config.AUX_COLUMNS:
                if col in self.df.columns:
                    aux_targets.append(self.df.iloc[index][col])
                else:
                    aux_targets.append(0.0)
            result["aux_targets"] = torch.tensor(aux_targets, dtype=torch.float)

            # Sample Weight
            if "sample_weight" in self.df.columns:
                weight = self.df.iloc[index]["sample_weight"]
                result["sample_weight"] = torch.tensor(weight, dtype=torch.float)
            else:
                result["sample_weight"] = torch.tensor(
                    Config.DEFAULT_WEIGHT, dtype=torch.float
                )

        return result
