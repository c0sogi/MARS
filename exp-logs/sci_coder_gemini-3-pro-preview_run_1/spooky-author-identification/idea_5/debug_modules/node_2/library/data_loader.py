import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from library.utils import ensure_directory

# Global constants
CACHE_DIR = "./working/idea_5/"
LABEL_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for Author Identification.
    Handles tokenization and label encoding.
    """

    def __init__(self, df, tokenizer, max_len=256, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'text' and 'id' columns.
                               Must contain 'author' if is_test is False.
            tokenizer: HuggingFace tokenizer instance.
            max_len (int): Maximum sequence length for tokenization.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Ensure text is string and fill NaNs
        self.texts = df["text"].fillna("").astype(str).values
        self.ids = df["id"].values

        if not self.is_test:
            self.labels = df["author"].map(LABEL_MAP).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        id_val = self.ids[idx]

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

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        item = {"id": id_val, "input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            # Convert label to tensor
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            item["target"] = label

        return item


def create_stratified_folds(
    data_path="./metadata/train.csv",
    n_folds=5,
    seed=42,
    load_cached_data=True,
    debug=False,
):
    """
    Creates stratified folds for cross-validation and caches the result.

    Args:
        data_path (str): Path to the training metadata CSV.
        n_folds (int): Number of folds.
        seed (int): Random seed for reproducibility.
        load_cached_data (bool): Whether to load from cache if available.
        debug (bool): If True, uses a smaller subset of data.

    Returns:
        pd.DataFrame: Dataframe with an added 'fold' column.
    """
    ensure_directory(CACHE_DIR)

    # Construct cache filename based on parameters to avoid collisions
    debug_suffix = "_debug" if debug else ""
    cache_filename = f"folds_{n_folds}f_seed{seed}{debug_suffix}.parquet"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached folds from {cache_path}...")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute folds
    print(f"Creating {n_folds} stratified folds (Debug={debug})...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)

    if debug:
        # Sample a subset for debugging (e.g., 1000 samples)
        # We stratify the sample as well to keep distribution
        from sklearn.model_selection import train_test_split

        try:
            df, _ = train_test_split(
                df, train_size=1000, stratify=df["author"], random_state=seed
            )
            print(f"Debug mode: Sampled {len(df)} rows.")
        except ValueError:
            # Fallback if dataset is too small for stratification
            df = df.sample(n=min(1000, len(df)), random_state=seed)

        # Reset index to ensure 0-based indexing for array mapping
        df = df.reset_index(drop=True)

    # Initialize fold column
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # StratifiedKFold expects y to be the target variable
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["author"])):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Saved folds to {cache_path}.")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df
