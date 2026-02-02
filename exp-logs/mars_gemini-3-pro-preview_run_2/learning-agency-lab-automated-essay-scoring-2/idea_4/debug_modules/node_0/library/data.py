import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Expects a DataFrame containing 'input_ids' and 'attention_mask' columns.
    If is_test is False, it also expects a 'score' column.
    """

    def __init__(self, df, is_test=False):
        self.df = df.reset_index(drop=True)
        self.is_test = is_test

        # Validation of required columns
        if (
            "input_ids" not in self.df.columns
            or "attention_mask" not in self.df.columns
        ):
            raise ValueError(
                "DataFrame must contain 'input_ids' and 'attention_mask' columns."
            )

        if not self.is_test and "score" not in self.df.columns:
            raise ValueError(
                "DataFrame must contain 'score' column for training/validation."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert pre-computed lists/arrays to tensors
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not self.is_test:
            # Regression target: float (for SmoothL1Loss)
            sample["labels"] = torch.tensor(row["score"], dtype=torch.float)

        return sample


def process_data(df, tokenizer):
    """
    Tokenizes the 'full_text' column in the DataFrame using the provided tokenizer.
    Adds 'input_ids' and 'attention_mask' columns to the DataFrame.
    """
    texts = df["full_text"].fillna("").astype(str).tolist()

    # Use batch_encode_plus for efficiency
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors=None,  # Return python lists to store in DataFrame
    )

    df["input_ids"] = encodings["input_ids"]
    df["attention_mask"] = encodings["attention_mask"]

    return df


def load_dataset(
    split="train", tokenizer=None, load_cached_data=True, debug=False, debug_size=None
):
    """
    Loads the dataset for a specific split with caching mechanism.

    Args:
        split (str): 'train', 'val', or 'test'.
        tokenizer: Transformers tokenizer instance. If None, loads from Config.MODEL_NAME.
        load_cached_data (bool): Whether to try loading from cache.
        debug (bool): If True, limits dataset size.
        debug_size (int): Number of samples to use in debug mode.
    """
    seed_everything(Config.SEED)

    # Determine paths based on split
    if split == "train":
        input_path = Config.TRAIN_PATH
        cache_filename = "train_processed.parquet"
    elif split == "val":
        input_path = Config.VAL_PATH
        cache_filename = "val_processed.parquet"
    elif split == "test":
        input_path = Config.TEST_PATH
        cache_filename = "test_processed.parquet"
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data for '{split}' from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)

            if debug:
                limit = (
                    debug_size if debug_size is not None else Config.DEBUG_SAMPLE_SIZE
                )
                print(f"Debug mode: truncating to {limit} samples.")
                df = df.head(limit)

            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Proceeding to re-compute.")

    # 2. Compute from scratch
    print(f"Processing data for '{split}' from {input_path}...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Apply debug truncation BEFORE processing to save time if debug=True
    if debug:
        limit = debug_size if debug_size is not None else Config.DEBUG_SAMPLE_SIZE
        print(f"Debug mode: truncating raw data to {limit} samples.")
        df = df.head(limit)

    # Initialize tokenizer if not provided
    if tokenizer is None:
        print(f"Initializing tokenizer: {Config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Process text
    df = process_data(df, tokenizer)

    # Save to cache (only if not debugging, to keep cache clean)
    if not debug:
        print(f"Saving processed data to {cache_path}")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


def get_folds(df, n_folds=None, seed=None):
    """
    Performs Stratified K-Fold splitting on the dataframe.
    Adds a 'fold' column to the dataframe.
    """
    if n_folds is None:
        n_folds = Config.N_FOLDS
    if seed is None:
        seed = Config.SEED

    # Ensure score exists for stratification
    if "score" not in df.columns:
        raise ValueError("Cannot perform stratified split: 'score' column missing.")

    # Reset index to ensure alignment
    df = df.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    df["fold"] = -1

    # Stratify by score
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(df, df["score"])):
        df.loc[val_idx, "fold"] = fold_id

    return df
