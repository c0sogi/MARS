import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from library.config import Config


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for Author Identification.
    Handles tokenization of text sequences and label encoding.
    """

    def __init__(self, texts, labels=None, tokenizer=None, max_length=None):
        """
        Args:
            texts (list or np.array): List of text sentences.
            labels (list or np.array, optional): List of author labels (strings or ints).
            tokenizer (PreTrainedTokenizer, optional): Transformer tokenizer.
            max_length (int, optional): Maximum sequence length. Defaults to Config.MAX_LENGTH.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else Config.MAX_LENGTH

        # Initialize tokenizer if not provided
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by tokenizer
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # Handle labels if present
        if self.labels is not None:
            label_val = self.labels[idx]

            # Convert string label to ID if necessary
            if isinstance(label_val, str):
                label_val = Config.LABEL2ID[label_val]

            item["labels"] = torch.tensor(label_val, dtype=torch.long)

        return item


def create_folds(load_cached_data=True, debug=False):
    """
    Loads training data and assigns Stratified K-Fold indices.

    Args:
        load_cached_data (bool): Whether to load from existing parquet cache.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        pd.DataFrame: Training data with an additional 'fold' column.
    """
    # Define cache path based on debug status
    cache_filename = "folds_debug.parquet" if debug else "folds.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # If in debug mode, ensure the cached file is actually small
            if debug and len(df) > Config.DEBUG_SAMPLE_SIZE:
                # Cache mismatch, proceed to recompute
                pass
            else:
                return df
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Load raw data and compute folds
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {Config.TRAIN_DATA_PATH}")

    df = pd.read_csv(Config.TRAIN_DATA_PATH)

    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Initialize Folds
    df["fold"] = -1
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Assign folds based on 'author' stratification
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["author"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_test_dataset(tokenizer=None, debug=False):
    """
    Loads test data and returns a Dataset object and IDs.

    Args:
        tokenizer: Transformer tokenizer.
        debug (bool): Whether to use a subset of test data.

    Returns:
        tuple: (AuthorDataset, numpy array of IDs)
    """
    if not os.path.exists(Config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {Config.TEST_DATA_PATH}")

    df = pd.read_csv(Config.TEST_DATA_PATH)

    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    texts = df["text"].values
    ids = df["id"].values

    dataset = AuthorDataset(texts=texts, labels=None, tokenizer=tokenizer)

    return dataset, ids
