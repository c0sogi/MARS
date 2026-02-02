import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Mapping class labels to integers based on submission format order
LABEL_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for Author Identification.
    Handles tokenization and label encoding for DeBERTa models.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'text' and optionally 'author'.
            tokenizer: Transformers tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            is_test (bool): Whether this is a test set (no labels).
        """
        self.texts = df["text"].values.astype(str)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        if not self.is_test:
            # Map string labels to integers
            self.labels = df["author"].map(LABEL_MAP).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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

        # Flatten to remove batch dimension added by return_tensors='pt'
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def get_tokenizer():
    """
    Initializes and returns the tokenizer defined in the configuration.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_NAME)


def load_train_data(load_cached_data=True, debug=False):
    """
    Loads the full training data (combining train and val metadata),
    creates Stratified K-Folds, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache.
        debug (bool): If True, returns a small subset for debugging.

    Returns:
        pd.DataFrame: Dataframe containing text, author, and fold assignments.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "train_folds.parquet")
    df = None

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            # Fallback to re-creation if load fails
            df = None

    # 2. Create from scratch if needed
    if df is None:
        # Load metadata files
        df_train_meta = pd.read_csv(Config.TRAIN_META)
        df_val_meta = pd.read_csv(Config.VAL_META)

        # Combine to maximize data for CV
        df = pd.concat([df_train_meta, df_val_meta], axis=0, ignore_index=True)

        # Create Stratified Folds
        seed_everything(Config.SEED)
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        df["fold"] = -1
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["author"])):
            df.loc[val_idx, "fold"] = fold

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    # 3. Handle Debug Mode
    if debug:
        seed_everything(Config.SEED)
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    return df


def load_test_data():
    """
    Loads the test data from metadata.

    Returns:
        pd.DataFrame: Test dataframe.
    """
    df_test = pd.read_csv(Config.TEST_META)
    return df_test
