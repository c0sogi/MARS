import os
import re
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class Tokenizer:
    """
    Simple whitespace/regex tokenizer that builds a vocabulary from text
    and converts text to integer sequences.
    """

    def __init__(
        self,
        vocab_size=20000,
        token_pattern=r"(?u)\b\w+\b",
        unk_token="<UNK>",
        pad_token="<PAD>",
    ):
        self.vocab_size = vocab_size
        self.token_pattern = token_pattern
        self.unk_token = unk_token
        self.pad_token = pad_token

        # Initialize vocab with special tokens
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.is_fitted = False

    def fit(self, texts):
        """
        Builds vocabulary from a list of text strings.
        """
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(str(text))
            word_counts.update(tokens)

        # Select top N frequent words
        most_common = word_counts.most_common(
            self.vocab_size - 2
        )  # Reserve 2 for PAD, UNK

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.is_fitted = True

    def encode(self, text, max_len=None):
        """
        Converts a text string into a list of integer token IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding.")

        tokens = self._tokenize(str(text))
        token_ids = [
            self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens
        ]

        if max_len is not None:
            token_ids = token_ids[:max_len]

        return token_ids

    def _tokenize(self, text):
        """
        Splits text into tokens based on regex pattern.
        """
        return re.findall(self.token_pattern, text.lower())

    def save(self, path):
        """
        Saves the vocabulary to a JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path):
        """
        Loads the vocabulary from a JSON file.
        """
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.is_fitted = True

    def __len__(self):
        return len(self.word2idx)


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    """

    def __init__(self, data, is_test=False):
        """
        Args:
            data (pd.DataFrame): DataFrame containing 'essay_id', 'input_ids', and optionally 'score'.
            is_test (bool): If True, 'score' column is not expected.
        """
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        essay_id = row["essay_id"]
        # input_ids might be a numpy array or list depending on how it was loaded
        input_ids = row["input_ids"]
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()

        if self.is_test:
            return {"essay_id": essay_id, "input_ids": input_ids}
        else:
            score = float(row["score"])
            return {"essay_id": essay_id, "input_ids": input_ids, "score": score}


def collate_fn(batch):
    """
    Custom collate function to pad sequences dynamically.
    """
    # Separate inputs
    essay_ids = [item["essay_id"] for item in batch]
    input_ids_list = [item["input_ids"] for item in batch]

    # Determine max length in this batch (clipped to Config.MAX_LEN)
    # Note: Tokenizer.encode might have already truncated to MAX_LEN,
    # but we double check here or handle variable lengths.
    batch_max_len = max(len(ids) for ids in input_ids_list)
    batch_max_len = min(batch_max_len, Config.MAX_LEN)

    # Pad sequences
    padded_input_ids = []
    for ids in input_ids_list:
        # Truncate if necessary
        ids = ids[:batch_max_len]
        # Pad
        padding_length = batch_max_len - len(ids)
        padded_ids = ids + [0] * padding_length  # 0 is PAD_TOKEN
        padded_input_ids.append(padded_ids)

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)

    result = {"essay_ids": essay_ids, "input_ids": input_ids_tensor}

    # Handle scores if present
    if "score" in batch[0]:
        scores = [item["score"] for item in batch]
        scores_tensor = torch.tensor(scores, dtype=torch.float)
        result["scores"] = scores_tensor

    return result


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders. Handles caching of tokenized data.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed Parquet files and vocab.

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    vocab_path = os.path.join(Config.CACHE_DIR, "vocab.json")
    train_cache_path = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(vocab_path)
        and os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    tokenizer = Tokenizer(
        vocab_size=Config.VOCAB_SIZE,
        token_pattern=Config.TOKEN_PATTERN,
        unk_token=Config.UNK_TOKEN,
        pad_token=Config.PAD_TOKEN,
    )

    if load_cached_data and cache_exists:
        print("Loading cached processed data...")
        tokenizer.load(vocab_path)
        df_train = pd.read_parquet(train_cache_path)
        df_val = pd.read_parquet(val_cache_path)
        df_test = pd.read_parquet(test_cache_path)

    else:
        print("Processing data from scratch...")
        # Load raw metadata
        df_train_raw = pd.read_csv(Config.TRAIN_DATA_PATH)
        df_val_raw = pd.read_csv(Config.VAL_DATA_PATH)
        df_test_raw = pd.read_csv(Config.TEST_DATA_PATH)

        # Fit tokenizer on training text
        print("Fitting tokenizer...")
        tokenizer.fit(df_train_raw["full_text"].astype(str).tolist())
        tokenizer.save(vocab_path)

        # Helper to process a dataframe
        def process_df(df, is_test=False):
            # Encode texts
            # We don't truncate strictly here to allow dynamic padding in collate_fn,
            # but we can cap huge outliers to save memory if needed.
            # Config.MAX_LEN is mainly used in collate_fn.
            encoded_seqs = [
                tokenizer.encode(text, max_len=Config.MAX_LEN)
                for text in df["full_text"].astype(str)
            ]

            out_df = pd.DataFrame(
                {"essay_id": df["essay_id"], "input_ids": encoded_seqs}
            )

            if not is_test:
                out_df["score"] = df["score"]

            return out_df

        print("Tokenizing datasets...")
        df_train = process_df(df_train_raw, is_test=False)
        df_val = process_df(df_val_raw, is_test=False)
        df_test = process_df(df_test_raw, is_test=True)

        # Save to cache
        print("Saving processed data to cache...")
        df_train.to_parquet(train_cache_path, index=False)
        df_val.to_parquet(val_cache_path, index=False)
        df_test.to_parquet(test_cache_path, index=False)

    # Create Datasets
    train_dataset = EssayDataset(df_train, is_test=False)
    val_dataset = EssayDataset(df_val, is_test=False)
    test_dataset = EssayDataset(df_test, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, tokenizer
