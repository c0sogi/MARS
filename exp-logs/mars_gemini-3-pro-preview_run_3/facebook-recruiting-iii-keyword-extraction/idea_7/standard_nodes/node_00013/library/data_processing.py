import os
import gc
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from collections import Counter

from library.config import Config
from library.utils import set_seed


class TokenizerHandler:
    def __init__(self, vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = None

    def train(self, texts, save_path):
        """Trains a BPE tokenizer on the provided texts."""
        print("Training BPE Tokenizer...")
        self.tokenizer = ByteLevelBPETokenizer()

        # Save texts to a temp file for tokenizer training to manage memory
        temp_text_path = os.path.join(Config.WORKING_DIR, "temp_train_text.txt")
        with open(temp_text_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        self.tokenizer.train(
            files=[temp_text_path],
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        )

        # Save tokenizer
        self.tokenizer.save(save_path)
        os.remove(temp_text_path)
        print(f"Tokenizer saved to {save_path}")

        # Configure post-training settings
        self._configure_tokenizer()

    def load(self, path):
        """Loads a trained tokenizer from file."""
        print(f"Loading tokenizer from {path}...")
        # We use the generic Tokenizer class for loading JSONs reliably
        self.tokenizer = Tokenizer.from_file(path)
        self._configure_tokenizer()

    def _configure_tokenizer(self):
        if self.tokenizer:
            self.tokenizer.enable_padding(
                pad_id=0, pad_token="<pad>", length=self.max_len
            )
            self.tokenizer.enable_truncation(max_length=self.max_len)

    def encode(self, texts):
        """Encodes a list of texts into a numpy array of token IDs."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Call train() or load() first.")

        # Batch encoding
        encodings = self.tokenizer.encode_batch(texts)
        return np.array([e.ids for e in encodings], dtype=np.int32)


class TargetEncoder:
    def __init__(self, top_k=Config.TOP_K_TAGS):
        self.top_k = top_k
        self.mlb = None
        self.classes_ = None

    def fit(self, tags_list):
        """
        Fits the encoder on a list of tag lists, keeping only top_k frequent tags.
        Args:
            tags_list: List of List of strings (e.g. [['python', 'pandas'], ['java']])
        """
        print(f"Fitting TargetEncoder with Top {self.top_k} tags...")
        all_tags = [tag for tags in tags_list for tag in tags]
        tag_counts = Counter(all_tags)
        top_tags = [t for t, c in tag_counts.most_common(self.top_k)]

        self.mlb = MultiLabelBinarizer(classes=top_tags)
        # We fit on the provided tags to initialize internal structures
        # Note: classes=top_tags ensures only those are considered
        self.mlb.fit(tags_list)
        self.classes_ = self.mlb.classes_

    def transform(self, tags_list):
        if self.mlb is None:
            raise ValueError("TargetEncoder not fitted.")
        return self.mlb.transform(tags_list).astype(np.float32)

    def inverse_transform(self, binary_preds):
        if self.mlb is None:
            raise ValueError("TargetEncoder not fitted.")
        return self.mlb.inverse_transform(binary_preds)

    def save(self, path):
        if self.classes_ is None:
            raise ValueError("Nothing to save.")
        np.save(path, self.classes_)
        print(f"TargetEncoder classes saved to {path}")

    def load(self, path):
        print(f"Loading TargetEncoder classes from {path}...")
        self.classes_ = np.load(path, allow_pickle=True)
        self.mlb = MultiLabelBinarizer(classes=self.classes_)
        # Dummy fit to initialize the transformer state
        self.mlb.fit([[]])


class StackExchangeDataset(Dataset):
    def __init__(self, tokens, labels=None):
        """
        Args:
            tokens (np.ndarray): Tokenized inputs (N, Max_Len)
            labels (np.ndarray, optional): Binary labels (N, Num_Classes)
        """
        self.tokens = torch.from_numpy(tokens).long()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.tokens[idx], self.labels[idx]
        return self.tokens[idx]


def get_dataloaders(
    train_tokens,
    train_labels,
    val_tokens,
    val_labels,
    test_tokens,
    batch_size=Config.BATCH_SIZE,
):
    """Creates DataLoaders for train, val, and test sets."""

    train_ds = StackExchangeDataset(train_tokens, train_labels)
    val_ds = StackExchangeDataset(val_tokens, val_labels)
    test_ds = StackExchangeDataset(test_tokens, None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def process_data(load_cached_data=True):
    """
    Main data processing pipeline.
    Loads raw data, tokenizes, encodes targets, and caches results.
    """
    set_seed()
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_tokens": Config.TRAIN_TOKENS_PATH,
        "train_labels": Config.TRAIN_LABELS_PATH,
        "val_tokens": Config.VAL_TOKENS_PATH,
        "val_labels": Config.VAL_LABELS_PATH,
        "test_tokens": Config.TEST_TOKENS_PATH,
        "test_ids": Config.TEST_IDS_PATH,
        "mlb_classes": os.path.join(Config.WORKING_DIR, "mlb_classes.npy"),
        "tokenizer": Config.TOKENIZER_PATH,
    }

    # 1. Check Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading cached data from disk...")
            train_tokens = np.load(cache_files["train_tokens"])
            train_labels = np.load(cache_files["train_labels"])
            val_tokens = np.load(cache_files["val_tokens"])
            val_labels = np.load(cache_files["val_labels"])
            test_tokens = np.load(cache_files["test_tokens"])
            test_ids = np.load(cache_files["test_ids"])

            tokenizer_handler = TokenizerHandler()
            tokenizer_handler.load(cache_files["tokenizer"])

            target_encoder = TargetEncoder()
            target_encoder.load(cache_files["mlb_classes"])

            return (
                train_tokens,
                train_labels,
                val_tokens,
                val_labels,
                test_tokens,
                test_ids,
                tokenizer_handler,
                target_encoder,
            )

    print("Cache missing or reload requested. Processing data from scratch...")

    # 2. Load Metadata
    print("Loading metadata...")
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Debugging subset logic
    if Config.DEBUG:
        print(f"DEBUG MODE: Using {Config.DEBUG_SIZE} samples.")
        df_train_meta = df_train_meta.iloc[: Config.DEBUG_SIZE]
        df_val_meta = df_val_meta.iloc[: int(Config.DEBUG_SIZE * 0.2)]
        df_test_meta = df_test_meta.iloc[: int(Config.DEBUG_SIZE * 0.2)]

    # 3. Load Raw Data and Merge
    print("Loading raw text data...")
    # Load only necessary columns to save memory
    df_raw_train = pd.read_csv(
        Config.RAW_TRAIN_PATH, usecols=["Id", "Title", "Body", "Tags"]
    )
    df_raw_test = pd.read_csv(Config.RAW_TEST_PATH, usecols=["Id", "Title", "Body"])

    df_raw_train.fillna("", inplace=True)
    df_raw_test.fillna("", inplace=True)

    # Merge to get splits
    train_df = pd.merge(df_train_meta, df_raw_train, on="Id", how="inner")
    val_df = pd.merge(df_val_meta, df_raw_train, on="Id", how="inner")
    test_df = pd.merge(df_test_meta, df_raw_test, on="Id", how="inner")

    # Cleanup raw dfs
    del df_raw_train, df_raw_test, df_train_meta, df_val_meta, df_test_meta
    gc.collect()

    # 4. Text Preprocessing
    print("Concatenating Title and Body...")
    train_text = (train_df["Title"] + " " + train_df["Body"]).tolist()
    val_text = (val_df["Title"] + " " + val_df["Body"]).tolist()
    test_text = (test_df["Title"] + " " + test_df["Body"]).tolist()

    # 5. Tokenization
    tokenizer_handler = TokenizerHandler(
        vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN
    )

    # Train tokenizer on training text
    tokenizer_handler.train(train_text, cache_files["tokenizer"])

    print("Tokenizing datasets...")
    train_tokens = tokenizer_handler.encode(train_text)
    val_tokens = tokenizer_handler.encode(val_text)
    test_tokens = tokenizer_handler.encode(test_text)

    # 6. Label Encoding
    print("Encoding labels...")
    train_tags = train_df["Tags_y"].apply(lambda x: str(x).split()).tolist()
    val_tags = val_df["Tags_y"].apply(lambda x: str(x).split()).tolist()

    target_encoder = TargetEncoder(top_k=Config.TOP_K_TAGS)
    target_encoder.fit(train_tags)
    target_encoder.save(cache_files["mlb_classes"])

    train_labels = target_encoder.transform(train_tags)
    val_labels = target_encoder.transform(val_tags)

    test_ids = test_df["Id"].values.astype(np.int32)

    # 7. Save to Cache
    print("Saving processed data to cache...")
    np.save(cache_files["train_tokens"], train_tokens)
    np.save(cache_files["train_labels"], train_labels)
    np.save(cache_files["val_tokens"], val_tokens)
    np.save(cache_files["val_labels"], val_labels)
    np.save(cache_files["test_tokens"], test_tokens)
    np.save(cache_files["test_ids"], test_ids)

    return (
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        test_ids,
        tokenizer_handler,
        target_encoder,
    )
