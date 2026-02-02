import os
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config
from library.utils import load_or_create

# =============================================================================
# Text Processing Utilities
# =============================================================================


class TextTokenizer:
    """
    Tokenizer that handles vocabulary building, caching, and text-to-sequence conversion.
    """

    def __init__(
        self,
        max_len=None,
        vocab_size=None,
        min_freq=None,
    ):
        self.max_len = max_len if max_len is not None else Config.MAX_LEN
        self.vocab_size = vocab_size if vocab_size is not None else Config.VOCAB_SIZE
        self.min_freq = min_freq if min_freq is not None else Config.MIN_WORD_FREQ
        self.vocab = None
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_idx = 0
        self.unk_idx = 1

        # Regex for cleaning HTML
        self.cleanr = re.compile(r"<[^>]+>")

    def clean_text(self, text):
        """Removes HTML tags and lowercases text."""
        if not isinstance(text, str):
            return ""
        text = self.cleanr.sub(" ", text)
        return text.lower().strip()

    def _compute_vocab(self, texts):
        """
        Computes vocabulary from a list/series of texts.
        Intended to be used with load_or_create.
        """
        print("Building vocabulary from scratch...")
        counter = Counter()
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            counter.update(tokens)

        # Filter by min_freq
        valid_words = [w for w, c in counter.items() if c >= self.min_freq]

        # Sort by frequency (descending) and take top VOCAB_SIZE
        # We need to re-count or just use the counter values.
        # Ideally, we sort the valid words by their counts.
        valid_words.sort(key=lambda w: counter[w], reverse=True)
        valid_words = valid_words[: self.vocab_size - 2]  # Reserve 2 spots for pad, unk

        vocab = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        for i, word in enumerate(valid_words):
            vocab[word] = i + 2

        print(f"Vocabulary built. Size: {len(vocab)}")
        return vocab

    def fit(self, texts, load_cached_data=True):
        """
        Fits the tokenizer on the provided texts.
        Uses caching to avoid re-computing.
        """
        self.vocab = load_or_create(
            file_path=Config.TOKENIZER_PATH,
            compute_func=self._compute_vocab,
            load_cached_data=load_cached_data,
            file_type="json",
            texts=texts,
        )

    def transform(self, texts):
        """
        Converts a list/series of texts to a padded numpy array of token IDs.
        """
        if self.vocab is None:
            raise ValueError("Tokenizer has not been fitted yet.")

        sequences = []
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            # Convert to indices
            seq = [self.vocab.get(token, self.unk_idx) for token in tokens]

            # Truncate or Pad
            if len(seq) > self.max_len:
                seq = seq[: self.max_len]
            else:
                seq = seq + [self.pad_idx] * (self.max_len - len(seq))
            sequences.append(seq)

        return np.array(sequences, dtype=np.int64)


class TagEncoder:
    """
    Encoder for multi-label targets. Handles mapping tags to indices and back.
    """

    def __init__(self, num_tags=None):
        self.num_tags = num_tags if num_tags is not None else Config.NUM_TAGS
        self.tag_list = None
        self.tag_to_idx = None

    def _compute_tag_map(self, tags_series):
        """
        Computes the top N frequent tags.
        """
        print("Building tag map from scratch...")
        all_tags = tags_series.astype(str).str.split().explode()
        tag_counts = all_tags.value_counts()

        top_tags = tag_counts.head(self.num_tags).index.tolist()
        # Sort for determinism
        top_tags.sort()

        print(f"Tag map built. Top {len(top_tags)} tags selected.")
        return np.array(top_tags)

    def fit(self, tags_series, load_cached_data=True):
        """
        Fits the encoder on the provided tags.
        """
        self.tag_list = load_or_create(
            file_path=Config.TAG_MAP_PATH,
            compute_func=self._compute_tag_map,
            load_cached_data=load_cached_data,
            file_type="npy",
            tags_series=tags_series,
        )
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tag_list)}

    def transform(self, tags_list):
        """
        Converts a list of tag strings (space-delimited) to a multi-hot binary matrix.
        """
        if self.tag_to_idx is None:
            raise ValueError("TagEncoder has not been fitted yet.")

        batch_size = len(tags_list)
        matrix = np.zeros((batch_size, len(self.tag_list)), dtype=np.float32)

        for i, tags_str in enumerate(tags_list):
            if not isinstance(tags_str, str):
                continue
            tags = tags_str.split()
            for tag in tags:
                if tag in self.tag_to_idx:
                    matrix[i, self.tag_to_idx[tag]] = 1.0

        return matrix

    def inverse_transform(self, probs, threshold=Config.PREDICTION_THRESHOLD):
        """
        Converts probabilities back to space-delimited tag strings.
        """
        if self.tag_list is None:
            raise ValueError("TagEncoder has not been fitted yet.")

        preds = []
        for row_probs in probs:
            # Get indices where prob > threshold
            indices = np.where(row_probs > threshold)[0]
            if len(indices) == 0:
                # Fallback: if no tag exceeds threshold, take the max one
                # or return empty string. Usually returning at least one is better.
                indices = [np.argmax(row_probs)]

            tags = self.tag_list[indices]
            preds.append(" ".join(tags))

        return preds


# =============================================================================
# Dataset
# =============================================================================


class StackExchangeDataset(Dataset):
    def __init__(
        self, metadata_df, text_df, tokenizer, tag_encoder=None, is_test=False
    ):
        """
        Args:
            metadata_df: DataFrame containing Id and Tags (if train/val).
            text_df: DataFrame containing Id, Title, Body.
            tokenizer: Fitted TextTokenizer instance.
            tag_encoder: Fitted TagEncoder instance (optional for test).
            is_test: Boolean flag.
        """
        self.is_test = is_test

        # Merge metadata with text
        # We assume metadata_df has 'Id' and text_df has 'Id'
        print(
            f"Initializing Dataset (Test={is_test}). Merging {len(metadata_df)} metadata rows with text..."
        )
        self.data = metadata_df.merge(text_df, on="Id", how="inner")
        print(f"Merged dataset shape: {self.data.shape}")

        # Pre-process text
        # Concatenate Title and Body
        full_text = self.data["Title"].fillna("") + " " + self.data["Body"].fillna("")
        print("Tokenizing text...")
        self.input_ids = tokenizer.transform(full_text)

        # Pre-process tags if not test
        if not is_test:
            if tag_encoder is None:
                raise ValueError(
                    "TagEncoder must be provided for training/validation data."
                )
            print("Encoding tags...")
            self.targets = tag_encoder.transform(self.data["Tags"].fillna(""))
        else:
            self.ids = self.data["Id"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return tensors
        input_seq = torch.tensor(self.input_ids[idx], dtype=torch.long)

        if self.is_test:
            row_id = self.ids[idx]
            return input_seq, row_id
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return input_seq, target


# =============================================================================
# Data Loading Functions
# =============================================================================


def get_dataloaders(debug=Config.DEBUG):
    """
    Prepares DataLoaders for Training and Validation.

    Returns:
        train_loader, val_loader, tokenizer, tag_encoder
    """
    print("--- Preparing Data Loaders ---")

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)

    if debug:
        print(f"DEBUG mode: Sampling {Config.DEBUG_SIZE} rows.")
        train_meta = train_meta.sample(
            n=min(len(train_meta), Config.DEBUG_SIZE), random_state=Config.SEED
        )
        val_meta = val_meta.sample(
            n=min(len(val_meta), Config.DEBUG_SIZE // 5), random_state=Config.SEED
        )

    # 2. Load Raw Text (Train.csv)
    # We only need rows that are in our metadata
    required_ids = set(train_meta["Id"]).union(set(val_meta["Id"]))

    print(f"Reading {Config.TRAIN_CSV}...")
    # Reading full file then filtering might be heavy, but pandas is efficient enough for 5M rows on 220GB RAM.
    # We use usecols to save memory.
    df_text = pd.read_csv(
        Config.TRAIN_CSV,
        usecols=["Id", "Title", "Body"],
        dtype={"Id": "int64", "Title": "object", "Body": "object"},
    )

    # Filter to reduce memory usage before processing
    df_text = df_text[df_text["Id"].isin(required_ids)]

    # 3. Fit Tokenizer (on Train Text only to avoid leakage)
    # We need to merge train_meta with df_text temporarily to get the text for fitting
    train_text_subset = train_meta.merge(df_text, on="Id", how="inner")
    full_train_text = (
        train_text_subset["Title"].fillna("")
        + " "
        + train_text_subset["Body"].fillna("")
    )

    tokenizer = TextTokenizer()
    tokenizer.fit(full_train_text, load_cached_data=True)

    # 4. Fit Tag Encoder (on Train Tags only)
    tag_encoder = TagEncoder()
    tag_encoder.fit(train_meta["Tags"].fillna(""), load_cached_data=True)

    # 5. Create Datasets
    train_dataset = StackExchangeDataset(
        train_meta, df_text, tokenizer, tag_encoder, is_test=False
    )
    val_dataset = StackExchangeDataset(
        val_meta, df_text, tokenizer, tag_encoder, is_test=False
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer, tag_encoder


def get_test_dataloader(tokenizer, debug=Config.DEBUG):
    """
    Prepares DataLoader for Testing/Inference.

    Args:
        tokenizer: The fitted TextTokenizer used during training.
        debug: If True, process a subset.
    """
    print("--- Preparing Test Data Loader ---")

    # 1. Load Metadata
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    if debug:
        test_meta = test_meta.head(Config.DEBUG_SIZE)

    # 2. Load Raw Text (Test.csv)
    print(f"Reading {Config.TEST_CSV}...")
    df_text = pd.read_csv(
        Config.TEST_CSV,
        usecols=["Id", "Title", "Body"],
        dtype={"Id": "int64", "Title": "object", "Body": "object"},
    )

    # 3. Create Dataset
    test_dataset = StackExchangeDataset(
        test_meta, df_text, tokenizer, tag_encoder=None, is_test=True
    )

    # 4. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
