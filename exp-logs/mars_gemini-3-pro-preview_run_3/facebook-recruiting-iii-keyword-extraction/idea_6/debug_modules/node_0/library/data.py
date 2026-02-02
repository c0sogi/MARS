import os
import json
import joblib
import gc
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from library import config, utils

# Initialize logger
logger = utils.get_logger("data")


class Tokenizer:
    """
    Custom Tokenizer to convert text to sequences of integer IDs.
    """

    def __init__(self, vocab_size=config.VOCAB_SIZE, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        # Reserve 0 for padding, 1 for unknown
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts):
        """
        Builds vocabulary from a list of text strings.
        """
        logger.info("Building vocabulary...")
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        # Filter by min_freq
        sorted_words = [w for w, c in word_counts.most_common() if c >= self.min_freq]

        # Keep top (vocab_size - 2) words
        top_words = sorted_words[: self.vocab_size - 2]

        for i, word in enumerate(top_words):
            idx = i + 2
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(f"Vocabulary built. Size: {len(self.word2idx)}")

    def transform(self, texts, max_len=config.MAX_LEN):
        """
        Converts a list of texts to a numpy array of fixed-length integer sequences.
        """
        seqs = []
        unk_idx = self.word2idx["<UNK>"]
        pad_idx = self.word2idx["<PAD>"]

        for text in texts:
            words = text.split()
            # Map words to indices
            seq = [self.word2idx.get(w, unk_idx) for w in words]

            # Pad or truncate
            if len(seq) < max_len:
                seq = seq + [pad_idx] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            seqs.append(seq)

        return np.array(seqs, dtype=np.int32)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path):
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}


class TagEncoder:
    """
    Encodes target tags into binary vectors (Multi-Hot Encoding).
    """

    def __init__(self, top_k=config.TOP_K_TAGS):
        self.top_k = top_k
        self.mlb = None
        self.classes_ = None

    def fit(self, tags_series):
        """
        Identifies top K tags and fits the MultiLabelBinarizer.
        """
        logger.info("Fitting TagEncoder...")
        # Flatten all tags to count frequencies
        all_tags = []
        for tags_str in tags_series:
            if isinstance(tags_str, str):
                all_tags.extend(tags_str.split())

        tag_counts = Counter(all_tags)
        top_tags = [t for t, c in tag_counts.most_common(self.top_k)]
        self.classes_ = top_tags
        logger.info(f"Selected {len(self.classes_)} top tags.")

        self.mlb = MultiLabelBinarizer(classes=self.classes_)
        # Fit with a dummy list containing all classes to initialize state
        self.mlb.fit([self.classes_])

    def transform(self, tags_series):
        """
        Transforms tags strings into a binary matrix.
        Returns int8 array to save memory.
        """
        tags_list = [t.split() if isinstance(t, str) else [] for t in tags_series]
        mat = self.mlb.transform(tags_list)
        return mat.astype(np.int8)

    def inverse_transform(self, binary_matrix):
        return self.mlb.inverse_transform(binary_matrix)

    def save(self, path):
        joblib.dump(self.mlb, path)

    def load(self, path):
        self.mlb = joblib.load(path)
        self.classes_ = self.mlb.classes_


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for Stack Exchange questions.
    """

    def __init__(self, features, labels=None, ids=None):
        self.features = torch.tensor(features, dtype=torch.long)
        self.labels = None
        if labels is not None:
            # Convert to float32 for BCEWithLogitsLoss
            self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        # For test set, return features and ID
        return self.features[idx], self.ids[idx]


def load_raw_data(metadata_path, raw_data_path):
    """
    Loads metadata and merges it with the raw text content.
    """
    logger.info(f"Reading metadata from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    logger.info(f"Reading raw content from {raw_data_path}...")
    # Optimize memory by reading only necessary columns
    usecols = ["Id", "Title", "Body"]
    df_raw = pd.read_csv(raw_data_path, usecols=usecols)

    logger.info("Merging data...")
    df = pd.merge(df_meta, df_raw, on="Id", how="inner")

    # Fill missing values
    df["Title"] = df["Title"].fillna("")
    df["Body"] = df["Body"].fillna("")
    if "Tags" in df.columns:
        df["Tags"] = df["Tags"].fillna("")

    return df


def prepare_data(load_cached_data=True):
    """
    Main data processing pipeline.
    Checks for cached artifacts; if not found, processes from scratch.
    """
    # Define cache paths
    cache_paths = [
        config.TRAIN_FEATURES_PATH,
        config.TRAIN_LABELS_PATH,
        config.VAL_FEATURES_PATH,
        config.VAL_LABELS_PATH,
        config.TEST_FEATURES_PATH,
        config.TEST_IDS_PATH,
        config.TOKENIZER_PATH,
        config.MLB_PATH,
    ]

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_paths)

    if load_cached_data and cache_exists:
        logger.info("Loading pre-processed data from cache...")
        train_features = np.load(config.TRAIN_FEATURES_PATH)
        train_labels = np.load(config.TRAIN_LABELS_PATH)
        val_features = np.load(config.VAL_FEATURES_PATH)
        val_labels = np.load(config.VAL_LABELS_PATH)
        test_features = np.load(config.TEST_FEATURES_PATH)
        test_ids = np.load(config.TEST_IDS_PATH)

        tokenizer = Tokenizer()
        tokenizer.load(config.TOKENIZER_PATH)

        tag_encoder = TagEncoder()
        tag_encoder.load(config.MLB_PATH)

        return (
            train_features,
            train_labels,
            val_features,
            val_labels,
            test_features,
            test_ids,
            tokenizer,
            tag_encoder,
        )

    logger.info("Cache missing or reload requested. Processing data from scratch...")

    # 1. Load Datasets
    df_train = load_raw_data(config.TRAIN_META_FILE, config.TRAIN_RAW_FILE)
    df_val = load_raw_data(config.VAL_META_FILE, config.TRAIN_RAW_FILE)
    df_test = load_raw_data(config.TEST_META_FILE, config.TEST_RAW_FILE)

    # 2. Text Cleaning & Preprocessing
    logger.info("Cleaning and concatenating text...")

    def get_clean_text(df):
        # Combine Title and Body
        raw_text = df["Title"] + " " + df["Body"]
        # Apply cleaning utility
        return raw_text.apply(utils.clean_text).tolist()

    train_texts = get_clean_text(df_train)
    val_texts = get_clean_text(df_val)
    test_texts = get_clean_text(df_test)

    # 3. Tokenization
    logger.info("Tokenizing text...")
    tokenizer = Tokenizer(vocab_size=config.VOCAB_SIZE)
    tokenizer.fit(train_texts)
    tokenizer.save(config.TOKENIZER_PATH)

    train_features = tokenizer.transform(train_texts)
    val_features = tokenizer.transform(val_texts)
    test_features = tokenizer.transform(test_texts)

    # 4. Label Encoding
    logger.info("Encoding tags...")
    tag_encoder = TagEncoder(top_k=config.TOP_K_TAGS)
    tag_encoder.fit(df_train["Tags"])
    tag_encoder.save(config.MLB_PATH)

    train_labels = tag_encoder.transform(df_train["Tags"])
    val_labels = tag_encoder.transform(df_val["Tags"])

    test_ids = df_test["Id"].values

    # 5. Save to Cache
    logger.info("Saving artifacts to cache...")
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    np.save(config.TRAIN_FEATURES_PATH, train_features)
    np.save(config.TRAIN_LABELS_PATH, train_labels)
    np.save(config.VAL_FEATURES_PATH, val_features)
    np.save(config.VAL_LABELS_PATH, val_labels)
    np.save(config.TEST_FEATURES_PATH, test_features)
    np.save(config.TEST_IDS_PATH, test_ids)

    # Cleanup
    del df_train, df_val, df_test, train_texts, val_texts, test_texts
    gc.collect()

    return (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_ids,
        tokenizer,
        tag_encoder,
    )


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached_data=True
):
    """
    Generates PyTorch DataLoaders for Train, Validation, and Test sets.
    """
    data_artifacts = prepare_data(load_cached_data=load_cached_data)
    train_feat, train_lbl, val_feat, val_lbl, test_feat, test_ids, _, tag_encoder = (
        data_artifacts
    )

    logger.info("Initializing DataSets and DataLoaders...")

    train_dataset = StackExchangeDataset(train_feat, train_lbl)
    val_dataset = StackExchangeDataset(val_feat, val_lbl)
    test_dataset = StackExchangeDataset(test_feat, ids=test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tag_encoder
