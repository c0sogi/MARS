import os
import json
import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from library import config
from library import utils


# ==========================================
# Vocabulary Class
# ==========================================
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = config.PAD_TOKEN
        self.unk_token = config.UNK_TOKEN
        # Initialize with special tokens
        self.add_token(self.pad_token)  # idx 0
        self.add_token(self.unk_token)  # idx 1

    def add_token(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build(self, texts, max_size=config.MAX_VOCAB_SIZE):
        """
        Builds vocabulary from a list of text strings.
        """
        print("Building vocabulary...")
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)

        # Keep top max_size - 2 (for PAD and UNK)
        most_common = counter.most_common(max_size - 2)

        for word, _ in most_common:
            self.add_token(word)

        print(f"Vocabulary built with {len(self)} tokens.")

    def encode(self, text):
        """
        Converts a text string into a list of indices.
        """
        tokens = text.split()
        # Truncate to MAX_SEQ_LEN
        tokens = tokens[: config.MAX_SEQ_LEN]
        return [
            self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens
        ]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}


# ==========================================
# TagEncoder Class
# ==========================================
class TagEncoder:
    def __init__(self):
        self.tag2idx = {}
        self.idx2tag = {}

    def build(self, tags_list, top_k=config.NUM_TARGET_TAGS):
        """
        Builds tag mapping from a list of space-delimited tag strings.
        """
        print("Building tag encoder...")
        counter = Counter()
        for tags in tags_list:
            if not isinstance(tags, str):
                continue
            t_split = tags.split()
            counter.update(t_split)

        most_common = counter.most_common(top_k)

        for idx, (tag, _) in enumerate(most_common):
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag

        print(f"TagEncoder built with {len(self)} tags.")

    def encode(self, tags_str):
        """
        Converts a space-delimited tag string into a list of active indices.
        """
        if not isinstance(tags_str, str):
            return []

        indices = []
        for tag in tags_str.split():
            if tag in self.tag2idx:
                indices.append(self.tag2idx[tag])
        return indices

    def to_multi_hot(self, indices):
        """
        Converts a list of indices to a multi-hot tensor.
        """
        vec = torch.zeros(len(self.tag2idx), dtype=torch.float32)
        if indices:
            vec[indices] = 1.0
        return vec

    def decode(self, probs, threshold=0.5):
        """
        Converts probabilities to a list of tag strings.
        """
        indices = np.where(probs > threshold)[0]
        return [self.idx2tag[i] for i in indices]

    def __len__(self):
        return len(self.tag2idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"tag2idx": self.tag2idx}, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.tag2idx = data["tag2idx"]
            self.idx2tag = {int(v): k for k, v in self.tag2idx.items()}


# ==========================================
# Dataset Class
# ==========================================
class StackExchangeDataset(Dataset):
    def __init__(self, df, vocab, tag_encoder, is_test=False):
        self.df = df
        self.vocab = vocab
        self.tag_encoder = tag_encoder
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Input: List of token indices
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)

        # Metadata
        q_id = row["Id"]

        if self.is_test:
            return input_ids, q_id
        else:
            # Label: Convert stored indices to multi-hot vector
            tag_indices = row["tag_indices"]
            label = self.tag_encoder.to_multi_hot(tag_indices)
            return input_ids, label, q_id


# ==========================================
# Collate Function
# ==========================================
def collate_fn(batch):
    # Separate inputs
    if len(batch[0]) == 3:  # Train/Val
        input_ids_list, labels_list, ids_list = zip(*batch)
        has_labels = True
    else:  # Test
        input_ids_list, ids_list = zip(*batch)
        has_labels = False

    # Pad sequences
    # batch_first=True -> (batch_size, max_seq_len)
    padded_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=config.PAD_TOKEN_ID if hasattr(config, "PAD_TOKEN_ID") else 0,
    )

    # Calculate lengths (for masking if needed)
    lengths = torch.tensor([len(x) for x in input_ids_list], dtype=torch.long)

    ids = torch.tensor(ids_list, dtype=torch.long)

    if has_labels:
        labels = torch.stack(labels_list)
        return padded_input_ids, lengths, labels, ids
    else:
        return padded_input_ids, lengths, ids


# ==========================================
# Data Preparation Logic
# ==========================================
def prepare_data(
    split, vocab=None, tag_encoder=None, load_cached_data=True, debug=False
):
    """
    Loads, processes, and caches data.

    Args:
        split (str): 'train', 'val', or 'test'.
        vocab (Vocabulary, optional): Existing vocabulary (required for val/test).
        tag_encoder (TagEncoder, optional): Existing tag encoder (required for val/test).
        load_cached_data (bool): Whether to try loading from cache.
        debug (bool): Whether to use a small subset.

    Returns:
        tuple: (processed_df, vocab, tag_encoder)
    """

    # Determine paths
    if split == "train":
        raw_path = config.TRAIN_METADATA_PATH
        processed_path = config.TRAIN_PROCESSED_PATH
    elif split == "val":
        raw_path = config.VAL_METADATA_PATH
        processed_path = config.VAL_PROCESSED_PATH
    else:
        raw_path = config.TEST_METADATA_PATH
        processed_path = config.TEST_PROCESSED_PATH

    # Check cache
    if load_cached_data and os.path.exists(processed_path):
        print(f"Loading cached {split} data from {processed_path}...")
        df = pd.read_parquet(processed_path)

        # If train, we also need to load vocab and encoder if not provided
        if split == "train":
            if vocab is None and os.path.exists(config.VOCAB_PATH):
                vocab = Vocabulary()
                vocab.load(config.VOCAB_PATH)
            if tag_encoder is None and os.path.exists(config.TAG_MAP_PATH):
                tag_encoder = TagEncoder()
                tag_encoder.load(config.TAG_MAP_PATH)

        return df, vocab, tag_encoder

    # Process from scratch
    print(f"Processing {split} data from {raw_path}...")
    df = pd.read_csv(raw_path)

    if debug:
        print(f"DEBUG mode: Sampling {config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.head(config.DEBUG_SAMPLE_SIZE).copy()

    # Preprocess Text: Title + " " + Body
    print("Cleaning text...")
    # Vectorized cleaning is hard with complex regex, using apply
    # Combining first to save one apply call
    full_text = df["Title"].fillna("") + " " + df["Body"].fillna("")
    clean_text_series = full_text.apply(utils.clean_text)

    # Build Vocab and Encoder if Train
    if split == "train":
        if vocab is None:
            vocab = Vocabulary()
            vocab.build(clean_text_series.tolist())
            vocab.save(config.VOCAB_PATH)

        if tag_encoder is None:
            tag_encoder = TagEncoder()
            tag_encoder.build(df["Tags"].fillna("").tolist())
            tag_encoder.save(config.TAG_MAP_PATH)

    # Tokenize
    print("Tokenizing text...")
    df["input_ids"] = clean_text_series.apply(vocab.encode)

    # Encode Tags (if not test)
    if split != "test":
        print("Encoding tags...")
        df["tag_indices"] = df["Tags"].fillna("").apply(tag_encoder.encode)

    # Keep only necessary columns
    cols_to_keep = ["Id", "input_ids"]
    if split != "test":
        cols_to_keep.append("tag_indices")

    df_processed = df[cols_to_keep]

    # Save to parquet
    print(f"Saving processed data to {processed_path}...")
    df_processed.to_parquet(processed_path, index=False)

    return df_processed, vocab, tag_encoder


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main entry point to get DataLoaders for train, val, and test.
    """
    utils.set_seed(config.SEED)

    # 1. Prepare Train
    train_df, vocab, tag_encoder = prepare_data(
        "train",
        vocab=None,
        tag_encoder=None,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 2. Prepare Val
    val_df, _, _ = prepare_data(
        "val",
        vocab=vocab,
        tag_encoder=tag_encoder,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 3. Prepare Test
    test_df, _, _ = prepare_data(
        "test",
        vocab=vocab,
        tag_encoder=tag_encoder,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    print(f"Train set size: {len(train_df)}")
    print(f"Val set size:   {len(val_df)}")
    print(f"Test set size:  {len(test_df)}")

    # Create Datasets
    train_dataset = StackExchangeDataset(train_df, vocab, tag_encoder, is_test=False)
    val_dataset = StackExchangeDataset(val_df, vocab, tag_encoder, is_test=False)
    test_dataset = StackExchangeDataset(test_df, vocab, tag_encoder, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, vocab, tag_encoder
