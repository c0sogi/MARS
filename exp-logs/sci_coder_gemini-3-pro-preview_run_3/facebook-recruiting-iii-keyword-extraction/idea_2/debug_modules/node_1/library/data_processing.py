import os
import re
import json
import torch
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed, save_numpy_cache, load_numpy_cache

# ==========================================
# 1. Text Processing & Vocabulary
# ==========================================


class Vocabulary:
    """
    Handles mapping between tokens and integer IDs.
    """

    def __init__(self, max_size=Config.VOCAB_SIZE, min_freq=Config.MIN_WORD_FREQ):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = {}
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.unk_index = 0
        self.pad_index = 1

    def fit(self, texts):
        """
        Builds vocabulary from a list of tokenized texts (list of lists of strings).
        """
        print("Building vocabulary...")
        counter = Counter()
        for tokens in texts:
            counter.update(tokens)

        # Filter by frequency and size
        most_common = counter.most_common(self.max_size - 2)  # Reserve 2 for unk, pad

        self.stoi = {self.unk_token: self.unk_index, self.pad_token: self.pad_index}
        self.itos = {self.unk_index: self.unk_token, self.pad_index: self.pad_token}

        for idx, (word, freq) in enumerate(most_common, start=2):
            if freq < self.min_freq:
                break
            self.stoi[word] = idx
            self.itos[idx] = word

        print(f"Vocabulary built. Size: {len(self.stoi)}")

    def transform(self, texts, max_len=Config.MAX_SEQ_LEN):
        """
        Converts list of texts to list of lists of IDs.
        """
        processed = []
        for tokens in texts:
            # Map words to IDs, use unk_index if not found
            ids = [self.stoi.get(t, self.unk_index) for t in tokens]
            # Truncate
            if len(ids) > max_len:
                ids = ids[:max_len]
            # Ensure at least one token (unk) if empty
            if not ids:
                ids = [self.unk_index]
            processed.append(ids)
        return processed

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "stoi": self.stoi,
                    "itos": {
                        k: v for k, v in self.itos.items()
                    },  # Ensure keys are strings for JSON
                    "params": {"max_size": self.max_size, "min_freq": self.min_freq},
                },
                f,
            )

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {int(k): v for k, v in data["itos"].items()}
            self.max_size = data["params"]["max_size"]
            self.min_freq = data["params"]["min_freq"]
            self.unk_index = self.stoi.get(self.unk_token, 0)


def preprocess_text(df):
    """
    Vectorized text preprocessing: Lowercase, remove non-alphanumeric, split.
    Returns a list of lists of tokens.
    """
    print("Preprocessing text...")
    # Fill NaNs
    df["Title"] = df["Title"].fillna("")
    df["Body"] = df["Body"].fillna("")

    # Concatenate
    text_series = df["Title"] + " " + df["Body"]

    # Lowercase
    text_series = text_series.str.lower()

    # Remove non-alphanumeric (keep spaces)
    # Note: This regex replaces anything that is NOT a-z or 0-9 with a space
    text_series = text_series.str.replace(r"[^a-z0-9]", " ", regex=True)

    # Split by whitespace
    tokens_list = text_series.str.split().tolist()

    return tokens_list


# ==========================================
# 2. Tag Encoding
# ==========================================


class TagEncoder:
    """
    Wraps MultiLabelBinarizer to encode tags.
    """

    def __init__(self, top_k=Config.NUM_TAGS):
        self.top_k = top_k
        self.classes_ = None
        self.tag_to_idx = {}

    def fit(self, tags_series):
        """
        Identifies top K tags and creates mapping.
        tags_series: pandas Series of space-separated tag strings.
        """
        print("Fitting TagEncoder...")
        # Split all tags
        all_tags = [tag for sublist in tags_series.str.split() for tag in sublist]
        counter = Counter(all_tags)

        # Select top K
        most_common = counter.most_common(self.top_k)
        self.classes_ = [t[0] for t in most_common]
        self.classes_.sort()  # Deterministic order

        self.tag_to_idx = {tag: i for i, tag in enumerate(self.classes_)}
        print(f"TagEncoder fitted. {len(self.classes_)} classes.")

    def transform(self, tags_series):
        """
        Converts tags to binary matrix (dense int8).
        """
        n_samples = len(tags_series)
        n_classes = len(self.classes_)

        # Create dense matrix
        y = np.zeros((n_samples, n_classes), dtype=np.int8)

        for idx, tag_string in enumerate(tags_series):
            if not isinstance(tag_string, str):
                continue
            tags = tag_string.split()
            for tag in tags:
                if tag in self.tag_to_idx:
                    col_idx = self.tag_to_idx[tag]
                    y[idx, col_idx] = 1
        return y

    def inverse_transform(self, binary_matrix):
        """
        Converts binary matrix back to list of lists of tags.
        """
        result = []
        for row in binary_matrix:
            tags = [self.classes_[i] for i, val in enumerate(row) if val == 1]
            result.append(tuple(tags))
        return result


# ==========================================
# 3. Dataset & Collate
# ==========================================


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset backed by numpy arrays.
    """

    def __init__(self, tokens, offsets, labels=None, ids=None):
        """
        tokens: 1D numpy array of all tokens flattened.
        offsets: 1D numpy array of start indices for each sample.
        labels: 2D numpy array of binary labels (optional).
        ids: 1D numpy array of sample IDs (optional).
        """
        self.tokens = tokens
        self.offsets = offsets
        self.labels = labels
        self.ids = ids
        self.length = len(offsets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Determine start and end indices in the flat token array
        start = self.offsets[idx]
        if idx + 1 < self.length:
            end = self.offsets[idx + 1]
        else:
            end = len(self.tokens)

        token_seq = self.tokens[start:end]

        # Convert to tensor
        token_tensor = torch.tensor(token_seq, dtype=torch.long)

        if self.labels is not None:
            # Convert int8 label to float32 for BCEWithLogitsLoss
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return token_tensor, label_tensor

        if self.ids is not None:
            return token_tensor, self.ids[idx]

        return token_tensor


def collate_fn(batch):
    """
    Custom collate function for EmbeddingBag.
    Batch is a list of tuples: (token_tensor, label_tensor) or (token_tensor, id).
    """
    # Unzip the batch
    if isinstance(batch[0], tuple):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
    else:
        inputs = batch
        targets = None

    # 1. Flatten inputs for EmbeddingBag
    # Concatenate all sequences into a single 1D tensor
    flat_inputs = torch.cat(inputs)

    # 2. Calculate offsets
    # Offsets are the starting index of each sequence in the flat_inputs tensor
    offsets = [0]
    for i in range(len(inputs) - 1):
        offsets.append(offsets[-1] + len(inputs[i]))
    offsets = torch.tensor(offsets, dtype=torch.long)

    # 3. Stack targets
    if targets is not None:
        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets)
        else:
            # For IDs (integers)
            targets = torch.tensor(targets, dtype=torch.long)
        return flat_inputs, offsets, targets

    return flat_inputs, offsets


# ==========================================
# 4. Data Preparation Pipeline
# ==========================================


def load_and_merge(meta_path, raw_path, use_tags=True):
    """
    Loads metadata and merges with raw data.
    """
    print(f"Loading metadata from {meta_path}...")
    df_meta = pd.read_csv(meta_path)

    print(f"Loading raw data from {raw_path}...")
    cols = ["Id", "Title", "Body"]
    # Cite debug_lesson_1: Prevent column collision by only loading Tags if not in metadata
    if use_tags and "Tags" not in df_meta.columns:
        cols.append("Tags")

    # Read raw data
    # Note: In a real scenario with files larger than RAM, we'd process in chunks.
    # Here we assume 220GB is enough for 5M rows.
    df_raw = pd.read_csv(raw_path, usecols=cols)

    # Merge
    print("Merging data...")
    df = pd.merge(df_meta, df_raw, on="Id", how="inner")

    return df


def process_split(df, vocab, tag_encoder=None, is_test=False):
    """
    Converts DataFrame into numpy arrays (tokens, offsets, labels).
    """
    # 1. Text to Tokens
    tokens_list = preprocess_text(df)

    # 2. Tokens to IDs
    print("Mapping tokens to IDs...")
    ids_list = vocab.transform(tokens_list)

    # 3. Flatten and Offsets
    print("Flattening arrays...")
    # Calculate offsets
    lengths = [len(x) for x in ids_list]
    offsets = np.zeros(len(lengths), dtype=np.int64)
    current_offset = 0
    for i, length in enumerate(lengths):
        offsets[i] = current_offset
        current_offset += length

    # Flatten tokens
    flat_tokens = np.array([t for sublist in ids_list for t in sublist], dtype=np.int32)

    # 4. Process Labels/IDs
    labels = None
    ids = df["Id"].values.astype(np.int64)

    if not is_test and tag_encoder is not None:
        print("Encoding labels...")
        labels = tag_encoder.transform(df["Tags"].fillna(""))

    return flat_tokens, offsets, labels, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=True
):
    """
    Main entry point. Loads data, processes it (or loads cache), and returns DataLoaders.
    """
    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.TRAIN_TOKENS_PATH)
        and os.path.exists(Config.TRAIN_LABELS_PATH)
        and os.path.exists(Config.VOCAB_PATH)
        and os.path.exists(Config.MLB_PATH)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        train_tokens = load_numpy_cache(Config.TRAIN_TOKENS_PATH)
        train_offsets = load_numpy_cache(Config.TRAIN_OFFSETS_PATH)
        train_labels = load_numpy_cache(Config.TRAIN_LABELS_PATH)

        val_tokens = load_numpy_cache(Config.VAL_TOKENS_PATH)
        val_offsets = load_numpy_cache(Config.VAL_OFFSETS_PATH)
        val_labels = load_numpy_cache(Config.VAL_LABELS_PATH)

        test_tokens = load_numpy_cache(Config.TEST_TOKENS_PATH)
        test_offsets = load_numpy_cache(Config.TEST_OFFSETS_PATH)
        test_ids = load_numpy_cache(Config.TEST_IDS_PATH)

        mlb = joblib.load(Config.MLB_PATH)

    else:
        print("Processing data from scratch...")

        # --- Train ---
        df_train = load_and_merge(
            Config.TRAIN_META_PATH, os.path.join(Config.INPUT_DIR, "train.csv")
        )
        if debug:
            df_train = df_train.sample(
                n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            )

        # Build Vocab
        tokens_list_train = preprocess_text(df_train)  # Get tokens for vocab building
        vocab = Vocabulary()
        vocab.fit(tokens_list_train)
        vocab.save(Config.VOCAB_PATH)

        # Build Tag Encoder
        mlb = TagEncoder()
        mlb.fit(df_train["Tags"].fillna(""))
        joblib.dump(mlb, Config.MLB_PATH)

        # Process Train (re-using logic, though we double-tokenized train for vocab, it's safer to standardize)
        # To save memory, we can reuse tokens_list_train but we need to map it.
        print("Transforming Train...")
        train_ids_list = vocab.transform(tokens_list_train)

        # Flatten Train
        train_lens = [len(x) for x in train_ids_list]
        train_offsets = np.zeros(len(train_lens), dtype=np.int64)
        curr = 0
        for i, l in enumerate(train_lens):
            train_offsets[i] = curr
            curr += l
        train_tokens = np.array(
            [t for sub in train_ids_list for t in sub], dtype=np.int32
        )
        train_labels = mlb.transform(df_train["Tags"].fillna(""))

        # Save Train
        save_numpy_cache(train_tokens, Config.TRAIN_TOKENS_PATH)
        save_numpy_cache(train_offsets, Config.TRAIN_OFFSETS_PATH)
        save_numpy_cache(train_labels, Config.TRAIN_LABELS_PATH)

        del df_train, tokens_list_train, train_ids_list
        import gc

        gc.collect()

        # --- Val ---
        df_val = load_and_merge(
            Config.VAL_META_PATH, os.path.join(Config.INPUT_DIR, "train.csv")
        )
        if debug:
            df_val = df_val.sample(
                n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            )

        val_tokens, val_offsets, val_labels, _ = process_split(
            df_val, vocab, mlb, is_test=False
        )
        save_numpy_cache(val_tokens, Config.VAL_TOKENS_PATH)
        save_numpy_cache(val_offsets, Config.VAL_OFFSETS_PATH)
        save_numpy_cache(val_labels, Config.VAL_LABELS_PATH)

        del df_val
        gc.collect()

        # --- Test ---
        df_test = load_and_merge(
            Config.TEST_META_PATH,
            os.path.join(Config.INPUT_DIR, "test.csv"),
            use_tags=False,
        )
        if debug:
            df_test = df_test.sample(
                n=min(len(df_test), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            )

        test_tokens, test_offsets, _, test_ids = process_split(
            df_test, vocab, is_test=True
        )
        save_numpy_cache(test_tokens, Config.TEST_TOKENS_PATH)
        save_numpy_cache(test_offsets, Config.TEST_OFFSETS_PATH)
        save_numpy_cache(test_ids, Config.TEST_IDS_PATH)

        del df_test
        gc.collect()

    print("Data loaded. Creating DataLoaders...")

    train_dataset = StackExchangeDataset(train_tokens, train_offsets, train_labels)
    val_dataset = StackExchangeDataset(val_tokens, val_offsets, val_labels)
    test_dataset = StackExchangeDataset(test_tokens, test_offsets, ids=test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, mlb
