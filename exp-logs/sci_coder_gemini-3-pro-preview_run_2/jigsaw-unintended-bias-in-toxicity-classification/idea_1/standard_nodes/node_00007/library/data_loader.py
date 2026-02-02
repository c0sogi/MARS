import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TEXT_PATH,
    TEST_TEXT_PATH,
    TOKENIZER_SAVE_PATH,
    ID_COL,
    TEXT_COL,
    TARGET_COL,
    IDENTITY_COLUMNS,
    VOCAB_SIZE,
    MAX_LEN,
    BATCH_SIZE,
    NUM_WORKERS,
    LOWERCASE,
)


class Tokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE, max_len=MAX_LEN, lowercase=LOWERCASE):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.lowercase = lowercase
        self.word_index = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            if self.lowercase:
                text = text.lower()
            counter.update(text.split())

        # Reserve 0 for PAD, 1 for UNK. Keep top (vocab_size - 2) words.
        most_common = counter.most_common(self.vocab_size - 2)

        self.word_index = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        for i, (word, _) in enumerate(most_common):
            self.word_index[word] = i + 2

    def transform(self, texts):
        sequences = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            if self.lowercase:
                text = text.lower()

            words = text.split()
            # Map words to indices, default to UNK
            seq = [self.word_index.get(w, self.unk_idx) for w in words]

            # Truncate only (Dynamic padding handled in collate_fn)
            if len(seq) > self.max_len:
                seq = seq[: self.max_len]

            sequences.append(seq)
        # Return object array to support variable lengths
        return np.array(sequences, dtype=object)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word_index, f)

    def load(self, path):
        with open(path, "r") as f:
            self.word_index = json.load(f)


class JigsawDataset(Dataset):
    def __init__(self, X, y=None, aux=None, ids=None):
        self.X = X
        self.y = y
        self.aux = aux
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"input_ids": torch.tensor(self.X[idx], dtype=torch.long)}

        if self.y is not None:
            item["target"] = torch.tensor(self.y[idx], dtype=torch.float)

        if self.aux is not None:
            item["aux_target"] = torch.tensor(self.aux[idx], dtype=torch.float)

        if self.ids is not None:
            item["id"] = torch.tensor(self.ids[idx], dtype=torch.long)

        return item


def collate_dynamic(batch):
    """
    Custom collate function to dynamically pad the batch to the maximum sequence length.
    Cite solution_lesson_node_00006
    """
    input_ids = [item["input_ids"] for item in batch]
    # Pad sequences with 0 (PAD token)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)

    batch_out = {"input_ids": input_ids_padded}

    if "target" in batch[0]:
        batch_out["target"] = torch.stack([item["target"] for item in batch])

    if "aux_target" in batch[0]:
        batch_out["aux_target"] = torch.stack([item["aux_target"] for item in batch])

    if "id" in batch[0]:
        batch_out["id"] = torch.stack([item["id"] for item in batch])

    return batch_out


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Loads data, processes it (or loads from cache), and returns PyTorch DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        debug (bool): If True, restricts datasets to a small subset for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    files = {
        "train_X": os.path.join(CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(CACHE_DIR, "train_y.npy"),
        "train_aux": os.path.join(CACHE_DIR, "train_aux.npy"),
        "val_X": os.path.join(CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(CACHE_DIR, "val_y.npy"),
        "val_aux": os.path.join(CACHE_DIR, "val_aux.npy"),
        "test_X": os.path.join(CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Check if we can load from cache
    cache_exists = all(os.path.exists(p) for p in files.values()) and os.path.exists(
        TOKENIZER_SAVE_PATH
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        # Allow pickle for object arrays (variable length sequences)
        train_X = np.load(files["train_X"], allow_pickle=True)
        train_y = np.load(files["train_y"])
        train_aux = np.load(files["train_aux"])

        val_X = np.load(files["val_X"], allow_pickle=True)
        val_y = np.load(files["val_y"])
        val_aux = np.load(files["val_aux"])

        test_X = np.load(files["test_X"], allow_pickle=True)
        test_ids = np.load(files["test_ids"])

        # We don't strictly need to load the tokenizer object to return dataloaders,
        # but it's good practice to verify it exists.

    else:
        print("Processing data from scratch...")

        # 1. Load Metadata
        print("Loading metadata...")
        meta_train = pd.read_csv(TRAIN_METADATA_PATH)
        meta_val = pd.read_csv(VAL_METADATA_PATH)
        meta_test = pd.read_csv(TEST_METADATA_PATH)

        # 2. Load Raw Text
        # We only need ID and Text columns.
        print("Loading raw text files...")
        train_text_df = pd.read_csv(TRAIN_TEXT_PATH, usecols=[ID_COL, TEXT_COL])
        test_text_df = pd.read_csv(TEST_TEXT_PATH, usecols=[ID_COL, TEXT_COL])

        # 3. Merge Metadata with Text
        print("Merging metadata with text...")
        # Train and Val come from input/train.csv
        df_train = meta_train.merge(train_text_df, on=ID_COL, how="left")
        df_val = meta_val.merge(train_text_df, on=ID_COL, how="left")
        # Test comes from input/test.csv
        df_test = meta_test.merge(test_text_df, on=ID_COL, how="left")

        # Handle missing text
        df_train[TEXT_COL] = df_train[TEXT_COL].fillna("").astype(str)
        df_val[TEXT_COL] = df_val[TEXT_COL].fillna("").astype(str)
        df_test[TEXT_COL] = df_test[TEXT_COL].fillna("").astype(str)

        # 4. Tokenization
        print("Fitting tokenizer...")
        tokenizer = Tokenizer(
            vocab_size=VOCAB_SIZE, max_len=MAX_LEN, lowercase=LOWERCASE
        )
        tokenizer.fit(df_train[TEXT_COL].values)
        tokenizer.save(TOKENIZER_SAVE_PATH)

        print("Transforming text to sequences...")
        train_X = tokenizer.transform(df_train[TEXT_COL].values)
        val_X = tokenizer.transform(df_val[TEXT_COL].values)
        test_X = tokenizer.transform(df_test[TEXT_COL].values)

        # 5. Extract Targets
        print("Extracting targets...")
        train_y = df_train[TARGET_COL].values.astype(np.float32)
        val_y = df_val[TARGET_COL].values.astype(np.float32)

        # Extract Auxiliary Identity Targets (Binary: 1 if >= 0.5 else 0)
        # Fill NaNs with 0 before thresholding
        train_aux = (df_train[IDENTITY_COLUMNS].fillna(0).values >= 0.5).astype(
            np.float32
        )
        val_aux = (df_val[IDENTITY_COLUMNS].fillna(0).values >= 0.5).astype(np.float32)

        test_ids = df_test[ID_COL].values

        # 6. Save to Cache
        print("Saving to cache...")
        np.save(files["train_X"], train_X)
        np.save(files["train_y"], train_y)
        np.save(files["train_aux"], train_aux)

        np.save(files["val_X"], val_X)
        np.save(files["val_y"], val_y)
        np.save(files["val_aux"], val_aux)

        np.save(files["test_X"], test_X)
        np.save(files["test_ids"], test_ids)

    # Debug Mode: Slice data
    if debug:
        print("Debug mode: slicing datasets to 2000 samples.")
        limit = 2000
        train_X = train_X[:limit]
        train_y = train_y[:limit]
        train_aux = train_aux[:limit]

        val_X = val_X[:limit]
        val_y = val_y[:limit]
        val_aux = val_aux[:limit]

        test_X = test_X[:limit]
        test_ids = test_ids[:limit]

    # Create Datasets
    train_dataset = JigsawDataset(train_X, train_y, train_aux)
    val_dataset = JigsawDataset(val_X, val_y, val_aux)
    test_dataset = JigsawDataset(test_X, ids=test_ids)

    # Create DataLoaders with dynamic padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_dynamic,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_dynamic,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_dynamic,
    )

    print(
        f"DataLoaders ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
