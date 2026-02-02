import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    Dataset class for Tweet Sentiment Extraction.
    Serves pre-tokenized inputs and targets.
    """

    def __init__(
        self,
        input_ids,
        attention_masks,
        token_type_ids,
        offsets,
        texts,
        sentiments,
        selected_texts=None,
        targets=None,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.offsets = offsets
        self.texts = texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.attention_masks[item], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long),
            "text": str(self.texts[item]),
            "sentiment": str(self.sentiments[item]),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
        }

        # Include targets for training
        if self.targets is not None:
            data["start_targets"] = torch.tensor(
                self.targets[item][0], dtype=torch.long
            )
            data["end_targets"] = torch.tensor(self.targets[item][1], dtype=torch.long)

        # Include raw selected_text for validation metric calculation
        if self.selected_texts is not None:
            data["selected_text"] = str(self.selected_texts[item])

        return data


def _find_target_indices(text, selected_text, offsets, sequence_ids):
    """
    Maps the character offsets of selected_text to token indices.
    """
    # Robust finding of char indices
    idx_start = text.find(selected_text)
    if idx_start == -1:
        idx_start = 0
        idx_end = len(text)
    else:
        idx_end = idx_start + len(selected_text)

    tokens_start_idx = 0
    tokens_end_idx = 0
    found_start = False

    # Iterate through offsets to find the tokens corresponding to the character span
    # We only look at sequence_id == 1 (the tweet text part)
    for i, (off_start, off_end) in enumerate(offsets):
        if sequence_ids[i] != 1:
            continue

        # Check for overlap between token span and selected_text span
        # Overlap condition: max(token_start, target_start) < min(token_end, target_end)
        if max(off_start, idx_start) < min(off_end, idx_end):
            if not found_start:
                tokens_start_idx = i
                found_start = True
            tokens_end_idx = i

    # Fallback if no overlap found (should be rare with clean data)
    if not found_start:
        text_indices = [i for i, s in enumerate(sequence_ids) if s == 1]
        if text_indices:
            tokens_start_idx = text_indices[0]
            tokens_end_idx = text_indices[-1]

    return tokens_start_idx, tokens_end_idx


def process_data(tokenizer, mode="train", load_cached_data=True):
    """
    Loads data, performs tokenization, computes targets, and manages caching.

    Args:
        tokenizer: HuggingFace tokenizer.
        mode (str): 'train' or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        df, input_ids, attention_masks, token_type_ids, offsets, targets
    """
    # Define cache paths
    cache_dir = Config.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_npz = os.path.join(cache_dir, f"cached_{mode}_{Config.max_len}.npz")
    cache_path_pq = os.path.join(cache_dir, f"cached_{mode}_{Config.max_len}.parquet")

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(cache_path_npz)
        and os.path.exists(cache_path_pq)
    ):
        print(f"Loading cached {mode} data from {cache_dir}...")
        try:
            # Load arrays
            arrays = np.load(cache_path_npz)
            input_ids = arrays["input_ids"]
            attention_masks = arrays["attention_masks"]
            token_type_ids = arrays["token_type_ids"]
            offsets = arrays["offsets"]
            targets = arrays["targets"] if "targets" in arrays else None

            # Load dataframe
            df = pd.read_parquet(cache_path_pq)

            return df, input_ids, attention_masks, token_type_ids, offsets, targets
        except Exception as e:
            print(f"Cache loading failed ({e}). Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")

    if mode == "train":
        # Combine train and val metadata to create a full training set for CV
        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)
        df = pd.concat([df_train, df_val]).reset_index(drop=True)

        # Clean data
        df.dropna(subset=["text", "sentiment", "selected_text"], inplace=True)

        # Create Stratified Folds
        skf = StratifiedKFold(
            n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
        )
        df["kfold"] = -1
        for fold, (_, val_idx) in enumerate(skf.split(df, df["sentiment"])):
            df.loc[val_idx, "kfold"] = fold
    else:
        # Load test set
        df = pd.read_csv(Config.test_path)
        df["selected_text"] = df["text"]  # Placeholder for test
        if "sentiment" not in df.columns:
            df["sentiment"] = "neutral"  # Fallback

    # Pre-allocate arrays
    size = len(df)
    input_ids = np.zeros((size, Config.max_len), dtype=np.int32)
    attention_masks = np.zeros((size, Config.max_len), dtype=np.int32)
    token_type_ids = np.zeros((size, Config.max_len), dtype=np.int32)
    offsets = np.zeros((size, Config.max_len, 2), dtype=np.int32)
    targets = np.zeros((size, 2), dtype=np.int32) if mode == "train" else None

    # Processing Loop
    for i, row in enumerate(df.itertuples()):
        text = str(row.text).strip()
        sentiment = str(row.sentiment).strip()

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=Config.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids[i] = encoded["input_ids"]
        attention_masks[i] = encoded["attention_mask"]
        token_type_ids[i] = encoded["token_type_ids"]
        offsets[i] = encoded["offset_mapping"]

        if mode == "train":
            selected_text = str(row.selected_text).strip()
            start_idx, end_idx = _find_target_indices(
                text, selected_text, encoded["offset_mapping"], encoded.sequence_ids()
            )
            targets[i] = [start_idx, end_idx]

    # 3. Save to Cache
    # Save numeric arrays to npz
    save_kwargs = {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "token_type_ids": token_type_ids,
        "offsets": offsets,
    }
    if targets is not None:
        save_kwargs["targets"] = targets

    np.savez(cache_path_npz, **save_kwargs)

    # Save DataFrame to parquet (handles strings efficiently)
    df.to_parquet(cache_path_pq, index=False)

    print(f"Data processed and saved to {cache_dir}")

    return df, input_ids, attention_masks, token_type_ids, offsets, targets


def get_loaders(fold, tokenizer):
    """
    Returns train and validation DataLoaders for the specified fold.
    """
    # Load data
    df, input_ids, masks, token_types, offsets, targets = process_data(
        tokenizer, mode="train"
    )

    # Split based on fold
    train_idx = df[df["kfold"] != fold].index.values
    val_idx = df[df["kfold"] == fold].index.values

    # Create Datasets
    train_dataset = TweetDataset(
        input_ids[train_idx],
        masks[train_idx],
        token_types[train_idx],
        offsets[train_idx],
        df.iloc[train_idx]["text"].values,
        df.iloc[train_idx]["sentiment"].values,
        df.iloc[train_idx]["selected_text"].values,
        targets[train_idx],
    )

    val_dataset = TweetDataset(
        input_ids[val_idx],
        masks[val_idx],
        token_types[val_idx],
        offsets[val_idx],
        df.iloc[val_idx]["text"].values,
        df.iloc[val_idx]["sentiment"].values,
        df.iloc[val_idx]["selected_text"].values,
        targets[val_idx],
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def get_test_loader(tokenizer):
    """
    Returns the DataLoader for the test set.
    """
    df, input_ids, masks, token_types, offsets, _ = process_data(tokenizer, mode="test")

    dataset = TweetDataset(
        input_ids,
        masks,
        token_types,
        offsets,
        df["text"].values,
        df["sentiment"].values,
        selected_texts=None,
        targets=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader, df
