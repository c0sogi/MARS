import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

from library.config import CFG
from library.utils import get_logger


def get_meta_features(text):
    """
    Calculates scalar features: char_count, word_count, sentence_count, unique_word_ratio.
    Uses regex for sentence splitting to avoid dependency on nltk data downloads.
    """
    char_count = len(text)
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        unique_word_ratio = 0.0
        sentence_count = 0
    else:
        unique_word_ratio = len(set(words)) / word_count
        # Simple regex for sentence splitting (periods, exclamations, question marks)
        sentence_count = len(re.findall(r"[.!?]+", text))
        if sentence_count == 0 and word_count > 0:
            sentence_count = 1  # Fallback for texts without punctuation

    return [char_count, word_count, sentence_count, unique_word_ratio]


def process_data(df_path, load_cached_data=True, cache_name="data"):
    """
    Loads raw data, computes meta-features, and caches the result.
    """
    os.makedirs(CFG.cache_dir, exist_ok=True)
    cache_file = os.path.join(CFG.cache_dir, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_parquet(cache_file)
    else:
        print(f"Processing data from {df_path}...")
        df = pd.read_csv(df_path)

        # Calculate meta-features
        meta_data = []
        texts = df["full_text"].astype(str).tolist()
        for text in texts:
            meta_data.append(get_meta_features(text))

        meta_df = pd.DataFrame(
            meta_data,
            columns=["char_count", "word_count", "sentence_count", "unique_word_ratio"],
        )
        df = pd.concat([df, meta_df], axis=1)

        print(f"Saving processed data to {cache_file}")
        df.to_parquet(cache_file, index=False)

    return df


class EssayDataset(Dataset):
    """
    Dataset that handles sliding window tokenization for DeBERTa.
    Returns:
        dict containing:
            - input_ids: (num_chunks, max_len)
            - attention_mask: (num_chunks, max_len)
            - meta_features: (4,)
            - labels: (1,) [if train]
    """

    def __init__(self, df, tokenizer, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.full_texts = df["full_text"].values
        # Ensure meta features are float32
        self.meta_features = df[
            ["char_count", "word_count", "sentence_count", "unique_word_ratio"]
        ].values.astype(np.float32)

        if self.is_train:
            self.scores = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.full_texts[idx]
        meta = self.meta_features[idx]

        # Tokenize full text first without special tokens
        tokens = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=False
        )
        input_ids = tokens["input_ids"]

        # Sliding Window Logic
        window_size = CFG.max_len - 2  # Reserve space for [CLS] and [SEP]
        stride = CFG.stride

        all_input_ids = []
        all_attention_mask = []

        # Handle short texts (single chunk)
        if len(input_ids) <= window_size:
            chunk_ids = (
                [self.tokenizer.cls_token_id]
                + input_ids
                + [self.tokenizer.sep_token_id]
            )
            chunk_mask = [1] * len(chunk_ids)

            # Pad
            padding_len = CFG.max_len - len(chunk_ids)
            if padding_len > 0:
                chunk_ids += [self.tokenizer.pad_token_id] * padding_len
                chunk_mask += [0] * padding_len

            all_input_ids.append(chunk_ids)
            all_attention_mask.append(chunk_mask)

        else:
            # Handle long texts (multiple chunks)
            for i in range(0, len(input_ids), stride):
                chunk = input_ids[i : i + window_size]

                chunk_ids = (
                    [self.tokenizer.cls_token_id]
                    + chunk
                    + [self.tokenizer.sep_token_id]
                )
                chunk_mask = [1] * len(chunk_ids)

                # Pad
                padding_len = CFG.max_len - len(chunk_ids)
                if padding_len > 0:
                    chunk_ids += [self.tokenizer.pad_token_id] * padding_len
                    chunk_mask += [0] * padding_len

                all_input_ids.append(chunk_ids)
                all_attention_mask.append(chunk_mask)

                # Break if we've processed the end of the text
                if i + window_size >= len(input_ids):
                    break

        # Stack chunks into tensors
        data = {
            "input_ids": torch.tensor(
                all_input_ids, dtype=torch.long
            ),  # (n_chunks, 512)
            "attention_mask": torch.tensor(
                all_attention_mask, dtype=torch.long
            ),  # (n_chunks, 512)
            "meta_features": torch.tensor(meta, dtype=torch.float32),
        }

        if self.is_train:
            data["labels"] = torch.tensor(self.scores[idx], dtype=torch.float32)

        return data


class Collate:
    """
    Custom collate function to handle variable number of chunks per essay.
    Pads the chunk dimension to the maximum number of chunks in the batch.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Determine max chunks in this batch
        max_chunks = max([item["input_ids"].size(0) for item in batch])

        batch_input_ids = []
        batch_attention_mask = []
        batch_chunk_mask = []  # Mask to identify valid chunks vs padded chunks
        batch_meta = []
        batch_labels = []

        for item in batch:
            n_chunks = item["input_ids"].size(0)

            # Pad chunks if necessary
            if n_chunks < max_chunks:
                pad_count = max_chunks - n_chunks

                # Create padding tensors
                pad_input = torch.full(
                    (pad_count, CFG.max_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                pad_mask = torch.full((pad_count, CFG.max_len), 0, dtype=torch.long)

                # Concatenate
                input_ids = torch.cat([item["input_ids"], pad_input], dim=0)
                attention_mask = torch.cat([item["attention_mask"], pad_mask], dim=0)

                # Chunk mask: 1 for valid chunks, 0 for padded chunks
                chunk_mask = torch.cat(
                    [torch.ones(n_chunks), torch.zeros(pad_count)], dim=0
                )
            else:
                input_ids = item["input_ids"]
                attention_mask = item["attention_mask"]
                chunk_mask = torch.ones(n_chunks)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_chunk_mask.append(chunk_mask)
            batch_meta.append(item["meta_features"])

            if "labels" in item:
                batch_labels.append(item["labels"])

        # Stack into batch tensors
        # Output Shape: (batch_size, max_chunks, seq_len)
        out = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "chunk_mask": torch.stack(batch_chunk_mask).bool(),
            "meta_features": torch.stack(batch_meta),
        }

        if len(batch_labels) > 0:
            out["labels"] = torch.stack(batch_labels)

        return out


def get_dataloaders(fold=0, load_cached_data=True):
    """
    Creates DataLoaders for the specified fold using the full dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # 1. Load and Process Training Data (Full train.csv)
    # We use the full dataset and split it manually to ensure 5-fold coverage
    train_df = process_data(
        CFG.train_path, load_cached_data=load_cached_data, cache_name="train_processed"
    )

    # Create Stratified Folds
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    train_df["fold"] = -1
    for f, (t_idx, v_idx) in enumerate(skf.split(train_df, train_df["score"])):
        train_df.loc[v_idx, "fold"] = f

    # Split into Train/Val for the requested fold
    trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

    # Debugging subset
    if CFG.debug:
        trn_df = trn_df.head(CFG.debug_subset_size)
        val_df = val_df.head(CFG.debug_subset_size)
        print(
            f"DEBUG MODE: Reduced train size to {len(trn_df)}, val size to {len(val_df)}"
        )

    # 2. Load and Process Test Data
    test_df = process_data(
        CFG.test_path, load_cached_data=load_cached_data, cache_name="test_processed"
    )

    # 3. Create Datasets
    train_dataset = EssayDataset(trn_df, tokenizer, is_train=True)
    val_dataset = EssayDataset(val_df, tokenizer, is_train=True)
    test_dataset = EssayDataset(test_df, tokenizer, is_train=False)

    # 4. Create DataLoaders
    collate_fn = Collate(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
