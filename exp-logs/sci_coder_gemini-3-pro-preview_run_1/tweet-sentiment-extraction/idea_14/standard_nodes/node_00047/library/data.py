import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from library.config import Config
from library.utils import normalize_text


class TweetDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        start_targets,
        end_targets,
        offsets,
        texts,
        sentiments,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets
        self.texts = texts
        self.sentiments = sentiments

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "text": str(self.texts[item]),
            "sentiment": str(self.sentiments[item]),
        }

        if self.start_targets is not None:
            data["start_targets"] = torch.tensor(
                self.start_targets[item], dtype=torch.float
            )
            data["end_targets"] = torch.tensor(
                self.end_targets[item], dtype=torch.float
            )

        return data


def get_gaussian_distribution(target_idx, max_len, sigma):
    if target_idx < 0 or target_idx >= max_len:
        # Fallback for invalid indices, though shouldn't happen with correct logic
        dist = np.zeros(max_len)
        return dist

    x = np.arange(max_len)
    dist = np.exp(-0.5 * ((x - target_idx) / sigma) ** 2)
    # Normalize to sum to 1
    dist_sum = dist.sum()
    if dist_sum > 0:
        dist = dist / dist_sum
    return dist


def process_data(df, tokenizer, max_len, sigma, is_train=True):
    n_samples = len(df)

    # Initialize arrays
    input_ids = np.zeros((n_samples, max_len), dtype=np.int32)
    attention_mask = np.zeros((n_samples, max_len), dtype=np.int32)
    offsets = np.zeros((n_samples, max_len, 2), dtype=np.int32)
    start_targets = np.zeros((n_samples, max_len), dtype=np.float32)
    end_targets = np.zeros((n_samples, max_len), dtype=np.float32)

    texts = []
    sentiments = []

    for idx, row in tqdm(
        enumerate(df.itertuples()), total=n_samples, desc="Processing Data"
    ):
        # Normalize text and sentiment
        text = normalize_text(str(row.text))
        sentiment = str(row.sentiment)

        texts.append(text)
        sentiments.append(sentiment)

        # Tokenize: [CLS] Sentiment [SEP] Text [SEP]
        # We use encode_plus to handle special tokens and offsets
        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        input_ids[idx] = encoded["input_ids"]
        attention_mask[idx] = encoded["attention_mask"]
        # Convert offsets to numpy
        current_offsets = np.array(encoded["offset_mapping"])
        offsets[idx] = current_offsets

        # Identify sequence IDs to distinguish sentiment tokens from text tokens
        # sequence_ids: None (special), 0 (sentiment), 1 (text)
        sequence_ids = encoded.sequence_ids()

        if is_train:
            selected_text = normalize_text(str(row.selected_text))

            # Find character indices of selected_text in text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: use entire text if exact match fails (rare with normalize-first)
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # Find token indices
            token_start_index = 0
            token_end_index = 0
            found_start = False
            found_end = False

            # Iterate through tokens to find the ones containing the start/end chars
            # We only look at tokens belonging to the text (sequence_id == 1)
            for i, seq_id in enumerate(sequence_ids):
                if seq_id != 1:
                    continue

                # offsets[i] is (start_char_idx, end_char_idx)
                off_start, off_end = current_offsets[i]

                # Check for start token: contains the first character of selected_text
                if not found_start and off_start <= start_char < off_end:
                    token_start_index = i
                    found_start = True

                # Check for end token: contains the last character of selected_text
                # Note: end_char is exclusive, so last char is end_char - 1
                if off_start <= (end_char - 1) < off_end:
                    token_end_index = i
                    found_end = True

            # If not found (e.g. empty selected_text or other edge case), default to 0
            # Ideally should not happen with cleaned data
            if not found_start:
                token_start_index = 0
            if not found_end:
                token_end_index = token_start_index  # Point to same token

            # Generate Gaussian Soft Targets
            start_targets[idx] = get_gaussian_distribution(
                token_start_index, max_len, sigma
            )
            end_targets[idx] = get_gaussian_distribution(
                token_end_index, max_len, sigma
            )

    return (
        input_ids,
        attention_mask,
        start_targets,
        end_targets,
        offsets,
        texts,
        sentiments,
    )


def get_loaders(tokenizer, load_cached_data=True):
    """
    Loads training and validation data, filters neutrals, processes/caches tensors,
    and returns DataLoaders.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # File names for cache
    files = {
        "train": {
            "input_ids": os.path.join(cache_dir, "cached_train_input_ids.npy"),
            "attention_mask": os.path.join(
                cache_dir, "cached_train_attention_mask.npy"
            ),
            "start_targets": os.path.join(cache_dir, "cached_train_start_targets.npy"),
            "end_targets": os.path.join(cache_dir, "cached_train_end_targets.npy"),
            "offsets": os.path.join(cache_dir, "cached_train_offsets.npy"),
            "meta": os.path.join(cache_dir, "cached_train_meta.parquet"),
        },
        "val": {
            "input_ids": os.path.join(cache_dir, "cached_val_input_ids.npy"),
            "attention_mask": os.path.join(cache_dir, "cached_val_attention_mask.npy"),
            "start_targets": os.path.join(cache_dir, "cached_val_start_targets.npy"),
            "end_targets": os.path.join(cache_dir, "cached_val_end_targets.npy"),
            "offsets": os.path.join(cache_dir, "cached_val_offsets.npy"),
            "meta": os.path.join(cache_dir, "cached_val_meta.parquet"),
        },
    }

    loaders = {}

    for split, path_dict in files.items():
        # Check if cache exists
        cache_exists = all(os.path.exists(p) for p in path_dict.values())

        if load_cached_data and cache_exists:
            print(f"Loading {split} data from cache...")
            input_ids = np.load(path_dict["input_ids"])
            attention_mask = np.load(path_dict["attention_mask"])
            start_targets = np.load(path_dict["start_targets"])
            end_targets = np.load(path_dict["end_targets"])
            offsets = np.load(path_dict["offsets"])

            df_meta = pd.read_parquet(path_dict["meta"])
            texts = df_meta["text"].tolist()
            sentiments = df_meta["sentiment"].tolist()

        else:
            print(f"Processing {split} data from scratch...")
            # Load original metadata
            if split == "train":
                df = pd.read_csv(Config.TRAIN_META_PATH)
            else:
                df = pd.read_csv(Config.VAL_META_PATH)

            # Filter out neutral tweets for training/validation loop
            initial_len = len(df)
            df = df[df["sentiment"] != "neutral"].reset_index(drop=True)
            print(f"Filtered {initial_len - len(df)} neutral samples from {split} set.")

            # Process
            (
                input_ids,
                attention_mask,
                start_targets,
                end_targets,
                offsets,
                texts,
                sentiments,
            ) = process_data(
                df, tokenizer, Config.max_len, Config.smoothing_sigma, is_train=True
            )

            # Save to cache
            np.save(path_dict["input_ids"], input_ids)
            np.save(path_dict["attention_mask"], attention_mask)
            np.save(path_dict["start_targets"], start_targets)
            np.save(path_dict["end_targets"], end_targets)
            np.save(path_dict["offsets"], offsets)

            # Save metadata for alignment
            pd.DataFrame({"text": texts, "sentiment": sentiments}).to_parquet(
                path_dict["meta"]
            )

        # Create Dataset
        dataset = TweetDataset(
            input_ids,
            attention_mask,
            start_targets,
            end_targets,
            offsets,
            texts,
            sentiments,
        )

        # Create DataLoader
        batch_size = (
            Config.train_batch_size if split == "train" else Config.valid_batch_size
        )
        shuffle = split == "train"

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"]


def get_test_loader(tokenizer):
    """
    Generates DataLoader for the test set.
    Does NOT filter neutrals (inference needs all rows).
    Does NOT generate targets.
    """
    df = pd.read_csv(Config.TEST_META_PATH)

    # We use the same process function but ignore targets
    # Note: process_data calculates targets if is_train=True.
    # We set is_train=False to skip target calculation.

    input_ids, attention_mask, _, _, offsets, texts, sentiments = process_data(
        df, tokenizer, Config.max_len, Config.smoothing_sigma, is_train=False
    )

    # Create Dataset (targets are None)
    dataset = TweetDataset(
        input_ids, attention_mask, None, None, offsets, texts, sentiments
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return loader
