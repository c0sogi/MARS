import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        start_labels=None,
        end_labels=None,
        token_type_ids=None,
        orig_texts=None,
        offsets=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.token_type_ids = token_type_ids
        self.orig_texts = orig_texts
        self.offsets = offsets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.token_type_ids is not None:
            item["token_type_ids"] = torch.tensor(
                self.token_type_ids[idx], dtype=torch.long
            )

        if self.start_labels is not None:
            item["start_labels"] = torch.tensor(
                self.start_labels[idx], dtype=torch.long
            )
            item["end_labels"] = torch.tensor(self.end_labels[idx], dtype=torch.long)

        if self.orig_texts is not None:
            item["orig_text"] = self.orig_texts[idx]

        if self.offsets is not None:
            item["offsets"] = torch.tensor(self.offsets[idx], dtype=torch.long)

        return item


def _get_cache_path(prefix):
    return os.path.join(Config.WORKING_DIR, f"{prefix}_cache.npz")


def process_data(
    df, tokenizer, max_len, is_test=False, cache_prefix="train", load_cached_data=True
):
    """
    Tokenizes data and generates targets using Mask-Based Overlap.
    Handles caching to disk.
    """
    cache_path = _get_cache_path(cache_prefix)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            results = {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
            }
            if "start_labels" in data:
                results["start_labels"] = data["start_labels"]
                results["end_labels"] = data["end_labels"]
            if "orig_texts" in data:
                results["orig_texts"] = data["orig_texts"]
            if "offsets" in data:
                results["offsets"] = data["offsets"]
            return results
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process Data
    print(f"Processing {len(df)} rows for {cache_prefix}...")

    input_ids_list = []
    attention_mask_list = []
    start_labels_list = []
    end_labels_list = []
    offsets_list = []
    orig_texts_list = []

    # Pre-compute special token IDs
    # Format: [CLS] sentiment [SEP] text [SEP]
    # Note: DeBERTa V3 tokenizer behavior check
    # We manually construct to ensure exact control
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        # Tokenize sentiment
        sentiment_ids = tokenizer.encode(sentiment, add_special_tokens=False)

        # Tokenize text with offsets
        encoded_text = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        text_ids = encoded_text["input_ids"]
        text_offsets = encoded_text["offset_mapping"]

        # Construct Input
        # [CLS] sentiment [SEP] text [SEP]
        input_ids = (
            [cls_token_id] + sentiment_ids + [sep_token_id] + text_ids + [sep_token_id]
        )

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Create offsets for the whole sequence
        # (0,0) for special tokens and sentiment
        prefix_len = 1 + len(sentiment_ids) + 1  # [CLS] sent [SEP]
        full_offsets = [(0, 0)] * prefix_len + text_offsets + [(0, 0)]

        # Padding
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            full_offsets = full_offsets + [(0, 0)] * padding_length
        else:
            # Truncate (should be rare given max_len=128 and tweet length)
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            full_offsets = full_offsets[:max_len]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        offsets_list.append(full_offsets)
        orig_texts_list.append(text)

        # Target Generation (Train only)
        if not is_test:
            selected_text = str(row["selected_text"])

            # Find character start/end of selected_text in text
            # We use find, assuming clean data (filtered beforehand)
            start_char = text.find(selected_text)

            if start_char == -1 or len(selected_text) == 0:
                # Fallback: if not found, point to CLS (ignore in loss via mask or smoothing)
                # Or just point to the whole text.
                # Given alignment filtering, this shouldn't happen often.
                start_labels_list.append(0)
                end_labels_list.append(0)
            else:
                end_char = start_char + len(selected_text)

                # Mask-Based Overlap
                # Find all tokens in the 'text' part that overlap with (start_char, end_char)
                tokens_overlap = []
                for i, (o_start, o_end) in enumerate(full_offsets):
                    # Skip special tokens (offsets are 0,0)
                    if o_start == 0 and o_end == 0:
                        continue

                    # Check overlap
                    # Overlap exists if max(o_start, start_char) < min(o_end, end_char)
                    if max(o_start, start_char) < min(o_end, end_char):
                        tokens_overlap.append(i)

                if len(tokens_overlap) > 0:
                    start_labels_list.append(tokens_overlap[0])
                    end_labels_list.append(tokens_overlap[-1])
                else:
                    # No overlap found (e.g. selected text is just a space that got stripped)
                    start_labels_list.append(0)
                    end_labels_list.append(0)

    # Convert to numpy
    res = {
        "input_ids": np.array(input_ids_list, dtype=np.int32),
        "attention_mask": np.array(attention_mask_list, dtype=np.int32),
        "orig_texts": np.array(
            orig_texts_list, dtype=object
        ),  # Keep as object array of strings
        "offsets": np.array(offsets_list, dtype=np.int32),
    }

    if not is_test:
        res["start_labels"] = np.array(start_labels_list, dtype=np.int32)
        res["end_labels"] = np.array(end_labels_list, dtype=np.int32)

    # 3. Save Cache
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(cache_path, **res)

    return res


def get_train_val_loaders(fold, load_cached_data=True):
    """
    Loads training data, filters neutrals, performs 5-fold split,
    and returns DataLoaders for the specified fold.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    df = pd.read_csv(Config.TRAIN_META_PATH)

    # 2. Filter Neutrals (Training Rule)
    # We only train on positive/negative
    df = df[df["sentiment"] != Config.SENTIMENT_NEUTRAL].reset_index(drop=True)

    # Debugging subsample
    if Config().DEBUG:
        df = df.head(Config().DEBUG_SAMPLE_SIZE)
        print(f"DEBUG Mode: Subsampled training data to {len(df)} rows.")

    # 3. Alignment Filtering
    # Remove rows where selected_text is not found in text (sanity check)
    initial_len = len(df)
    valid_mask = df.apply(lambda x: str(x["selected_text"]) in str(x["text"]), axis=1)
    df = df[valid_mask].reset_index(drop=True)
    if len(df) < initial_len:
        print(f"Filtered {initial_len - len(df)} rows due to alignment issues.")

    # 4. Process Data (Tokenize & Cache)
    # We process the ENTIRE filtered training set once.
    # The cache key depends on debug mode to avoid size mismatches
    cache_prefix = "train_debug" if Config().DEBUG else "train_full"

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    data = process_data(
        df,
        tokenizer,
        Config.MAX_LEN,
        is_test=False,
        cache_prefix=cache_prefix,
        load_cached_data=load_cached_data,
    )

    # 5. Split Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We need to list indices for the current fold
    # Stratified by sentiment
    splits = list(skf.split(df, df["sentiment"]))
    train_idx, val_idx = splits[fold]

    # 6. Create Datasets
    train_dataset = TweetDataset(
        input_ids=data["input_ids"][train_idx],
        attention_mask=data["attention_mask"][train_idx],
        start_labels=data["start_labels"][train_idx],
        end_labels=data["end_labels"][train_idx],
    )

    val_dataset = TweetDataset(
        input_ids=data["input_ids"][val_idx],
        attention_mask=data["attention_mask"][val_idx],
        start_labels=data["start_labels"][val_idx],
        end_labels=data["end_labels"][val_idx],
        orig_texts=data["orig_texts"][val_idx],
        offsets=data["offsets"][val_idx],
    )

    # 7. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Loads test data and returns a DataLoader for inference.
    """
    # 1. Load Metadata
    df = pd.read_csv(Config.TEST_META_PATH)

    # Debugging
    if Config().DEBUG:
        df = df.head(Config().DEBUG_SAMPLE_SIZE)

    # 2. Process Data
    cache_prefix = "test_debug" if Config().DEBUG else "test_full"
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    data = process_data(
        df,
        tokenizer,
        Config.MAX_LEN,
        is_test=True,
        cache_prefix=cache_prefix,
        load_cached_data=load_cached_data,
    )

    # 3. Create Dataset
    test_dataset = TweetDataset(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        orig_texts=data["orig_texts"],
        offsets=data["offsets"],
    )

    # 4. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, df
