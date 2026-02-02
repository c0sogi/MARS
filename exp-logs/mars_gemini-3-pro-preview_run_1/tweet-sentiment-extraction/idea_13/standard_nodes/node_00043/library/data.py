import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import normalize_text


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Sentiment Extraction.
    Returns tokenized inputs, attention masks, soft targets, and offsets.
    Also returns original text metadata for evaluation.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        start_targets,
        end_targets,
        offsets,
        texts=None,
        selected_texts=None,
        sentiments=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets
        self.texts = texts
        self.selected_texts = selected_texts
        self.sentiments = sentiments

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "start_targets": torch.tensor(self.start_targets[item], dtype=torch.float),
            "end_targets": torch.tensor(self.end_targets[item], dtype=torch.float),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
        }

        # Include metadata if available (useful for validation/inference)
        if self.texts is not None:
            data["text"] = str(self.texts[item])
        if self.selected_texts is not None:
            data["selected_text"] = str(self.selected_texts[item])
        if self.sentiments is not None:
            data["sentiment"] = str(self.sentiments[item])

        return data


def get_gaussian_target(index, length, sigma=1.0):
    """
    Generates a Gaussian distribution centered at `index` with standard deviation `sigma`.
    """
    x = np.arange(length)
    gaussian = np.exp(-0.5 * ((x - index) / sigma) ** 2)
    return gaussian / (gaussian.sum() + 1e-9)


def process_data(df, tokenizer, max_len, sigma):
    """
    Tokenizes data and generates soft targets.
    Returns numpy arrays for caching.
    """
    n_samples = len(df)
    input_ids = np.zeros((n_samples, max_len), dtype=np.int32)
    attention_mask = np.zeros((n_samples, max_len), dtype=np.int32)
    start_targets = np.zeros((n_samples, max_len), dtype=np.float32)
    end_targets = np.zeros((n_samples, max_len), dtype=np.float32)
    offsets_arr = np.zeros((n_samples, max_len, 2), dtype=np.int32)

    has_selected_text = "selected_text" in df.columns

    # Iterate with enumeration to ensure correct array indexing
    for idx, (_, row) in enumerate(df.iterrows()):
        text = normalize_text(str(row["text"]))
        sentiment = str(row["sentiment"])

        # Tokenize: [CLS] Sentiment [SEP] Text [SEP]
        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True,
        )

        input_ids[idx] = encoded["input_ids"]
        attention_mask[idx] = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]
        offsets_arr[idx] = offsets

        if has_selected_text and pd.notna(row["selected_text"]):
            selected_text = normalize_text(str(row["selected_text"]))

            # Find character indices in the normalized text
            start_char = text.find(selected_text)
            end_char = start_char + len(selected_text)

            # Fallback for rare edge cases
            if start_char == -1:
                start_char = 0
                end_char = len(text)

            # Identify tokens belonging to the 'text' part (sequence_id == 1)
            sequence_ids = encoded.sequence_ids()
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if not text_token_indices:
                # If no text tokens (empty text), set target to CLS or 0
                start_targets[idx] = get_gaussian_target(0, max_len, sigma)
                end_targets[idx] = get_gaussian_target(0, max_len, sigma)
                continue

            start_token = -1
            end_token = -1

            # Find token span using overlap logic
            for i in text_token_indices:
                off_start, off_end = offsets[i]
                # Check for overlap: token_start < sel_end AND token_end > sel_start
                if off_start < end_char and off_end > start_char:
                    if start_token == -1:
                        start_token = i
                    end_token = i

            # Fallback if no overlap found
            if start_token == -1:
                start_token = text_token_indices[0]
                end_token = text_token_indices[0]

            # Generate soft targets
            start_targets[idx] = get_gaussian_target(start_token, max_len, sigma)
            end_targets[idx] = get_gaussian_target(end_token, max_len, sigma)
        else:
            # No targets available (e.g. test set), leave as zeros
            pass

    return input_ids, attention_mask, start_targets, end_targets, offsets_arr


def get_loaders(fold):
    """
    Prepares DataLoaders for training and validation for a specific fold.
    Handles caching, splitting, and neutral filtering.
    """
    # 1. Load and Prepare Data
    train_df = pd.read_csv(Config.TRAIN_FILE)
    val_df = pd.read_csv("./metadata/validation_metadata.csv")

    # Concatenate to recover full dataset for StratifiedKFold
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    full_df = full_df.dropna(subset=["text", "sentiment", "selected_text"])

    # Create Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(full_df, full_df["sentiment"]))

    train_idx, val_idx = splits[fold]
    train_data = full_df.iloc[train_idx].reset_index(drop=True)
    val_data = full_df.iloc[val_idx].reset_index(drop=True)

    # Filter out 'neutral' tweets for training/validation as per strategy
    train_data = train_data[train_data["sentiment"] != "neutral"].reset_index(drop=True)
    val_data = val_data[val_data["sentiment"] != "neutral"].reset_index(drop=True)

    # 2. Caching & Processing
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    def load_or_process(df, prefix):
        f_ids = os.path.join(cache_dir, f"{prefix}_input_ids.npy")
        f_mask = os.path.join(cache_dir, f"{prefix}_attention_mask.npy")
        f_st = os.path.join(cache_dir, f"{prefix}_start_targets.npy")
        f_et = os.path.join(cache_dir, f"{prefix}_end_targets.npy")
        f_off = os.path.join(cache_dir, f"{prefix}_offsets.npy")

        if Config.LOAD_CACHED_DATA and os.path.exists(f_ids):
            return (
                np.load(f_ids),
                np.load(f_mask),
                np.load(f_st),
                np.load(f_et),
                np.load(f_off),
            )
        else:
            ids, mask, st, et, off = process_data(
                df, tokenizer, Config.MAX_LEN, Config.LABEL_SMOOTHING_SIGMA
            )
            np.save(f_ids, ids)
            np.save(f_mask, mask)
            np.save(f_st, st)
            np.save(f_et, et)
            np.save(f_off, off)
            return ids, mask, st, et, off

    # Process Train
    t_ids, t_mask, t_st, t_et, t_off = load_or_process(train_data, f"fold_{fold}_train")

    # Process Val
    v_ids, v_mask, v_st, v_et, v_off = load_or_process(val_data, f"fold_{fold}_val")

    # 3. Create Datasets
    train_dataset = TweetDataset(
        t_ids,
        t_mask,
        t_st,
        t_et,
        t_off,
        texts=train_data["text"].values,
        selected_texts=train_data["selected_text"].values,
        sentiments=train_data["sentiment"].values,
    )

    val_dataset = TweetDataset(
        v_ids,
        v_mask,
        v_st,
        v_et,
        v_off,
        texts=val_data["text"].values,
        selected_texts=val_data["selected_text"].values,
        sentiments=val_data["sentiment"].values,
    )

    # 4. Create Loaders
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


def get_test_loader(df):
    """
    Creates a DataLoader for the test set (inference).
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

    # Process data (no targets)
    ids, mask, st, et, off = process_data(
        df, tokenizer, Config.MAX_LEN, Config.LABEL_SMOOTHING_SIGMA
    )

    dataset = TweetDataset(
        ids,
        mask,
        st,
        et,
        off,
        texts=df["text"].values,
        selected_texts=None,
        sentiments=df["sentiment"].values,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
