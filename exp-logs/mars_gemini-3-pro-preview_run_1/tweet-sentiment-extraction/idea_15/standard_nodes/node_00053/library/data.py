import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from library.config import Config
from library.utils import seed_everything


def process_text(text):
    """
    Normalizes whitespace by collapsing multiple spaces into one and stripping.
    This is the 'Normalize-First' protocol.
    """
    return " ".join(str(text).split())


class TweetDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        offsets,
        texts,
        sentiments,
        start_targets=None,
        end_targets=None,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offsets = offsets
        self.texts = texts
        self.sentiments = sentiments
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "text": self.texts[idx],
            "sentiment": self.sentiments[idx],
        }

        if self.start_targets is not None:
            item["start_targets"] = torch.tensor(
                self.start_targets[idx], dtype=torch.float
            )
            item["end_targets"] = torch.tensor(self.end_targets[idx], dtype=torch.float)

        if self.selected_texts is not None:
            item["selected_text"] = self.selected_texts[idx]

        return item


def get_gaussian_target(idx, length, sigma=1.0):
    """
    Generates a Gaussian distribution centered at idx.
    """
    if idx < 0 or idx >= length:
        # Fallback for invalid indices: uniform or zero (here uniform small prob)
        return np.ones(length) / length

    x = np.arange(length)
    target = np.exp(-0.5 * ((x - idx) / sigma) ** 2)
    target = target / target.sum()
    return target


def process_data(df, tokenizer, max_len, is_train=True, has_labels=False):
    """
    Tokenizes data and generates soft targets.
    """
    # Normalize text columns
    df["text"] = df["text"].apply(process_text)
    if "selected_text" in df.columns:
        df["selected_text"] = df["selected_text"].apply(process_text)

    # Filter neutrals for training if configured
    if is_train and Config.FILTER_NEUTRAL:
        initial_len = len(df)
        df = df[df["sentiment"] != "neutral"].reset_index(drop=True)
        # print(f"Filtered neutral tweets: {initial_len} -> {len(df)}")

    input_ids = []
    attention_masks = []
    offsets_list = []
    start_targets = []
    end_targets = []

    # Pre-compute arrays for speed
    n_samples = len(df)

    # Iterate over dataframe
    for i, row in df.iterrows():
        text = row["text"]
        sentiment = row["sentiment"]

        # Early Fusion: [CLS] Sentiment [SEP] Text [SEP]
        # DeBERTa tokenizer handles this via text_pair
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )

        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

        # Adjust offsets: We only care about the second sequence (the text)
        # sequence_ids: None (special), 0 (sentiment), 1 (text)
        seq_ids = encoded.sequence_ids()
        offsets = encoded["offset_mapping"]

        # Filter offsets to only include those belonging to the text (seq_id=1)
        # and keep the original structure for the model
        # We will store the full offsets but use logic to identify text tokens
        offsets_list.append(offsets)

        if has_labels:
            selected_text = row["selected_text"]

            # Find character start/end of selected_text in text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: Use full text if exact match fail (rare after normalization)
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # Find token span
            token_start_idx = -1
            token_end_idx = -1

            # Iterate through tokens to find the span
            # We only look at tokens corresponding to the text (sequence_id == 1)
            text_token_indices = [
                idx for idx, seq_id in enumerate(seq_ids) if seq_id == 1
            ]

            if not text_token_indices:
                # Should not happen with valid text
                token_start_idx = 0
                token_end_idx = 0
            else:
                # Find first token that starts after or at start_char
                # Containment logic: token should be inside or overlapping significantly?
                # Prompt: "Containment mapping: token_start <= char_idx < token_end"

                # Logic: Find the first token that overlaps with the start
                for idx in text_token_indices:
                    off_start, off_end = offsets[idx]
                    if off_start <= start_char < off_end:
                        token_start_idx = idx
                        break
                    # If the token is contained within the selection (start_char < off_start)
                    # This handles cases where selection starts between tokens (whitespace)
                    if off_start >= start_char:
                        token_start_idx = idx
                        break

                # Find last token
                for idx in reversed(text_token_indices):
                    off_start, off_end = offsets[idx]
                    # if end_char falls inside this token
                    if off_start < end_char <= off_end:
                        token_end_idx = idx
                        break
                    # if token ends before end_char
                    if off_end <= end_char:
                        token_end_idx = idx
                        break

                # Fallback if not found
                if token_start_idx == -1:
                    token_start_idx = text_token_indices[0]
                if token_end_idx == -1:
                    token_end_idx = text_token_indices[-1]

            # Generate Gaussian Soft Targets
            s_target = get_gaussian_target(
                token_start_idx, max_len, Config.TARGET_SMOOTHING_SIGMA
            )
            e_target = get_gaussian_target(
                token_end_idx, max_len, Config.TARGET_SMOOTHING_SIGMA
            )

            start_targets.append(s_target)
            end_targets.append(e_target)

    # Convert to numpy arrays
    data_dict = {
        "input_ids": np.array(input_ids),
        "attention_mask": np.array(attention_masks),
        "offsets": np.array(offsets_list),
        "texts": df["text"].values,
        "sentiments": df["sentiment"].values,
        "textIDs": df["textID"].values,
    }

    if has_labels:
        data_dict["start_targets"] = np.array(start_targets)
        data_dict["end_targets"] = np.array(end_targets)
        data_dict["selected_texts"] = df["selected_text"].values

    return data_dict


def load_and_cache_data(file_path, mode, load_cached_data=True):
    """
    Loads data from CSV or Cache.
    """
    cache_dir = Config.OUTPUT_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    # We use mode (train/val/test) to distinguish
    # Note: For CV, we cache the full train file once, then split in memory.

    cache_files = {
        "input_ids": os.path.join(cache_dir, f"cached_{mode}_input_ids.npy"),
        "attention_mask": os.path.join(cache_dir, f"cached_{mode}_attention_mask.npy"),
        "offsets": os.path.join(cache_dir, f"cached_{mode}_offsets.npy"),
        "meta": os.path.join(cache_dir, f"cached_{mode}_meta.parquet"),
    }

    has_labels = mode in ["train", "val"]

    if has_labels:
        cache_files["start_targets"] = os.path.join(
            cache_dir, f"cached_{mode}_start_targets.npy"
        )
        cache_files["end_targets"] = os.path.join(
            cache_dir, f"cached_{mode}_end_targets.npy"
        )

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        # print(f"Loading cached data for {mode}...")
        data = {}
        data["input_ids"] = np.load(cache_files["input_ids"])
        data["attention_mask"] = np.load(cache_files["attention_mask"])
        data["offsets"] = np.load(cache_files["offsets"])

        meta_df = pd.read_parquet(cache_files["meta"])
        data["texts"] = meta_df["text"].values
        data["sentiments"] = meta_df["sentiment"].values
        data["textIDs"] = meta_df["textID"].values

        if has_labels:
            data["start_targets"] = np.load(cache_files["start_targets"])
            data["end_targets"] = np.load(cache_files["end_targets"])
            data["selected_texts"] = meta_df["selected_text"].values

        return data

    # Process from scratch
    # print(f"Processing data for {mode} from {file_path}...")
    df = pd.read_csv(file_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Process
    is_train = mode == "train"
    data = process_data(
        df, tokenizer, Config.MAX_LEN, is_train=is_train, has_labels=has_labels
    )

    # Save to cache
    np.save(cache_files["input_ids"], data["input_ids"])
    np.save(cache_files["attention_mask"], data["attention_mask"])
    np.save(cache_files["offsets"], data["offsets"])

    meta_cols = ["textID", "text", "sentiment"]
    if has_labels:
        np.save(cache_files["start_targets"], data["start_targets"])
        np.save(cache_files["end_targets"], data["end_targets"])
        meta_cols.append("selected_text")

    # Save metadata as parquet
    meta_df = pd.DataFrame(
        {
            k: data[k + "s"] if k + "s" in data else data["selected_texts"]
            for k in ["textID", "text", "sentiment"]
        }
    )
    if has_labels:
        meta_df["selected_text"] = data["selected_texts"]

    meta_df.to_parquet(cache_files["meta"], index=False)

    return data


def get_loaders(fold, load_cached_data=True, debug=False):
    """
    Creates Train and Validation DataLoaders for a specific fold.
    Uses StratifiedKFold on the training data.
    """
    # Load full training data
    data = load_and_cache_data(Config.TRAIN_FILE, "train", load_cached_data)

    # Create Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We split based on sentiment to ensure balanced folds
    splits = list(skf.split(data["input_ids"], data["sentiments"]))
    train_idx, val_idx = splits[fold]

    if debug:
        train_idx = train_idx[:100]
        val_idx = val_idx[:100]

    # Helper to slice dictionary arrays
    def slice_data(indices):
        return {
            "input_ids": data["input_ids"][indices],
            "attention_mask": data["attention_mask"][indices],
            "offsets": data["offsets"][indices],
            "texts": data["texts"][indices],
            "sentiments": data["sentiments"][indices],
            "start_targets": data["start_targets"][indices],
            "end_targets": data["end_targets"][indices],
            "selected_texts": data["selected_texts"][indices],
        }

    train_data = slice_data(train_idx)
    val_data = slice_data(val_idx)

    train_dataset = TweetDataset(**train_data)
    val_dataset = TweetDataset(**val_data)

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


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the Test set.
    """
    data = load_and_cache_data(Config.TEST_FILE, "test", load_cached_data)

    dataset = TweetDataset(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        offsets=data["offsets"],
        texts=data["texts"],
        sentiments=data["sentiments"],
        start_targets=None,
        end_targets=None,
        selected_texts=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader, data["textIDs"]
