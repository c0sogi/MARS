import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import normalize_text


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Sentiment Extraction.
    Wraps pre-processed tensors for input into the model.
    """

    def __init__(self, input_ids, attention_mask, start_targets, end_targets, offsets):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "start_targets": torch.tensor(self.start_targets[item], dtype=torch.float),
            "end_targets": torch.tensor(self.end_targets[item], dtype=torch.float),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
        }


def get_gaussian_target(target_idx, length, sigma=1.0):
    """
    Generates a Gaussian distribution centered at target_idx.
    Used for Soft Targets (Label Smoothing).
    """
    x = np.arange(length)
    # Calculate Gaussian
    gaussian = np.exp(-0.5 * ((x - target_idx) / sigma) ** 2)
    # Normalize to sum to 1 (probability distribution)
    return gaussian / gaussian.sum()


def process_data(df, tokenizer, max_len, is_test=False):
    """
    Processes the dataframe into model inputs:
    - Normalizes text
    - Tokenizes (sentiment, text) pairs
    - Generates Soft Targets for start/end indices
    """
    input_ids = []
    attention_masks = []
    start_targets = []
    end_targets = []
    offsets_list = []

    sigma = float(Config.TARGET_SMOOTHING)

    for _, row in df.iterrows():
        # Normalize text to ensure consistency
        text = normalize_text(str(row["text"]))
        sentiment = str(row["sentiment"])

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # This structure allows the model to attend to sentiment explicitly
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        offsets = encoded["offset_mapping"]
        offsets_list.append(offsets)

        if is_test:
            # Dummy targets for test set
            start_targets.append(np.zeros(max_len))
            end_targets.append(np.zeros(max_len))
        else:
            selected_text = normalize_text(str(row["selected_text"]))

            # Find start/end character indices in normalized text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: if normalization causes mismatch, use full text
                # This ensures we always have a valid target
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # Identify tokens corresponding to the text part (sequence_id == 1)
            sequence_ids = encoded.sequence_ids()
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            s_idx = 0
            e_idx = 0

            if len(text_token_indices) > 0:
                # Default to full span of text tokens
                s_idx = text_token_indices[0]
                e_idx = text_token_indices[-1]

                # Refine to specific tokens using containment logic
                for i in text_token_indices:
                    off_start, off_end = offsets[i]
                    if off_start == off_end:
                        continue  # Skip zero-width tokens

                    # Start token: contains start_char
                    if off_start <= start_char < off_end:
                        s_idx = i

                    # End token: contains (end_char - 1)
                    # We look for the token containing the last character of the selection
                    target_end_char = end_char - 1
                    if off_start <= target_end_char < off_end:
                        e_idx = i

            # Generate Gaussian Soft Targets
            s_target = get_gaussian_target(s_idx, max_len, sigma)
            e_target = get_gaussian_target(e_idx, max_len, sigma)

            start_targets.append(s_target)
            end_targets.append(e_target)

    return (
        np.array(input_ids),
        np.array(attention_masks),
        np.array(start_targets),
        np.array(end_targets),
        np.array(offsets_list),
    )


def get_loaders(tokenizer, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles caching to disk to speed up subsequent runs.
    """
    # Ensure artifact directory exists
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)

    splits = ["train", "val", "test"]
    loaders = {}

    # If Debugging, disable cache loading to force re-processing of small subset
    if Config.DEBUG:
        load_cached_data = False
        print("DEBUG mode enabled: Skipping cache loading.")

    for split in splits:
        # Determine file path and filtering logic
        if split == "train":
            file_path = Config.TRAIN_FILE
            # Suffix depends on neutral filtering configuration
            suffix = "_no_neutral" if not Config.TRAIN_ON_NEUTRAL else "_all"
            is_test = False
        elif split == "val":
            file_path = Config.VAL_FILE
            suffix = ""
            is_test = False
        else:  # test
            file_path = Config.TEST_FILE
            suffix = ""
            is_test = True

        cache_prefix = os.path.join(Config.ARTIFACT_DIR, f"{split}{suffix}")
        if Config.DEBUG:
            cache_prefix += "_debug"

        # Cache file paths
        f_ids = f"{cache_prefix}_input_ids.npy"
        f_mask = f"{cache_prefix}_attention_mask.npy"
        f_st = f"{cache_prefix}_start_targets.npy"
        f_et = f"{cache_prefix}_end_targets.npy"
        f_off = f"{cache_prefix}_offsets.npy"
        f_meta = f"{cache_prefix}_meta.parquet"

        # Check if cache exists
        cache_exists = (
            os.path.exists(f_ids)
            and os.path.exists(f_mask)
            and os.path.exists(f_st)
            and os.path.exists(f_et)
            and os.path.exists(f_off)
            and os.path.exists(f_meta)
        )

        if load_cached_data and cache_exists:
            print(f"Loading cached data for {split} from {Config.ARTIFACT_DIR}...")
            input_ids = np.load(f_ids)
            attention_mask = np.load(f_mask)
            start_targets = np.load(f_st)
            end_targets = np.load(f_et)
            offsets = np.load(f_off)
        else:
            print(f"Processing data for {split}...")
            df = pd.read_csv(file_path)

            # Debugging: use small subset
            if Config.DEBUG:
                df = df.head(100)

            # Apply filtering for train (exclude neutrals if configured)
            if split == "train" and not Config.TRAIN_ON_NEUTRAL:
                initial_len = len(df)
                df = df[df["sentiment"] != "neutral"].reset_index(drop=True)
                print(
                    f"Filtered out {initial_len - len(df)} neutral tweets from training set."
                )

            # Process data
            input_ids, attention_mask, start_targets, end_targets, offsets = (
                process_data(df, tokenizer, Config.MAX_LEN, is_test=is_test)
            )

            # Save to cache
            np.save(f_ids, input_ids)
            np.save(f_mask, attention_mask)
            np.save(f_st, start_targets)
            np.save(f_et, end_targets)
            np.save(f_off, offsets)
            df.to_parquet(f_meta, index=False)

        # Create Dataset
        dataset = TweetDataset(
            input_ids, attention_mask, start_targets, end_targets, offsets
        )

        # Create Loader
        shuffle = split == "train"
        batch_size = (
            Config.TRAIN_BATCH_SIZE if split == "train" else Config.VALID_BATCH_SIZE
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]
