import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Stores pre-tokenized inputs and targets to minimize runtime overhead.
    """

    def __init__(
        self,
        input_ids,
        attention_masks,
        start_pos,
        end_pos,
        offsets,
        orig_texts,
        sentiments,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.offsets = offsets
        self.orig_texts = orig_texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        data = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "start_positions": torch.tensor(self.start_pos[idx], dtype=torch.long),
            "end_positions": torch.tensor(self.end_pos[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "text": str(self.orig_texts[idx]),
            "sentiment": str(self.sentiments[idx]),
        }

        # Include selected_text if available (for validation/debugging)
        if self.selected_texts is not None:
            data["selected_text"] = str(self.selected_texts[idx])

        return data


def _process_data(df, tokenizer, max_len, is_test=False):
    """
    Tokenizes data and computes start/end token indices for the selected text.
    """
    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []
    offsets_list = []

    # Iterate over dataframe efficiently
    for row in df.itertuples(index=False):
        text = str(row.text)
        sentiment = str(row.sentiment)

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # We pass (sentiment, text) to encode_plus
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]

        # sequence_ids identifies which part of the input the token belongs to.
        # None: special tokens, 0: sentiment, 1: text
        seq_ids = encoded.sequence_ids()

        start_idx = 0
        end_idx = 0

        if not is_test:
            selected_text = str(row.selected_text)

            # Find the character start/end of selected_text within text
            start_char = text.find(selected_text)

            # Fallback for whitespace inconsistencies
            if start_char == -1:
                start_char = text.find(selected_text.strip())

            if start_char == -1:
                # If still not found, default to the entire text
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # Find the tokens corresponding to this character span
            # We only look at tokens belonging to the 'text' sequence (seq_id == 1)
            token_indices = [i for i, s in enumerate(seq_ids) if s == 1]

            if token_indices:
                # Default to the full text span
                s_token = token_indices[0]
                e_token = token_indices[-1]

                # Refine start token: Find first token that overlaps with start_char
                for idx in token_indices:
                    off_start, off_end = offsets[idx]
                    # Check if start_char falls within this token's span
                    if off_start <= start_char < off_end:
                        s_token = idx
                        break
                    # Or if the token starts after the character (shouldn't happen if contained)
                    if off_start > start_char:
                        s_token = idx
                        break

                # Refine end token: Find token that covers end_char
                for idx in token_indices:
                    off_start, off_end = offsets[idx]
                    # Check if end_char falls within or at the end of this token
                    if off_start < end_char <= off_end:
                        e_token = idx
                        break

                start_idx = s_token
                end_idx = e_token

            # Safety clamp
            if start_idx >= max_len:
                start_idx = max_len - 1
            if end_idx >= max_len:
                end_idx = max_len - 1
            if start_idx > end_idx:
                end_idx = start_idx

        input_ids.append(ids)
        attention_masks.append(mask)
        start_positions.append(start_idx)
        end_positions.append(end_idx)
        offsets_list.append(offsets)

    return (
        np.array(input_ids),
        np.array(attention_masks),
        np.array(start_positions),
        np.array(end_positions),
        np.array(offsets_list),
    )


def get_data(load_cached_data=True):
    """
    Loads data, processes it (with caching), and returns DataLoaders.
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Metadata DataFrames
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Filter 'neutral' from training data if configured
    if Config.FILTER_NEUTRAL_TRAIN:
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)

    # Filter rows where selected_text is not in text (Cite solution_lesson_node_00013)
    # This prevents the model from training on noisy/impossible samples.
    initial_len = len(train_df)
    train_df = train_df[
        train_df.apply(
            lambda x: str(x["selected_text"]).strip() in str(x["text"])
            or str(x["selected_text"]) in str(x["text"]),
            axis=1,
        )
    ].reset_index(drop=True)
    print(f"Filtered {initial_len - len(train_df)} rows with invalid spans.")

    # Helper to process or load cache
    def get_split(df, split_name, is_test=False):
        cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        prefix = os.path.join(cache_dir, split_name)
        files = {
            "ids": f"{prefix}_ids.npy",
            "masks": f"{prefix}_masks.npy",
            "start": f"{prefix}_start.npy",
            "end": f"{prefix}_end.npy",
            "offsets": f"{prefix}_offsets.npy",
        }

        # Check if all cache files exist
        cache_exists = all(os.path.exists(f) for f in files.values())

        if load_cached_data and cache_exists:
            input_ids = np.load(files["ids"])
            attention_masks = np.load(files["masks"])
            start_pos = np.load(files["start"])
            end_pos = np.load(files["end"])
            offsets = np.load(files["offsets"])
        else:
            # Compute from scratch
            input_ids, attention_masks, start_pos, end_pos, offsets = _process_data(
                df, tokenizer, Config.MAX_LEN, is_test
            )
            # Save to cache
            np.save(files["ids"], input_ids)
            np.save(files["masks"], attention_masks)
            np.save(files["start"], start_pos)
            np.save(files["end"], end_pos)
            np.save(files["offsets"], offsets)

        return TweetDataset(
            input_ids=input_ids,
            attention_masks=attention_masks,
            start_pos=start_pos,
            end_pos=end_pos,
            offsets=offsets,
            orig_texts=df["text"].values,
            sentiments=df["sentiment"].values,
            selected_texts=df["selected_text"].values if not is_test else None,
        )

    # Create Datasets
    # Use distinct cache names to avoid collisions (especially with filtering)
    train_cache_name = "train_filtered" if Config.FILTER_NEUTRAL_TRAIN else "train"

    train_dataset = get_split(train_df, train_cache_name, is_test=False)
    val_dataset = get_split(val_df, "val", is_test=False)
    test_dataset = get_split(test_df, "test", is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
