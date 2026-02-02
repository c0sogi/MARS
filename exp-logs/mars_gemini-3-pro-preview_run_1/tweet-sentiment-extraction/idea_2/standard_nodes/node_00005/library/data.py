import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast
from library.config import Config, seed_everything


class TweetDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        start_tokens,
        end_tokens,
        offsets,
        original_texts,
        sentiments,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.offsets = offsets
        self.original_texts = original_texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "start_positions": torch.tensor(self.start_tokens[item], dtype=torch.long),
            "end_positions": torch.tensor(self.end_tokens[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "text": str(self.original_texts[item]),
            "sentiment": str(self.sentiments[item]),
        }
        if self.selected_texts is not None:
            data["selected_text"] = str(self.selected_texts[item])
        return data


def process_data(df, tokenizer, max_len, cache_path, load_cached_data=True):
    """
    Tokenizes data and finds start/end token targets.
    Implements caching using .npy files.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    f_input_ids = cache_path + "_input_ids.npy"
    f_attention_mask = cache_path + "_attention_mask.npy"
    f_start_tokens = cache_path + "_start_tokens.npy"
    f_end_tokens = cache_path + "_end_tokens.npy"
    f_offsets = cache_path + "_offsets.npy"

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(f_input_ids)
        and os.path.exists(f_start_tokens)
    ):
        input_ids = np.load(f_input_ids)
        attention_mask = np.load(f_attention_mask)
        start_tokens = np.load(f_start_tokens)
        end_tokens = np.load(f_end_tokens)
        offsets = np.load(f_offsets)
        return input_ids, attention_mask, start_tokens, end_tokens, offsets

    # Processing from scratch
    texts = df["text"].astype(str).values.tolist()
    sentiments = df["sentiment"].astype(str).values.tolist()

    if "selected_text" in df.columns:
        selected_texts = df["selected_text"].astype(str).values.tolist()
    else:
        selected_texts = [None] * len(texts)

    n_samples = len(texts)
    input_ids = np.zeros((n_samples, max_len), dtype=np.int32)
    attention_mask = np.zeros((n_samples, max_len), dtype=np.int32)
    start_tokens = np.zeros(n_samples, dtype=np.int32)
    end_tokens = np.zeros(n_samples, dtype=np.int32)
    offsets_arr = np.zeros((n_samples, max_len, 2), dtype=np.int32)

    for i in range(n_samples):
        text = texts[i]
        sentiment = sentiments[i]
        selected_text = selected_texts[i]

        # Tokenize: <s> sentiment </s> </s> text </s>
        # We use separate arguments so the tokenizer handles the special tokens and offsets correctly
        enc = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_offsets_mapping=True,
            truncation=True,
            return_token_type_ids=True,
        )

        input_ids[i] = enc["input_ids"]
        attention_mask[i] = enc["attention_mask"]

        # Store offsets
        raw_offsets = enc["offset_mapping"]
        seq_ids = enc.sequence_ids()

        # We only care about offsets for the text part (sequence_id == 1)
        # But we store all for completeness, masking others later if needed
        for j, off in enumerate(raw_offsets):
            if j < max_len:
                offsets_arr[i, j] = off

        # Find targets
        if selected_text is not None:
            # 1. Find character indices of selected_text in text
            start_idx = text.find(selected_text)
            if start_idx == -1:
                # Try stripping as fallback
                start_idx = text.find(selected_text.strip())

            if start_idx == -1:
                # If still not found, default to full text
                start_idx = 0
                end_idx = len(text)
            else:
                end_idx = (
                    start_idx + len(selected_text) if start_idx != -1 else len(text)
                )
                if start_idx == -1:  # Should be covered but safety check
                    end_idx = len(text)
                    start_idx = 0

            # 2. Find tokens corresponding to these character indices
            # Filter for tokens belonging to the second sequence (the tweet text)
            text_token_indices = [
                idx for idx, seq_id in enumerate(seq_ids) if seq_id == 1
            ]

            if not text_token_indices:
                start_tokens[i] = 0
                end_tokens[i] = 0
                continue

            st_token = 0
            en_token = 0
            found_start = False

            # Iterate through text tokens to find overlap
            for idx in text_token_indices:
                # offsets_arr[i, idx] contains [start_char, end_char] relative to text
                o_start, o_end = offsets_arr[i, idx]

                # Check for overlap
                # Token is part of selection if it overlaps with [start_idx, end_idx)
                if o_start < end_idx and o_end > start_idx:
                    if not found_start:
                        st_token = idx
                        found_start = True
                    en_token = idx

            # If no overlap found (e.g. selected text was just a space that got tokenized away),
            # default to the whole text span or the first token
            if not found_start:
                st_token = text_token_indices[0]
                en_token = text_token_indices[-1]

            start_tokens[i] = st_token
            end_tokens[i] = en_token

    # Save to cache
    np.save(f_input_ids, input_ids)
    np.save(f_attention_mask, attention_mask)
    np.save(f_start_tokens, start_tokens)
    np.save(f_end_tokens, end_tokens)
    np.save(f_offsets, offsets_arr)

    return input_ids, attention_mask, start_tokens, end_tokens, offsets_arr


def get_dataloaders(load_cached_data=True):
    seed_everything(Config.SEED)

    # Initialize Tokenizer
    # add_prefix_space=True is generally required for RoBERTa to handle leading words correctly
    tokenizer = RobertaTokenizerFast.from_pretrained(
        Config.TOKENIZER_PATH, add_prefix_space=True
    )

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Debug Mode
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Filter Neutral Tweets for Training
    if Config.FILTER_NEUTRAL_TRAIN:
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)
        # Note: We do not filter validation set here to allow full evaluation if needed,
        # though the model is only trained for pos/neg.

    # Process Train
    train_cache = os.path.join(Config.WORKING_DIR, "cached_train")
    t_ids, t_mask, t_start, t_end, t_off = process_data(
        train_df, tokenizer, Config.MAX_LEN, train_cache, load_cached_data
    )

    train_dataset = TweetDataset(
        t_ids,
        t_mask,
        t_start,
        t_end,
        t_off,
        train_df["text"].values,
        train_df["sentiment"].values,
        train_df["selected_text"].values,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Process Validation
    val_cache = os.path.join(Config.WORKING_DIR, "cached_val")
    v_ids, v_mask, v_start, v_end, v_off = process_data(
        val_df, tokenizer, Config.MAX_LEN, val_cache, load_cached_data
    )

    val_dataset = TweetDataset(
        v_ids,
        v_mask,
        v_start,
        v_end,
        v_off,
        val_df["text"].values,
        val_df["sentiment"].values,
        val_df["selected_text"].values,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Process Test
    test_cache = os.path.join(Config.WORKING_DIR, "cached_test")
    te_ids, te_mask, te_start, te_end, te_off = process_data(
        test_df, tokenizer, Config.MAX_LEN, test_cache, load_cached_data
    )

    test_dataset = TweetDataset(
        te_ids,
        te_mask,
        te_start,
        te_end,
        te_off,
        test_df["text"].values,
        test_df["sentiment"].values,
        None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
