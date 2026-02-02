import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
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
        start_labels,
        end_labels,
        span_masks,
        offsets,
        raw_texts,
        sentiments,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.span_masks = span_masks
        self.offsets = offsets
        self.raw_texts = raw_texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "start_labels": torch.tensor(self.start_labels[idx], dtype=torch.float),
            "end_labels": torch.tensor(self.end_labels[idx], dtype=torch.float),
            "span_masks": torch.tensor(self.span_masks[idx], dtype=torch.float),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "raw_text": str(self.raw_texts[idx]),
            "sentiment": str(self.sentiments[idx]),
        }

        if self.selected_texts is not None:
            item["selected_text"] = str(self.selected_texts[idx])

        return item


def get_gaussian_target(target_idx, length, sigma=1.0):
    """
    Generates a Gaussian distribution centered at target_idx.
    """
    if target_idx < 0 or target_idx >= length:
        # Fallback: uniform distribution to avoid NaNs in loss
        return np.ones(length) / length

    x = np.arange(length)
    g = np.exp(-0.5 * ((x - target_idx) / sigma) ** 2)
    return g / g.sum()


def process_data(df, tokenizer, config, mode="train", load_cached_data=True):
    """
    Processes the dataframe into numpy arrays for the model.
    Handles caching to speed up subsequent runs.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Unique suffix based on data length to handle debug vs full runs safely
    suffix = f"{len(df)}"

    # Define cache filenames
    cache_files = {
        "input_ids": os.path.join(cache_dir, f"cached_{mode}_{suffix}_input_ids.npy"),
        "attention_mask": os.path.join(
            cache_dir, f"cached_{mode}_{suffix}_attention_mask.npy"
        ),
        "start_labels": os.path.join(
            cache_dir, f"cached_{mode}_{suffix}_start_tokens.npy"
        ),
        "end_labels": os.path.join(cache_dir, f"cached_{mode}_{suffix}_end_tokens.npy"),
        "span_masks": os.path.join(cache_dir, f"cached_{mode}_{suffix}_span_masks.npy"),
        "offsets": os.path.join(cache_dir, f"cached_{mode}_{suffix}_offsets.npy"),
        "meta": os.path.join(cache_dir, f"cached_{mode}_{suffix}_meta.parquet"),
    }

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print(f"Loading cached data for {mode} (Size: {len(df)})...")
        input_ids = np.load(cache_files["input_ids"])
        attention_mask = np.load(cache_files["attention_mask"])
        start_labels = np.load(cache_files["start_labels"])
        end_labels = np.load(cache_files["end_labels"])
        span_masks = np.load(cache_files["span_masks"])
        offsets = np.load(cache_files["offsets"])
        meta_df = pd.read_parquet(cache_files["meta"])
        return (
            input_ids,
            attention_mask,
            start_labels,
            end_labels,
            span_masks,
            offsets,
            meta_df,
        )

    print(f"Processing data for {mode} (Size: {len(df)})...")

    # Lists to store data
    input_ids_list = []
    attention_mask_list = []
    start_labels_list = []
    end_labels_list = []
    span_masks_list = []
    offsets_list = []

    # Meta lists
    raw_texts = []
    sentiments = []
    selected_texts = []

    for _, row in df.iterrows():
        # strict whitespace normalization
        text = " ".join(str(row["text"]).split())
        sentiment = str(row["sentiment"])

        # Construct input: "sentiment text"
        # The tokenizer will handle special tokens ([CLS], [SEP])
        input_text = f"{sentiment} {text}"

        encoded = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        offset_mapping = encoded["offset_mapping"]

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        offsets_list.append(offset_mapping)

        raw_texts.append(text)
        sentiments.append(sentiment)

        # Initialize Targets
        start_vec = np.zeros(config.MAX_LEN)
        end_vec = np.zeros(config.MAX_LEN)
        span_vec = np.zeros(config.MAX_LEN)

        # Generate targets if selected_text exists
        if "selected_text" in row and pd.notna(row["selected_text"]):
            sel_text = " ".join(str(row["selected_text"]).split())
            selected_texts.append(sel_text)

            # Calculate offset for the text part in "sentiment text"
            # +1 for the space between sentiment and text
            text_start_offset = len(sentiment) + 1

            # Find selected_text within the normalized text
            idx_in_text = text.find(sel_text)

            if idx_in_text != -1:
                # Absolute char indices in input_text
                char_start = text_start_offset + idx_in_text
                char_end = char_start + len(sel_text)

                token_start_idx = 0
                token_end_idx = 0
                found_start = False

                # Map char indices to token indices
                for i, (o_start, o_end) in enumerate(offset_mapping):
                    if o_start == 0 and o_end == 0:
                        continue  # Skip special tokens

                    # Start Token: First token containing the start char
                    if not found_start and o_start <= char_start < o_end:
                        token_start_idx = i
                        found_start = True

                    # End Token: Last token containing part of the span
                    # We check if the token overlaps with the span [char_start, char_end)
                    if o_start < char_end:
                        token_end_idx = i
                    else:
                        break  # Token is past the span

                # Safety check
                if token_end_idx < token_start_idx:
                    token_end_idx = token_start_idx

                # Generate Gaussian Targets
                start_vec = get_gaussian_target(
                    token_start_idx, config.MAX_LEN, config.TARGET_SMOOTHING_SIGMA
                )
                end_vec = get_gaussian_target(
                    token_end_idx, config.MAX_LEN, config.TARGET_SMOOTHING_SIGMA
                )

                # Generate Auxiliary Dense Mask
                span_vec[token_start_idx : token_end_idx + 1] = 1.0
        else:
            selected_texts.append("")

        start_labels_list.append(start_vec)
        end_labels_list.append(end_vec)
        span_masks_list.append(span_vec)

    # Convert to numpy arrays
    input_ids = np.array(input_ids_list)
    attention_mask = np.array(attention_mask_list)
    start_labels = np.array(start_labels_list)
    end_labels = np.array(end_labels_list)
    span_masks = np.array(span_masks_list)
    offsets = np.array(offsets_list)

    # Create meta dataframe
    meta_df = pd.DataFrame(
        {"text": raw_texts, "sentiment": sentiments, "selected_text": selected_texts}
    )

    # Save to cache
    np.save(cache_files["input_ids"], input_ids)
    np.save(cache_files["attention_mask"], attention_mask)
    np.save(cache_files["start_labels"], start_labels)
    np.save(cache_files["end_labels"], end_labels)
    np.save(cache_files["span_masks"], span_masks)
    np.save(cache_files["offsets"], offsets)
    meta_df.to_parquet(cache_files["meta"])

    return (
        input_ids,
        attention_mask,
        start_labels,
        end_labels,
        span_masks,
        offsets,
        meta_df,
    )


def get_dataloaders(config, load_cached_data=True):
    """
    Prepares DataLoaders for train, validation, and test sets.
    """
    seed_everything(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_META_PATH)
    val_df = pd.read_csv(config.VAL_META_PATH)
    test_df = pd.read_csv(config.TEST_META_PATH)

    # Apply Debugging Limit
    if config.DEBUG_SAMPLE_SIZE:
        print(f"DEBUG MODE: Reducing dataset size to {config.DEBUG_SAMPLE_SIZE}")
        train_df = train_df.head(config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

    # Filter Neutral Tweets from Training
    if config.FILTER_NEUTRAL_TRAIN:
        initial_len = len(train_df)
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)
        print(
            f"Filtered neutral tweets from training. Size: {initial_len} -> {len(train_df)}"
        )

    # Process Data
    train_data = process_data(
        train_df, tokenizer, config, mode="train", load_cached_data=load_cached_data
    )
    val_data = process_data(
        val_df, tokenizer, config, mode="val", load_cached_data=load_cached_data
    )
    test_data = process_data(
        test_df, tokenizer, config, mode="test", load_cached_data=load_cached_data
    )

    # Unpack Data
    t_ids, t_mask, t_start, t_end, t_span, t_off, t_meta = train_data
    v_ids, v_mask, v_start, v_end, v_span, v_off, v_meta = val_data
    te_ids, te_mask, te_start, te_end, te_span, te_off, te_meta = test_data

    # Initialize Datasets
    train_dataset = TweetDataset(
        t_ids,
        t_mask,
        t_start,
        t_end,
        t_span,
        t_off,
        t_meta["text"].values,
        t_meta["sentiment"].values,
        t_meta["selected_text"].values,
    )

    val_dataset = TweetDataset(
        v_ids,
        v_mask,
        v_start,
        v_end,
        v_span,
        v_off,
        v_meta["text"].values,
        v_meta["sentiment"].values,
        v_meta["selected_text"].values,
    )

    test_dataset = TweetDataset(
        te_ids,
        te_mask,
        te_start,
        te_end,
        te_span,
        te_off,
        te_meta["text"].values,
        te_meta["sentiment"].values,
        None,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
