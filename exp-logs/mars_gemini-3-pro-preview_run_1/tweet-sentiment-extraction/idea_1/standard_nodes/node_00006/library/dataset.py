import torch
import numpy as np
import pandas as pd
import os
from library.config import Config


class TweetDataset(torch.utils.data.Dataset):
    """
    Dataset class for Sentiment Extraction.
    Wraps pre-processed tensors and metadata for training and inference.
    """

    def __init__(self, data):
        self.ids = data["ids"]
        self.mask = data["mask"]
        self.offsets = data["offsets"]

        # Metadata
        self.orig_tweet = data["orig_tweet"]
        self.sentiment = data["sentiment"]
        self.text_ids = data["text_ids"]

        # Optional targets (for training/validation)
        self.targets_start = data.get("targets_start", None)
        self.targets_end = data.get("targets_end", None)
        self.orig_selected = data.get("orig_selected", None)
        self.token_type_ids = data.get("token_type_ids", None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "mask": torch.tensor(self.mask[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "orig_tweet": str(self.orig_tweet[idx]),
            "sentiment": str(self.sentiment[idx]),
            "textID": str(self.text_ids[idx]),
        }

        if self.token_type_ids is not None:
            item["token_type_ids"] = torch.tensor(
                self.token_type_ids[idx], dtype=torch.long
            )

        if self.targets_start is not None:
            item["targets_start"] = torch.tensor(
                self.targets_start[idx], dtype=torch.long
            )
            item["targets_end"] = torch.tensor(self.targets_end[idx], dtype=torch.long)

        if self.orig_selected is not None:
            item["orig_selected"] = str(self.orig_selected[idx])

        return item


def process_data(
    df,
    tokenizer,
    max_len,
    cache_dir,
    prefix="train",
    load_cached_data=True,
    debug=False,
):
    """
    Processes the dataframe into input arrays for the model.
    Implements caching using .npy and .parquet files to avoid re-processing.

    Args:
        df (pd.DataFrame): The input dataframe containing text and sentiment.
        tokenizer: The HuggingFace tokenizer.
        max_len (int): Maximum sequence length.
        cache_dir (str): Directory to save/load cached files.
        prefix (str): Prefix for cache filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, processes only a small subset of data.

    Returns:
        dict: A dictionary containing numpy arrays and lists required for TweetDataset.
    """
    if debug:
        df = df.iloc[:100].reset_index(drop=True)
        prefix = f"{prefix}_debug"

    os.makedirs(cache_dir, exist_ok=True)

    # Define file paths
    ids_path = os.path.join(cache_dir, f"{prefix}_ids.npy")
    mask_path = os.path.join(cache_dir, f"{prefix}_mask.npy")
    offsets_path = os.path.join(cache_dir, f"{prefix}_offsets.npy")
    targets_start_path = os.path.join(cache_dir, f"{prefix}_targets_start.npy")
    targets_end_path = os.path.join(cache_dir, f"{prefix}_targets_end.npy")
    meta_path = os.path.join(cache_dir, f"{prefix}_meta.parquet")

    # Determine if we have targets
    has_targets = "selected_text" in df.columns

    # Check cache validity
    cache_exists = (
        os.path.exists(ids_path)
        and os.path.exists(mask_path)
        and os.path.exists(offsets_path)
        and os.path.exists(meta_path)
    )

    if has_targets:
        cache_exists = (
            cache_exists
            and os.path.exists(targets_start_path)
            and os.path.exists(targets_end_path)
        )

    if load_cached_data and cache_exists:
        # Load from cache
        ids = np.load(ids_path)
        mask = np.load(mask_path)
        offsets = np.load(offsets_path)
        meta_df = pd.read_parquet(meta_path)

        data = {
            "ids": ids,
            "mask": mask,
            "offsets": offsets,
            "orig_tweet": meta_df["text"].values,
            "sentiment": meta_df["sentiment"].values,
            "text_ids": meta_df["textID"].values,
        }

        if has_targets:
            data["targets_start"] = np.load(targets_start_path)
            data["targets_end"] = np.load(targets_end_path)
            data["orig_selected"] = meta_df["selected_text"].values

        return data

    # Process data from scratch
    ids_list = []
    mask_list = []
    offsets_list = []
    targets_start_list = []
    targets_end_list = []

    # Metadata lists
    processed_texts = []
    processed_sentiments = []
    processed_ids = []
    processed_selected = []

    for _, row in df.iterrows():
        text = str(row.text)
        sentiment = str(row.sentiment)
        text_id = str(row.textID)

        # Normalize spaces to ensure alignment between raw text and tokenizer offsets
        # " ".join(split()) replaces multiple spaces with a single space
        text_clean = " " + " ".join(text.split())

        # Tokenize: <s> sentiment </s> </s> text </s>
        encoded = tokenizer.encode_plus(
            sentiment,
            text_clean,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]

        # sequence_ids: None (special), 0 (sentiment), 1 (text)
        sequence_ids = encoded.sequence_ids()

        ids_list.append(input_ids)
        mask_list.append(attention_mask)
        offsets_list.append(offsets)

        processed_texts.append(text)
        processed_sentiments.append(sentiment)
        processed_ids.append(text_id)

        if has_targets:
            selected_text = str(row.selected_text)
            processed_selected.append(selected_text)

            # Calculate targets
            selected_text_clean = " " + " ".join(selected_text.split())

            # Find character indices in the cleaned text
            start_idx = text_clean.find(selected_text_clean)
            end_idx = start_idx + len(selected_text_clean)

            target_start = 0
            target_end = 0

            if start_idx != -1:
                found_start = False

                for idx, (seq_id, offset) in enumerate(zip(sequence_ids, offsets)):
                    # We only care about the text part (sequence_id == 1)
                    if seq_id != 1:
                        continue

                    # Check for overlap between token offset and selected span
                    # Token interval: [offset[0], offset[1])
                    # Selection interval: [start_idx, end_idx)
                    # Overlap condition: max(start1, start2) < min(end1, end2)
                    if offset[1] > start_idx and offset[0] < end_idx:
                        if not found_start:
                            target_start = idx
                            found_start = True
                        target_end = idx

            targets_start_list.append(target_start)
            targets_end_list.append(target_end)

    # Convert to numpy arrays
    ids_arr = np.array(ids_list)
    mask_arr = np.array(mask_list)
    offsets_arr = np.array(offsets_list)

    # Save to cache
    np.save(ids_path, ids_arr)
    np.save(mask_path, mask_arr)
    np.save(offsets_path, offsets_arr)

    meta_dict = {
        "textID": processed_ids,
        "text": processed_texts,
        "sentiment": processed_sentiments,
    }

    data = {
        "ids": ids_arr,
        "mask": mask_arr,
        "offsets": offsets_arr,
        "orig_tweet": np.array(processed_texts),
        "sentiment": np.array(processed_sentiments),
        "text_ids": np.array(processed_ids),
    }

    if has_targets:
        targets_start_arr = np.array(targets_start_list)
        targets_end_arr = np.array(targets_end_list)

        np.save(targets_start_path, targets_start_arr)
        np.save(targets_end_path, targets_end_arr)

        meta_dict["selected_text"] = processed_selected
        data["targets_start"] = targets_start_arr
        data["targets_end"] = targets_end_arr
        data["orig_selected"] = np.array(processed_selected)

    # Save metadata
    pd.DataFrame(meta_dict).to_parquet(meta_path)

    return data
