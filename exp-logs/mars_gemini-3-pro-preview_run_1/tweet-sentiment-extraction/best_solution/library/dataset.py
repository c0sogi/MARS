import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import normalize_text, get_soft_targets


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Stores pre-processed tensors and metadata.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        start_targets,
        end_targets,
        offsets,
        texts,
        selected_texts,
        sentiments,
        ids,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets
        self.texts = texts
        self.selected_texts = selected_texts
        self.sentiments = sentiments
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "start_targets": torch.tensor(self.start_targets[item], dtype=torch.float),
            "end_targets": torch.tensor(self.end_targets[item], dtype=torch.float),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "text": str(self.texts[item]),
            "selected_text": str(self.selected_texts[item]),
            "sentiment": str(self.sentiments[item]),
            "textID": str(self.ids[item]),
        }


def get_data(df, tokenizer, split_name, load_cached_data=True, sample_size=None):
    """
    Prepares the dataset by processing raw text into tokenized features and targets.
    Implements caching to speed up subsequent runs.

    Args:
        df (pd.DataFrame): The raw dataframe.
        tokenizer: The HuggingFace tokenizer.
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): If set, truncates data for debugging.

    Returns:
        TweetDataset: The instantiated dataset.
    """

    # Handle Debugging / Sampling
    if sample_size is not None:
        df = df.head(sample_size)
        split_name = f"{split_name}_debug"

    # Ensure working directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    paths = {
        "input_ids": os.path.join(cache_dir, f"cached_{split_name}_input_ids.npy"),
        "attention_mask": os.path.join(
            cache_dir, f"cached_{split_name}_attention_mask.npy"
        ),
        "start_targets": os.path.join(
            cache_dir, f"cached_{split_name}_start_targets.npy"
        ),
        "end_targets": os.path.join(cache_dir, f"cached_{split_name}_end_targets.npy"),
        "offsets": os.path.join(cache_dir, f"cached_{split_name}_offsets.npy"),
        "meta": os.path.join(cache_dir, f"cached_{split_name}_meta.parquet"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data for {split_name} from {cache_dir}...")
        input_ids = np.load(paths["input_ids"])
        attention_mask = np.load(paths["attention_mask"])
        start_targets = np.load(paths["start_targets"])
        end_targets = np.load(paths["end_targets"])
        offsets = np.load(paths["offsets"])

        meta_df = pd.read_parquet(paths["meta"])
        texts = meta_df["text"].values
        # Handle potential missing column in test set cache if structure changed
        selected_texts = (
            meta_df["selected_text"].values
            if "selected_text" in meta_df.columns
            else np.full(len(texts), "")
        )
        sentiments = meta_df["sentiment"].values
        ids = meta_df["textID"].values

    else:
        print(f"Processing data for {split_name}...")

        # Filter out neutral tweets for training if configured
        if split_name == "train" and Config.TRAIN_EXCLUDE_NEUTRAL:
            initial_len = len(df)
            df = df[df["sentiment"] != "neutral"].reset_index(drop=True)
            print(f"Filtered out neutral tweets. Rows: {initial_len} -> {len(df)}")

        # Initialize lists
        input_ids_list = []
        attention_mask_list = []
        start_targets_list = []
        end_targets_list = []
        offsets_list = []

        texts_list = []
        selected_texts_list = []
        sentiments_list = []
        ids_list = []

        # Pre-fetch special token IDs
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        for idx, row in df.iterrows():
            # Normalize text first (Critical for alignment)
            text = normalize_text(row["text"])
            sentiment = row["sentiment"]
            text_id = row["textID"]

            # Check if selected_text exists (Train/Val)
            has_selected = "selected_text" in row and pd.notna(row["selected_text"])
            selected_text = normalize_text(row["selected_text"]) if has_selected else ""

            # --- Tokenization Strategy: Early Fusion ---
            # Format: [CLS] Sentiment [SEP] Text [SEP]

            # 1. Tokenize Sentiment
            sentiment_ids = tokenizer.encode(sentiment, add_special_tokens=False)

            # 2. Tokenize Text with offsets
            # We use return_offsets_mapping to get character positions
            encoded_text = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            text_ids = encoded_text["input_ids"]
            text_offsets = encoded_text["offset_mapping"]

            # 3. Construct Input Sequence
            input_ids = [cls_id] + sentiment_ids + [sep_id] + text_ids + [sep_id]

            # 4. Construct Offset Mapping
            # (0,0) for special tokens and sentiment, actual offsets for text
            prefix_len = 1 + len(sentiment_ids) + 1  # [CLS] + sent + [SEP]
            full_offsets = [(0, 0)] * prefix_len + text_offsets + [(0, 0)]

            # 5. Padding
            pad_len = Config.MAX_LEN - len(input_ids)
            mask = [1] * len(input_ids)

            if pad_len > 0:
                input_ids = input_ids + [pad_id] * pad_len
                full_offsets = full_offsets + [(0, 0)] * pad_len
                mask = mask + [0] * pad_len
            else:
                # Truncate if necessary (rare for tweets with MAX_LEN=128)
                input_ids = input_ids[: Config.MAX_LEN]
                full_offsets = full_offsets[: Config.MAX_LEN]
                mask = mask[: Config.MAX_LEN]

            # --- Target Generation ---
            start_target = np.zeros(Config.MAX_LEN)
            end_target = np.zeros(Config.MAX_LEN)

            if has_selected:
                # Find character indices of selected_text in text
                start_char = text.find(selected_text)

                if start_char == -1:
                    # Fallback for data anomalies (should be rare with normalization)
                    start_char = 0
                    end_char = len(text)
                else:
                    end_char = start_char + len(selected_text)

                # Map character indices to token indices
                token_start_idx = prefix_len
                token_end_idx = prefix_len

                # Find Start Token: The token containing the start character
                # Logic: token_start <= char_start < token_end
                found_start = False
                for i in range(prefix_len, len(input_ids)):
                    if input_ids[i] == sep_id or input_ids[i] == pad_id:
                        break

                    t_start, t_end = full_offsets[i]
                    if t_start <= start_char < t_end:
                        token_start_idx = i
                        found_start = True
                        break

                # Find End Token: The token containing the last character (end_char - 1)
                found_end = False
                target_last_char = max(start_char, end_char - 1)

                for i in range(prefix_len, len(input_ids)):
                    if input_ids[i] == sep_id or input_ids[i] == pad_id:
                        break

                    t_start, t_end = full_offsets[i]
                    if t_start <= target_last_char < t_end:
                        token_end_idx = i
                        found_end = True
                        break

                # Generate Gaussian Soft Targets
                start_target = get_soft_targets(
                    Config.MAX_LEN, token_start_idx, Config.SIGMA
                )
                end_target = get_soft_targets(
                    Config.MAX_LEN, token_end_idx, Config.SIGMA
                )

            # Append to lists
            input_ids_list.append(input_ids)
            attention_mask_list.append(mask)
            start_targets_list.append(start_target)
            end_targets_list.append(end_target)
            offsets_list.append(full_offsets)

            texts_list.append(text)
            selected_texts_list.append(selected_text)
            sentiments_list.append(sentiment)
            ids_list.append(text_id)

        # Convert lists to numpy arrays
        input_ids = np.array(input_ids_list)
        attention_mask = np.array(attention_mask_list)
        start_targets = np.array(start_targets_list)
        end_targets = np.array(end_targets_list)
        offsets = np.array(offsets_list)

        texts = np.array(texts_list)
        selected_texts = np.array(selected_texts_list)
        sentiments = np.array(sentiments_list)
        ids = np.array(ids_list)

        # Save to cache
        np.save(paths["input_ids"], input_ids)
        np.save(paths["attention_mask"], attention_mask)
        np.save(paths["start_targets"], start_targets)
        np.save(paths["end_targets"], end_targets)
        np.save(paths["offsets"], offsets)

        # Save metadata as parquet
        meta_df = pd.DataFrame(
            {
                "textID": ids,
                "text": texts,
                "selected_text": selected_texts,
                "sentiment": sentiments,
            }
        )
        meta_df.to_parquet(paths["meta"], index=False)

        print(f"Data processed and cached to {cache_dir}")

    return TweetDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_targets=start_targets,
        end_targets=end_targets,
        offsets=offsets,
        texts=texts,
        selected_texts=selected_texts,
        sentiments=sentiments,
        ids=ids,
    )
