import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from scipy.stats import norm
from library.config import Config
from library.utils import normalize_text


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Holds pre-processed tensors and metadata.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
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
        self.token_type_ids = token_type_ids
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets
        self.texts = texts
        self.selected_texts = selected_texts
        self.sentiments = sentiments
        self.ids = ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "text": self.texts[item],
            "sentiment": self.sentiments[item],
            "textID": self.ids[item],
        }

        # Include targets if they exist (Training/Validation)
        if self.start_targets is not None:
            data["start_targets"] = torch.tensor(
                self.start_targets[item], dtype=torch.float
            )
            data["end_targets"] = torch.tensor(
                self.end_targets[item], dtype=torch.float
            )
            data["selected_text"] = self.selected_texts[item]

        return data


def generate_gaussian_target(center, length, sigma):
    """
    Generates a Gaussian distribution centered at 'center' with standard deviation 'sigma'.
    Normalized to sum to 1.
    """
    if center < 0 or center >= length:
        # Fallback for invalid centers (should not happen with correct processing)
        return np.zeros(length)

    x = np.arange(length)
    gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    # Normalize to create a probability distribution
    return gaussian / gaussian.sum()


def process_data(df, tokenizer, max_len, is_test=False):
    """
    Processes the DataFrame into tensors required for the model.
    """
    # Initialize lists
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_targets_list = []
    end_targets_list = []
    offsets_list = []

    # Metadata lists
    texts_list = []
    selected_texts_list = []
    sentiments_list = []
    ids_list = []

    # Iterate over dataframe
    for idx, row in df.iterrows():
        # 1. Normalize Text (Protocol: Normalize-First)
        text = normalize_text(row["text"])
        sentiment = row["sentiment"]
        text_id = row["textID"]

        # Handle Selected Text
        selected_text = ""
        if not is_test:
            selected_text = normalize_text(row["selected_text"])

        # 2. Tokenize
        # DeBERTa tokenizer handles special tokens automatically
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]
        offsets = encoded["offset_mapping"]

        # 3. Generate Targets (Train/Val only)
        start_target = np.zeros(max_len)
        end_target = np.zeros(max_len)

        if not is_test:
            # Find character indices of selected_text in text
            # We search in the normalized text
            start_char_idx = text.find(selected_text)
            end_char_idx = start_char_idx + len(selected_text)

            # Identify token span
            token_start_idx = 0
            token_end_idx = 0
            found_start = False

            # Iterate through offsets to find the tokens corresponding to the character span
            # Offsets are tuples (start_char, end_char)
            # We skip special tokens (offset (0,0) usually, but check input_ids)

            # DeBERTa v3 uses SentencePiece, offsets can be tricky.
            # We look for the first token that overlaps with the start of the selection
            # and the last token that overlaps with the end.

            tokens_in_span = []

            for i, (o_start, o_end) in enumerate(offsets):
                # Skip padding or special tokens that have 0 length (except [CLS] etc which might be 0,0)
                if o_start == o_end and i != 0:
                    continue

                # Check overlap
                # Token interval: [o_start, o_end)
                # Selection interval: [start_char_idx, end_char_idx)

                # Intersection
                inter_start = max(o_start, start_char_idx)
                inter_end = min(o_end, end_char_idx)

                if inter_start < inter_end:
                    tokens_in_span.append(i)

            if len(tokens_in_span) > 0:
                token_start_idx = tokens_in_span[0]
                token_end_idx = tokens_in_span[-1]

                # Gaussian Targets
                start_target = generate_gaussian_target(
                    token_start_idx, max_len, Config.SMOOTHING_SIGMA
                )
                end_target = generate_gaussian_target(
                    token_end_idx, max_len, Config.SMOOTHING_SIGMA
                )

                # Binary Mask Target (Auxiliary Head)
                mask_target[token_start_idx : token_end_idx + 1] = 1.0
            else:
                # Fallback: if alignment fails (rare), point to CLS or ignore
                # For safety, we leave them as zeros (or uniform), but usually this implies bad data.
                # Given strict normalization, this should be minimal.
                pass

        # Append to lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        start_targets_list.append(start_target)
        end_targets_list.append(end_target)
        offsets_list.append(offsets)

        texts_list.append(text)
        selected_texts_list.append(selected_text)
        sentiments_list.append(sentiment)
        ids_list.append(text_id)

    return (
        np.array(input_ids_list),
        np.array(attention_mask_list),
        np.array(token_type_ids_list),
        np.array(start_targets_list),
        np.array(end_targets_list),
        np.array(offsets_list),
        texts_list,
        selected_texts_list,
        sentiments_list,
        ids_list,
    )


def load_and_cache_data(subset="train", load_cached_data=True):
    """
    Main function to load data. Implements caching strategy.

    Args:
        subset (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        TweetDataset: The prepared dataset.
    """
    # Determine paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Naming convention for cache files
    prefix = f"{subset}"
    if subset == "train" and Config.FILTER_NEUTRAL:
        prefix += "_no_neutral"

    cache_files = {
        "input_ids": os.path.join(cache_dir, f"{prefix}_input_ids.npy"),
        "attention_mask": os.path.join(cache_dir, f"{prefix}_attention_mask.npy"),
        "token_type_ids": os.path.join(cache_dir, f"{prefix}_token_type_ids.npy"),
        "start_targets": os.path.join(cache_dir, f"{prefix}_start_targets.npy"),
        "end_targets": os.path.join(cache_dir, f"{prefix}_end_targets.npy"),
        "offsets": os.path.join(cache_dir, f"{prefix}_offsets.npy"),
        "meta": os.path.join(cache_dir, f"{prefix}_meta.parquet"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading {subset} data from cache...")
        input_ids = np.load(cache_files["input_ids"])
        attention_mask = np.load(cache_files["attention_mask"])
        token_type_ids = np.load(cache_files["token_type_ids"])
        start_targets = np.load(cache_files["start_targets"])
        end_targets = np.load(cache_files["end_targets"])
        offsets = np.load(cache_files["offsets"])

        # Load metadata
        meta_df = pd.read_parquet(cache_files["meta"])
        texts = meta_df["text"].tolist()
        selected_texts = meta_df["selected_text"].tolist()
        sentiments = meta_df["sentiment"].tolist()
        ids = meta_df["textID"].tolist()

    else:
        print(f"Processing {subset} data from scratch...")

        # Load Raw Data
        if subset == "train":
            file_path = Config.TRAIN_META
        elif subset == "val":
            file_path = Config.VAL_META
        elif subset == "test":
            file_path = Config.TEST_META
        else:
            raise ValueError(f"Unknown subset: {subset}")

        df = pd.read_csv(file_path)

        # Filter Neutrals (Train only, based on Config)
        if subset == "train" and Config.FILTER_NEUTRAL:
            initial_len = len(df)
            df = df[df["sentiment"] != "neutral"].reset_index(drop=True)
            print(f"Filtered neutral tweets: {initial_len} -> {len(df)}")

        # Debugging
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)
            print(f"DEBUG MODE: Reduced dataset to {len(df)} samples.")

        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

        # Process
        is_test = subset == "test"
        (
            input_ids,
            attention_mask,
            token_type_ids,
            start_targets,
            end_targets,
            offsets,
            texts,
            selected_texts,
            sentiments,
            ids,
        ) = process_data(df, tokenizer, Config.MAX_LEN, is_test=is_test)

        # Save to Cache
        print(f"Saving {subset} data to cache at {cache_dir}...")
        np.save(cache_files["input_ids"], input_ids)
        np.save(cache_files["attention_mask"], attention_mask)
        np.save(cache_files["token_type_ids"], token_type_ids)
        np.save(cache_files["start_targets"], start_targets)
        np.save(cache_files["end_targets"], end_targets)
        np.save(cache_files["offsets"], offsets)

        # Save Metadata
        meta_df = pd.DataFrame(
            {
                "textID": ids,
                "text": texts,
                "selected_text": selected_texts,
                "sentiment": sentiments,
            }
        )
        meta_df.to_parquet(cache_files["meta"], index=False)

    # For test set, targets are placeholders (zeros), but we return them for consistency
    if subset == "test":
        start_targets = None
        end_targets = None

    return TweetDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        start_targets=start_targets,
        end_targets=end_targets,
        offsets=offsets,
        texts=texts,
        selected_texts=selected_texts,
        sentiments=sentiments,
        ids=ids,
    )
