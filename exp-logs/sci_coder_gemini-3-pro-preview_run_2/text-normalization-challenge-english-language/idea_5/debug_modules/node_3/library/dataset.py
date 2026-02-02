import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict, List, Optional, Tuple, Union

from library.config import Config
from library.utils import get_logger

logger = get_logger()


def get_tokenizer() -> PreTrainedTokenizerBase:
    """
    Initializes the tokenizer for the model defined in Config.
    Sets add_prefix_space=True which is often required for RoBERTa
    when processing pre-tokenized lists of words.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_NAME, add_prefix_space=True)


class TextNormalizationDataset(Dataset):
    """
    PyTorch Dataset for Transformer-CRF Sequence Labeling.
    Handles tokenization, label alignment, and tensor creation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        label_map: Dict[str, int],
        max_len: int = Config.MAX_LEN,
        is_test: bool = False,
    ):
        """
        Args:
            data (pd.DataFrame): Grouped dataframe where each row is a sentence.
                                 Must contain 'before' (list of tokens).
                                 Train/Val must contain 'class' (list of labels).
            tokenizer: HuggingFace tokenizer.
            label_map: Dictionary mapping class names to IDs.
            max_len: Maximum sequence length.
            is_test: If True, dummy labels are generated.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
        self.is_test = is_test

        # Pre-convert columns to lists to avoid overhead in __getitem__
        self.tokens_list = self.data["before"].tolist()
        self.ids_list = self.data["id"].tolist()

        if not self.is_test:
            self.labels_list = self.data["class"].tolist()
        else:
            self.labels_list = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Retrieve raw tokens (words) for the sentence
        words = self.tokens_list[index]

        # Tokenize
        # is_split_into_words=True tells tokenizer that input is already split by whitespace
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return python lists, convert to tensor later
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Prepare labels
        labels = []

        if not self.is_test:
            raw_labels = self.labels_list[index]
            word_ids = encoding.word_ids()

            previous_word_idx = None

            for word_idx in word_ids:
                # Special tokens (None) or padding
                if word_idx is None:
                    labels.append(-100)
                # New word start -> assign label
                elif word_idx != previous_word_idx:
                    try:
                        label_str = raw_labels[word_idx]
                        label_id = self.label_map.get(
                            label_str, 0
                        )  # Default to PLAIN (0) if unknown
                        labels.append(label_id)
                    except IndexError:
                        # Fallback for truncation edge cases
                        labels.append(-100)
                # Same word (sub-token) -> ignore
                else:
                    labels.append(-100)

                previous_word_idx = word_idx
        else:
            # For test set, fill labels with -100 (ignored index)
            # Length must match input_ids
            labels = [-100] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # We don't return 'id' tensors here as they are strings,
            # inference loop will handle mapping back via the dataframe index if needed.
        }


def _apply_strategic_sampling(df_grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the training data to address class imbalance.
    1. Keeps all sentences with at least one non-PLAIN/PUNCT token.
    2. Keeps 'Hard' PLAIN sentences (containing digits).
    3. Downsamples 'Trivial' PLAIN sentences.
    """
    logger.info(f"Applying strategic sampling. Original sentences: {len(df_grouped)}")

    # Helper to check if a list of classes contains interesting labels
    def has_interesting_class(classes):
        # Convert to set for O(1) lookup
        unique_classes = set(classes)
        # If there is anything other than PLAIN or PUNCT, keep it
        return not unique_classes.issubset({"PLAIN", "PUNCT"})

    # Helper to check for digits in the raw text (Hard Negative)
    def is_hard_plain(tokens):
        # Join tokens to check simple regex
        text = " ".join(tokens)
        # If it contains a digit, it might be ambiguous (e.g. "Chapter 5")
        return bool(re.search(r"\d", text))

    # Apply masks
    # 1. Interesting classes
    mask_interesting = df_grouped["class"].apply(has_interesting_class)

    # 2. Hard Plain (only check rows that aren't already interesting)
    # We look at the 'before' column
    mask_hard_plain = (~mask_interesting) & df_grouped["before"].apply(is_hard_plain)

    # 3. Trivial Plain
    mask_trivial = (~mask_interesting) & (~mask_hard_plain)

    # Select subsets
    df_interesting = df_grouped[mask_interesting]
    df_hard = df_grouped[mask_hard_plain]
    df_trivial = df_grouped[mask_trivial]

    # Downsample trivial
    if not df_trivial.empty:
        df_trivial_sampled = df_trivial.sample(
            frac=Config.TRIVIAL_PLAIN_KEEP_RATE, random_state=Config.SEED
        )
    else:
        df_trivial_sampled = pd.DataFrame()

    logger.info(f"  Interesting sentences (Keep 100%): {len(df_interesting)}")
    logger.info(f"  Hard PLAIN sentences  (Keep 100%): {len(df_hard)}")
    logger.info(
        f"  Trivial sentences     (Sampled)  : {len(df_trivial_sampled)} (from {len(df_trivial)})"
    )

    # Combine
    df_final = (
        pd.concat([df_interesting, df_hard, df_trivial_sampled])
        .sample(frac=1, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    logger.info(f"Final training set size: {len(df_final)} sentences.")
    return df_final


def load_data(
    split: str = "train", load_cached_data: bool = True, debug: bool = Config.DEBUG
) -> pd.DataFrame:
    """
    Loads, processes, and caches the dataset.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from parquet cache.
        debug (bool): If True, limits data size.

    Returns:
        pd.DataFrame: Grouped dataframe ready for Dataset class.
    """
    # Determine paths
    if split == "train":
        meta_path = Config.TRAIN_META_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif split == "val":
        meta_path = Config.VAL_META_PATH
        cache_path = Config.VAL_CACHE_PATH
    else:
        meta_path = Config.TEST_META_PATH
        cache_path = Config.TEST_CACHE_PATH

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading {split} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            if debug:
                df = df.head(Config.DEBUG_SAMPLE_SIZE)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load Source
    logger.info(f"Loading raw metadata for {split} from {meta_path}...")
    # keep_default_na=False is critical for text tokens like "null"
    df_raw = pd.read_csv(
        meta_path, keep_default_na=False, dtype={"sentence_id": int, "token_id": int}
    )

    # 3. Group by Sentence
    logger.info(f"Grouping {split} data by sentence_id...")

    # Define aggregation schema
    agg_dict = {"before": list, "id": list}
    if "class" in df_raw.columns:
        agg_dict["class"] = list

    # Ensure correct order of tokens
    df_raw = df_raw.sort_values(["sentence_id", "token_id"])

    # Group
    df_grouped = df_raw.groupby("sentence_id").agg(agg_dict).reset_index()

    # 4. Apply Sampling (Train only)
    if split == "train":
        df_grouped = _apply_strategic_sampling(df_grouped)

    # 5. Save Cache
    logger.info(f"Saving processed {split} data to {cache_path}...")
    # Parquet handles lists in columns efficiently via PyArrow
    df_grouped.to_parquet(cache_path, engine="pyarrow", index=False)

    if debug:
        df_grouped = df_grouped.head(Config.DEBUG_SAMPLE_SIZE)

    return df_grouped
