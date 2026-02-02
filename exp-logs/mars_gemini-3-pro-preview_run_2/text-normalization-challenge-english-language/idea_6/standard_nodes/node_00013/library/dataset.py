import torch
import pandas as pd
import numpy as np
import os
import re
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Any, Optional
from library.config import Config
from library.utils import get_logger, ensure_dir
from library.label_manager import LabelEngineer

logger = get_logger("dataset")


class NormalizationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Fine-Grained Text Normalization.
    Handles tokenization, label alignment, and hard-negative sampling.
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = Config.MAX_LEN,
        debug: bool = Config.DEBUG,
        force_process: bool = False,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
            debug (bool): If True, uses a subset of data.
            force_process (bool): If True, ignores cache and re-processes data.
        """
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.debug = debug

        # Initialize Label Engineer to get raw/labeled data
        self.label_engineer = LabelEngineer()
        self.label_engineer._load_or_create_label_encoder()

        # 1. Load and Group Data
        self.data = self._load_grouped_data(split, debug, force_process)

        # 2. Apply Sampling (Only for Training)
        if split == "train":
            self.data = self._apply_sampling(self.data)

        # 3. Pre-compute/Extract lists for faster indexing
        # Dataframe indexing can be slow in __getitem__
        self.sentences = self.data["before"].tolist()
        self.labels = self.data["label_id"].tolist()
        self.submission_ids = self.data["id"].tolist()

        logger.info(
            f"Dataset {split} initialized with {len(self.sentences)} sentences."
        )

    def _load_grouped_data(
        self, split: str, debug: bool, force_process: bool
    ) -> pd.DataFrame:
        """
        Loads flat data from LabelEngineer, groups it by sentence, and caches the result.
        """
        # Define cache path
        cache_filename = f"{split}_grouped.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # Try loading cache
        if not force_process and not debug and os.path.exists(cache_path):
            try:
                logger.info(f"Loading grouped data from {cache_path}")
                # PyArrow engine required for list columns
                df = pd.read_parquet(cache_path, engine="pyarrow")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-processing.")

        # Process from scratch
        logger.info(f"Grouping {split} data (Debug={debug})...")

        # Get flat data (token-level)
        # label_engineer handles its own caching
        flat_df = self.label_engineer.process_dataset(split, debug=debug)

        # Group by sentence_id
        # Assuming data is sorted by sentence_id, token_id (metadata ensures this)
        # We aggregate 'before', 'label_id', and 'id' into lists

        # Optimization: groupby is faster if we limit columns
        cols_to_keep = ["sentence_id", "before", "label_id", "id"]

        # Grouping
        # Note: 'before' are strings, 'label_id' are ints, 'id' are strings
        grouped_df = (
            flat_df[cols_to_keep]
            .groupby("sentence_id")
            .agg({"before": list, "label_id": list, "id": list})
            .reset_index()
        )

        # Save cache (if not debug)
        if not debug:
            logger.info(f"Saving grouped data to {cache_path}")
            ensure_dir(cache_path)
            grouped_df.to_parquet(cache_path, engine="pyarrow", index=False)

        return grouped_df

    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Hard-Negative Sampling for the training set.
        Keeps all sentences with non-PLAIN transformations.
        Filters PLAIN sentences to keep only 'hard' cases (digits, caps).
        """
        logger.info("Applying Hard-Negative Sampling...")

        # Get ID for TRANS_PLAIN
        # If not found, assume 0, but check registry/encoder
        plain_id = self.label_engineer.name_to_id.get("TRANS_PLAIN", 0)

        # Helper to check if a sentence is purely trivial plain
        # Trivial = All labels are PLAIN AND text contains no digits/caps
        # We can check text content efficiently

        def is_interesting(row):
            labels = row["label_id"]
            tokens = row["before"]

            # Condition 1: Has non-PLAIN label
            # Using set for speed
            if any(l != plain_id for l in labels):
                return True

            # Condition 2: Hard Plain (contains digits or significant caps)
            # We join tokens to check the whole sentence content
            text = "".join(str(t) for t in tokens)

            # Regex: Check for any digit or uppercase letter
            if re.search(r"[0-9A-Z]", text):
                return True

            return False

        # Apply filter
        keep_mask = []
        # Random sampling for trivial cases (keep 5%)
        rng = np.random.RandomState(Config.SEED)

        for idx, row in df.iterrows():
            if is_interesting(row):
                keep_mask.append(True)
            else:
                # Trivial plain (lowercase, no numbers)
                # Keep small fraction to maintain distribution
                keep_mask.append(rng.rand() < 0.05)

        filtered_df = df[keep_mask].copy()

        logger.info(
            f"Sampling complete. Reduced from {len(df)} to {len(filtered_df)} sentences."
        )
        return filtered_df

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'input_ids': tensor,
                'attention_mask': tensor,
                'labels': tensor,
                'word_ids': list (for mapping back),
                'raw_tokens': list (original tokens),
                'submission_ids': list (ids for submission)
            }
        """
        tokens = self.sentences[idx]
        label_ids = self.labels[idx]
        sub_ids = self.submission_ids[idx]

        # Tokenize
        # is_split_into_words=True indicates input is already list of tokens
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Align labels
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First sub-token of a word -> assign label
                # Ensure word_idx is within bounds (truncation might cut off end)
                if word_idx < len(label_ids):
                    aligned_labels.append(label_ids[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                # Subsequent sub-tokens -> ignore
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        # Convert to tensor
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        # Squeeze batch dimension added by return_tensors='pt'
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = labels_tensor

        # Metadata for inference/submission reconstruction
        # We return lists; the DataLoader's collate_fn should handle them (e.g., as lists of lists)
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            # Helper data (not used for backprop)
            "word_ids": [w if w is not None else -1 for w in word_ids],
            "raw_tokens": tokens,
            "submission_ids": sub_ids,
        }
