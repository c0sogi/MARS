import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from library.config import Config
from library.vocabulary import WordVocabulary


def load_and_cache_data(
    source_path, cache_path, load_cached=True, debug=False, debug_size=1000
):
    """
    Loads data from source_path. Implements caching mechanism using Parquet.

    Args:
        source_path (str): Path to the original metadata parquet file.
        cache_path (str): Path to save/load the cached parquet file.
        load_cached (bool): If True, attempts to load from cache first.
        debug (bool): If True, limits the dataset size.
        debug_size (int): Number of samples to load in debug mode.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try loading from cache
    if load_cached and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if debug:
                df = df.iloc[:debug_size]
            return df
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from source."
            )

    # 2. Load from source
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source data not found at {source_path}")

    df = pd.read_parquet(source_path)

    # 3. Save to cache (save full dataset before debug slicing)
    try:
        df.to_parquet(cache_path)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    # Apply debug slicing if needed
    if debug:
        df = df.iloc[:debug_size]

    return df


class InfillingDataset(Dataset):
    def __init__(
        self,
        split: str,
        vocabulary: WordVocabulary,
        tokenizer: PreTrainedTokenizerBase = None,
        load_cached: bool = True,
    ):
        """
        Dataset for the Dual-Head Infilling Task.

        Args:
            split (str): 'train', 'val', or 'test'.
            vocabulary (WordVocabulary): Instance for encoding target words.
            tokenizer (PreTrainedTokenizerBase, optional): HF Tokenizer.
            load_cached (bool): Whether to use cached data.
        """
        self.split = split
        self.vocabulary = vocabulary
        self.max_len = Config.MAX_SEQ_LEN
        self.debug = Config.DEBUG

        # Set fixed random seed for reproducibility in this module
        np.random.seed(Config.SEED)

        # Initialize Tokenizer
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            # Disable parallelism to avoid deadlocks in DataLoaders
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME, use_fast=True
            )

        # Determine paths
        if split == "train":
            source = Config.TRAIN_DATA_PATH
            cache = Config.TRAIN_CACHE_PATH
        elif split == "val":
            source = Config.VAL_DATA_PATH
            cache = Config.VAL_CACHE_PATH
        elif split == "test":
            source = Config.TEST_DATA_PATH
            cache = Config.TEST_CACHE_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load Data
        self.data = load_and_cache_data(
            source,
            cache,
            load_cached=load_cached,
            debug=self.debug,
            debug_size=Config.DEBUG_SAMPLE_SIZE,
        )

        # Pre-extract lists for faster access
        self.sentences = self.data["sentence"].tolist()
        if "id" in self.data.columns:
            self.ids = self.data["id"].tolist()
        else:
            self.ids = list(range(len(self.sentences)))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # ------------------------------------------------------------------
        # Test Mode: Inference (Word already removed)
        # ------------------------------------------------------------------
        if self.split == "test":
            encoding = self.tokenizer(
                sentence,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=False,
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "id": self.ids[idx],
                "original_text": sentence,
            }

        # ------------------------------------------------------------------
        # Train/Val Mode: Dynamic Corruption
        # ------------------------------------------------------------------
        else:
            words = sentence.split()

            # Constraint: "Never the first or last word". Need at least 3 words.
            if len(words) < 3:
                # Fallback for extremely short sentences (rare): return dummy padded sample
                return {
                    "input_ids": torch.full(
                        (self.max_len,), self.tokenizer.pad_token_id, dtype=torch.long
                    ),
                    "attention_mask": torch.zeros(self.max_len, dtype=torch.long),
                    "loc_label": torch.zeros(self.max_len, dtype=torch.float),
                    "word_label": torch.tensor(0, dtype=torch.long),  # Pad ID
                }

            # 1. Randomly remove a word (excluding first and last)
            remove_idx = np.random.randint(1, len(words) - 1)
            target_word_str = words[remove_idx]

            # 2. Get Target Word Label
            target_word_id = self.vocabulary.token_to_id(target_word_str)

            # 3. Reconstruct Sentence
            prefix_words = words[:remove_idx]
            suffix_words = words[remove_idx + 1 :]

            prefix_str = " ".join(prefix_words)
            suffix_str = " ".join(suffix_words)
            corrupted_sentence = f"{prefix_str} {suffix_str}"

            # 4. Tokenize with Offsets
            encoding = self.tokenizer(
                corrupted_sentence,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            offsets = encoding["offset_mapping"].squeeze(0)

            # 5. Determine Location Label
            # We want to label the token that corresponds to the word *before* the gap.
            # That word ends at char index: len(prefix_str) - 1
            char_idx_to_find = len(prefix_str) - 1

            loc_token_idx = -1
            seq_len = attention_mask.sum().item()

            # Find the token containing the last character of the prefix
            for i in range(seq_len):
                start, end = offsets[i]
                if start <= char_idx_to_find < end:
                    loc_token_idx = i
                    break

            # Create Binary Location Vector
            loc_label = torch.zeros(self.max_len, dtype=torch.float)
            if loc_token_idx != -1 and loc_token_idx < self.max_len:
                loc_label[loc_token_idx] = 1.0

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loc_label": loc_label,
                "word_label": torch.tensor(target_word_id, dtype=torch.long),
            }


def collate_fn(batch):
    """
    Collate function to stack tensors and handle variable keys between train/test.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    result = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Training/Val keys
    if "loc_label" in batch[0]:
        result["loc_label"] = torch.stack([item["loc_label"] for item in batch])
        result["word_label"] = torch.stack([item["word_label"] for item in batch])

    # Test keys
    if "id" in batch[0]:
        result["id"] = [item["id"] for item in batch]

    if "original_text" in batch[0]:
        result["original_text"] = [item["original_text"] for item in batch]

    return result
