import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from library.config import Config
from library.utils import get_logger, seed_everything

logger = get_logger("data")


class SyntheticCorruptor:
    """
    Responsible for generating synthetic training data by removing words from sentences.
    """

    def __init__(self, seed=Config.SEED):
        self.rng = np.random.RandomState(seed)

    def corrupt(self, sentences):
        """
        Corrupts a list of sentences by removing one word from each.

        Args:
            sentences (list): List of strings.

        Returns:
            pd.DataFrame: DataFrame with columns ['sentence', 'missing_word', 'gap_index']
                          'sentence' is the corrupted sentence.
                          'gap_index' is the index of the word immediately PRECEDING the gap.
        """
        data = []
        for sentence in tqdm(sentences, desc="Corrupting sentences"):
            # Simple whitespace tokenization as per task description
            words = sentence.strip().split()

            # Constraints: never first or last word.
            # Indices: 0 to len-1.
            # Removable: 1 to len-2.
            if len(words) < 3:
                continue

            # Pick index to remove
            remove_idx = self.rng.randint(1, len(words) - 1)

            missing_word = words[remove_idx]

            # Create corrupted sentence
            # Gap is after word at remove_idx - 1
            gap_index = remove_idx - 1

            corrupted_words = words[:remove_idx] + words[remove_idx + 1 :]
            corrupted_sentence = " ".join(corrupted_words)

            data.append(
                {
                    "sentence": corrupted_sentence,
                    "missing_word": missing_word,
                    "gap_index": gap_index,
                }
            )

        return pd.DataFrame(data)


class LocatorDataset(Dataset):
    """
    Dataset for the Locator model (DeBERTa).
    Predicts which token is followed by a missing word.
    """

    def __init__(self, data, tokenizer, max_len=Config.MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["sentence"]
        gap_idx = row["gap_index"]  # Word index before gap

        # Tokenize with offset mapping to align words to tokens
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create label: Binary mask, 1 at the token preceding the gap
        labels = torch.zeros_like(input_ids, dtype=torch.float)

        # Find the token corresponding to the word at gap_idx
        word_ids = encoding.word_ids(batch_index=0)

        # Find the last token index that corresponds to the word at gap_idx
        target_token_idx = -1
        for i, w_id in enumerate(word_ids):
            if w_id == gap_idx:
                target_token_idx = i
            elif w_id is not None and w_id > gap_idx:
                # Optimized: once we pass the word, we can stop
                break

        # Handle truncation case: if the gap is truncated out, label remains 0
        if target_token_idx != -1 and target_token_idx < self.max_len:
            labels[target_token_idx] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InfillerDataset(Dataset):
    """
    Dataset for the In-Filler model (RoBERTa).
    Predicts the missing word given a sentence with a <mask> token.
    """

    def __init__(self, data, tokenizer, max_len=Config.MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_token = tokenizer.mask_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["sentence"]
        gap_idx = row["gap_index"]
        missing_word = row["missing_word"]

        # Insert mask token
        words = text.split()
        # gap_idx is the word BEFORE the gap. So insert AFTER gap_idx.
        words.insert(gap_idx + 1, self.mask_token)
        masked_text = " ".join(words)

        encoding = self.tokenizer(
            masked_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Label: encode the missing word
        # We assume the dataset has been filtered for single-token words
        target_ids = self.tokenizer.encode(missing_word, add_special_tokens=False)
        label_id = target_ids[0] if len(target_ids) > 0 else self.tokenizer.unk_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


class TestDataset(Dataset):
    """
    Dataset for the Test set (Inference).
    """

    def __init__(self, data, tokenizer, max_len=Config.MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["sentence"]
        row_id = row["id"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "id": row_id,
            "text": text,
        }


def process_data(load_cached_data=True):
    """
    Main data processing pipeline.
    Loads raw data, corrupts it, caches it, and returns DataFrames for Locator and Infiller.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache first.

    Returns:
        tuple: (locator_train_df, locator_val_df, infiller_train_df, infiller_val_df)
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(Config.LOCATOR_TRAIN_CACHE)
        and os.path.exists(Config.INFILLER_TRAIN_CACHE)
    ):
        logger.info("Loading cached datasets...")
        locator_train = pd.read_parquet(Config.LOCATOR_TRAIN_CACHE)
        locator_val = pd.read_parquet(Config.LOCATOR_VAL_CACHE)
        infiller_train = pd.read_parquet(Config.INFILLER_TRAIN_CACHE)
        infiller_val = pd.read_parquet(Config.INFILLER_VAL_CACHE)
        return locator_train, locator_val, infiller_train, infiller_val

    logger.info("Cache not found or ignored. Processing data from scratch...")

    # 2. Load Raw Metadata
    logger.info(f"Loading raw train metadata from {Config.TRAIN_METADATA_PATH}")
    df_train_raw = pd.read_parquet(Config.TRAIN_METADATA_PATH)
    df_val_raw = pd.read_parquet(Config.VAL_METADATA_PATH)

    # 3. Sample
    if len(df_train_raw) > Config.TRAIN_SAMPLE_SIZE:
        df_train_raw = df_train_raw.sample(
            n=Config.TRAIN_SAMPLE_SIZE, random_state=Config.SEED
        )
    if len(df_val_raw) > Config.VAL_SAMPLE_SIZE:
        df_val_raw = df_val_raw.sample(
            n=Config.VAL_SAMPLE_SIZE, random_state=Config.SEED
        )

    # 4. Corrupt
    corruptor = SyntheticCorruptor(seed=Config.SEED)

    logger.info("Corrupting training set...")
    train_corrupted = corruptor.corrupt(df_train_raw["sentence"].tolist())

    logger.info("Corrupting validation set...")
    val_corrupted = corruptor.corrupt(df_val_raw["sentence"].tolist())

    # 5. Save Locator Data (Full corrupted sets)
    logger.info(f"Saving Locator cache to {Config.LOCATOR_TRAIN_CACHE}")
    train_corrupted.to_parquet(Config.LOCATOR_TRAIN_CACHE)
    val_corrupted.to_parquet(Config.LOCATOR_VAL_CACHE)

    # 6. Filter for Infiller
    # We need to filter out missing words that tokenize to multiple tokens for the Infiller
    # to ensure clean training for the MLM head (which typically predicts one token).
    logger.info("Filtering data for Infiller (single-token targets)...")

    # Initialize tokenizer for filtering logic
    try:
        infiller_tokenizer = AutoTokenizer.from_pretrained(
            Config.INFILLER_MODEL_NAME, use_fast=True
        )
    except Exception as e:
        logger.warning(
            f"Could not load fast tokenizer: {e}. Falling back to slow tokenizer."
        )
        infiller_tokenizer = AutoTokenizer.from_pretrained(
            Config.INFILLER_MODEL_NAME, use_fast=False
        )

    def is_single_token(word):
        # add_special_tokens=False ensures we just get the word tokens
        ids = infiller_tokenizer.encode(word, add_special_tokens=False)
        return len(ids) == 1

    # Apply filter
    tqdm.pandas(desc="Filtering Train")
    train_mask = train_corrupted["missing_word"].progress_apply(is_single_token)
    infiller_train = train_corrupted[train_mask].copy()

    tqdm.pandas(desc="Filtering Val")
    val_mask = val_corrupted["missing_word"].progress_apply(is_single_token)
    infiller_val = val_corrupted[val_mask].copy()

    logger.info(
        f"Infiller Train size: {len(infiller_train)} (Original: {len(train_corrupted)})"
    )

    # 7. Save Infiller Data
    logger.info(f"Saving Infiller cache to {Config.INFILLER_TRAIN_CACHE}")
    infiller_train.to_parquet(Config.INFILLER_TRAIN_CACHE)
    infiller_val.to_parquet(Config.INFILLER_VAL_CACHE)

    return train_corrupted, val_corrupted, infiller_train, infiller_val
