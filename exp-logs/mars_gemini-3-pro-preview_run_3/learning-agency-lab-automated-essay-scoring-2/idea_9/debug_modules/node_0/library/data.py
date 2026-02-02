import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger, get_cache_path

# Initialize logger
logger = get_logger("data")


class EssayDataset(Dataset):
    """
    Dataset class for Essay Scoring.
    Handles input_ids, attention_masks, scores, and essay_ids.
    """

    def __init__(self, input_ids, attention_mask, scores=None, essay_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.scores = scores
        self.essay_ids = essay_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.scores is not None:
            # Return float for regression.
            # Ordinal classification heads can convert this to binary targets internally.
            item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)

        if self.essay_ids is not None:
            item["essay_ids"] = str(self.essay_ids[idx])

        return item


def load_data_from_metadata(split="train"):
    """
    Loads the dataset from the generated metadata CSV files.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split name: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def preprocess_and_cache(
    df, split_name, tokenizer, max_length=None, load_cached_data=True
):
    """
    Tokenizes the text data using a sliding window approach and caches the results.

    Args:
        df (pd.DataFrame): Dataframe containing 'full_text' and 'essay_id'.
        split_name (str): Identifier for the split (used for cache naming).
        tokenizer: HuggingFace tokenizer instance.
        max_length (int): Maximum sequence length.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        dict: Dictionary containing numpy arrays for inputs and targets.
    """
    if max_length is None:
        max_length = Config.MAX_LENGTH

    # Create a unique configuration object for hashing
    cache_config = {
        "split": split_name,
        "model_path": Config.MODEL_PATH,
        "max_length": max_length,
        "stride": 256,
        "num_samples": len(df),
    }

    # Define cache file paths
    cache_files = {
        "input_ids": get_cache_path(f"{split_name}_input_ids", cache_config),
        "attention_mask": get_cache_path(f"{split_name}_attention_mask", cache_config),
        "essay_ids": get_cache_path(f"{split_name}_essay_ids", cache_config),
        "scores": get_cache_path(f"{split_name}_scores", cache_config),
    }

    # 1. Try Loading from Cache
    if load_cached_data:
        files_exist = (
            os.path.exists(cache_files["input_ids"])
            and os.path.exists(cache_files["attention_mask"])
            and os.path.exists(cache_files["essay_ids"])
        )
        # If scores are expected, check for them too
        if "score" in df.columns and not os.path.exists(cache_files["scores"]):
            files_exist = False

        if files_exist:
            logger.info(f"Loading cached data for '{split_name}'...")
            try:
                data = {
                    "input_ids": np.load(cache_files["input_ids"]),
                    "attention_mask": np.load(cache_files["attention_mask"]),
                    "essay_ids": np.load(cache_files["essay_ids"], allow_pickle=True),
                }
                if "score" in df.columns:
                    data["scores"] = np.load(cache_files["scores"])
                else:
                    data["scores"] = None
                return data
            except Exception as e:
                logger.warning(
                    f"Failed to load cache for '{split_name}': {e}. Re-processing."
                )
        else:
            logger.info(f"Cache miss for '{split_name}'. Processing from scratch...")

    # 2. Process Data
    logger.info(
        f"Tokenizing {len(df)} texts for '{split_name}' with max_len={max_length}..."
    )

    texts = df["full_text"].astype(str).tolist()
    essay_ids = df["essay_id"].tolist()
    scores = df["score"].tolist() if "score" in df.columns else None

    # Tokenize with sliding window
    # return_overflowing_tokens=True creates multiple samples for long texts
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        stride=256,
        return_overflowing_tokens=True,
        return_tensors="np",
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Map the new chunks back to the original sample index
    sample_map = tokenized["overflow_to_sample_mapping"]

    # Expand metadata to match the chunks
    expanded_essay_ids = np.array([essay_ids[i] for i in sample_map])

    expanded_scores = None
    if scores is not None:
        expanded_scores = np.array([scores[i] for i in sample_map])

    # 3. Save to Cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_files["input_ids"], input_ids)
    np.save(cache_files["attention_mask"], attention_mask)
    np.save(cache_files["essay_ids"], expanded_essay_ids)
    if expanded_scores is not None:
        np.save(cache_files["scores"], expanded_scores)

    logger.info(f"Processed '{split_name}': {len(input_ids)} chunks generated.")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "essay_ids": expanded_essay_ids,
        "scores": expanded_scores,
    }


def get_dataloaders(
    df,
    split_name,
    batch_size=Config.TRAIN_BATCH_SIZE,
    shuffle=False,
    load_cached_data=True,
    max_length=None,
):
    """
    Generates a DataLoader for the provided dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        split_name (str): Name for cache identification.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        load_cached_data (bool): Whether to use cached pre-processed data.
        max_length (int, optional): Tokenizer max length. Defaults to Config.MAX_LENGTH.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    if max_length is None:
        max_length = Config.MAX_LENGTH

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

    # Preprocess (or load from cache)
    data = preprocess_and_cache(
        df,
        split_name,
        tokenizer,
        max_length=max_length,
        load_cached_data=load_cached_data,
    )

    # Create Dataset
    dataset = EssayDataset(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        scores=data["scores"],
        essay_ids=data["essay_ids"],
    )

    # Create DataLoader
    # Note: We do not need a custom collate_fn because the Dataset returns
    # standard tensors that stack nicely.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=(shuffle and len(dataset) > batch_size),
    )

    return loader


def get_test_dataloader(batch_size=Config.VALID_BATCH_SIZE):
    """
    Helper function to get the test set dataloader specifically.
    Uses INFERENCE_MAX_LENGTH.
    """
    df_test = load_data_from_metadata("test")

    return get_dataloaders(
        df_test,
        "test_set",
        batch_size=batch_size,
        shuffle=False,
        load_cached_data=True,
        max_length=Config.INFERENCE_MAX_LENGTH,
    )
