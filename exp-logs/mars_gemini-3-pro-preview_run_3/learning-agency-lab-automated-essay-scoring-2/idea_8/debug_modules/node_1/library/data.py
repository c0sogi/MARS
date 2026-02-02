import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List, Dict, Tuple, Optional, Union
import gc

from library.config import Config
from library.utils import get_logger, get_hash

# Initialize logger
logger = get_logger(os.path.join(Config.WORKING_DIR, "output", "data.log"))


def get_meta_features(texts: pd.Series) -> np.ndarray:
    """
    Calculates explicit scalar features for the essays.
    Features:
    1. Character Count
    2. Word Count
    3. Sentence Count (approximate)
    4. Unique Word Ratio
    """
    logger.info("Generating meta-features...")

    # Ensure string type
    texts = texts.astype(str).fillna("")

    # Vectorized operations where possible, apply for complex ones
    char_counts = texts.apply(len).values

    def get_stats(text):
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return 0, 0, 0

        # Approximate sentence count
        sentences = text.replace("?", ".").replace("!", ".").split(".")
        sentence_count = len([s for s in sentences if s.strip()])

        # Unique word ratio
        unique_ratio = len(set(words)) / word_count

        return word_count, sentence_count, unique_ratio

    stats = texts.apply(get_stats)

    word_counts = np.array([x[0] for x in stats])
    sent_counts = np.array([x[1] for x in stats])
    unique_ratios = np.array([x[2] for x in stats])

    # Stack features: (N, 4)
    meta_features = np.column_stack(
        [char_counts, word_counts, sent_counts, unique_ratios]
    )

    # Replace NaNs or Infs if any (though unlikely with logic above)
    meta_features = np.nan_to_num(meta_features)

    return meta_features.astype(np.float32)


def preprocess_and_cache(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    cache_prefix: str,
    load_cached_data: bool,
) -> Dict[str, np.ndarray]:
    """
    Tokenizes data using sliding window approach and caches the result to disk.
    Returns dictionary containing input_ids, attention_mask, meta_features, etc.
    """
    # Create cache directory
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Generate a unique hash for this configuration to ensure cache validity
    config_dict = {
        "model": Config.MODEL_NAME,
        "max_len": Config.MAX_LENGTH,
        "stride": Config.WINDOW_STRIDE,
        "data_len": len(df),
        "first_id": df.iloc[0]["essay_id"] if not df.empty else "empty",
    }
    config_hash = get_hash(config_dict)

    # Define file paths
    files = {
        "input_ids": os.path.join(
            Config.CACHE_DIR, f"{cache_prefix}_input_ids_{config_hash}.npy"
        ),
        "attention_mask": os.path.join(
            Config.CACHE_DIR, f"{cache_prefix}_attention_mask_{config_hash}.npy"
        ),
        "sample_map": os.path.join(
            Config.CACHE_DIR, f"{cache_prefix}_sample_map_{config_hash}.npy"
        ),
        "meta_features": os.path.join(
            Config.CACHE_DIR, f"{cache_prefix}_meta_features_{config_hash}.npy"
        ),
        "scores": os.path.join(
            Config.CACHE_DIR, f"{cache_prefix}_scores_{config_hash}.npy"
        ),
    }

    # Check if all files exist
    all_exist = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_exist:
        logger.info(
            f"Loading cached data for {cache_prefix} from {Config.CACHE_DIR}..."
        )
        data = {k: np.load(v) for k, v in files.items()}
        return data

    logger.info(f"Processing data for {cache_prefix} (Cache miss or force reload)...")

    # 1. Meta Features
    meta_features = get_meta_features(df["full_text"])

    # 2. Tokenization with Sliding Window
    # We process in batches to manage memory
    all_input_ids = []
    all_attention_masks = []
    all_sample_maps = []  # Maps chunk -> original sample index

    texts = df["full_text"].tolist()
    batch_size = 1000

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            max_length=Config.MAX_LENGTH,
            stride=Config.WINDOW_STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # overflow_to_sample_mapping maps chunk -> index within the batch_texts list
        # We need to adjust it to be the global index
        sample_mapping = encoded["overflow_to_sample_mapping"] + i

        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])
        all_sample_maps.append(sample_mapping)

        if (i // batch_size) % 5 == 0:
            logger.info(
                f"Tokenized {min(i + batch_size, len(texts))}/{len(texts)} texts..."
            )

    # Concatenate all chunks
    input_ids = np.concatenate(all_input_ids, axis=0)
    attention_mask = np.concatenate(all_attention_masks, axis=0)
    sample_map = np.concatenate(all_sample_maps, axis=0)

    # 3. Scores
    if "score" in df.columns:
        scores = df["score"].values.astype(np.float32)
    else:
        # For test set, fill with -1
        scores = np.full(len(df), -1.0, dtype=np.float32)

    # Save to cache
    np.save(files["input_ids"], input_ids)
    np.save(files["attention_mask"], attention_mask)
    np.save(files["sample_map"], sample_map)
    np.save(files["meta_features"], meta_features)
    np.save(files["scores"], scores)

    logger.info(f"Data processed and saved to {Config.CACHE_DIR}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sample_map": sample_map,
        "meta_features": meta_features,
        "scores": scores,
    }


class EssayDataset(Dataset):
    """
    Dataset that handles sliding window chunks.
    Since one essay can have multiple chunks, __getitem__ returns all chunks for that essay.
    """

    def __init__(self, data_dict: Dict[str, np.ndarray], num_samples: int):
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.sample_map = data_dict["sample_map"]
        self.meta_features = data_dict["meta_features"]
        self.scores = data_dict["scores"]
        self.num_samples = num_samples

        # Pre-compute start and count indices for fast retrieval
        # This avoids using np.where in __getitem__ which is slow
        self.sample_indices = [[] for _ in range(num_samples)]

        # Iterate once to build the map
        # sample_map is sorted by sample index because we processed sequentially
        # but let's be safe and efficient
        unique_ids, counts = np.unique(self.sample_map, return_counts=True)

        # We assume sample_map is contiguous blocks for each sample
        # (e.g. 0,0,0,1,1,2,2,2,2...) which is true from tokenizer output
        # We can just store start_idx and count
        self.offsets = np.zeros((num_samples, 2), dtype=np.int32)

        current_idx = 0
        for sample_id in range(num_samples):
            # Find count for this sample
            # Since unique_ids might skip if a sample was somehow empty (unlikely with padding),
            # we handle strict mapping
            if sample_id in unique_ids:
                # Find where in unique_ids this sample_id is
                loc = np.searchsorted(unique_ids, sample_id)
                count = counts[loc]
                self.offsets[sample_id] = [current_idx, count]
                current_idx += count
            else:
                # Should not happen with standard tokenizer settings
                self.offsets[sample_id] = [0, 0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start, count = self.offsets[idx]

        if count == 0:
            # Fallback for empty text (should not occur)
            return {
                "input_ids": np.zeros((1, Config.MAX_LENGTH), dtype=np.int64),
                "attention_mask": np.zeros((1, Config.MAX_LENGTH), dtype=np.int64),
                "meta_features": self.meta_features[idx],
                "score": self.scores[idx],
                "num_chunks": 0,
            }

        return {
            "input_ids": self.input_ids[start : start + count],
            "attention_mask": self.attention_mask[start : start + count],
            "meta_features": self.meta_features[idx],
            "score": self.scores[idx],
            "num_chunks": count,
        }


class CollateFn:
    """
    Custom collate function to handle variable number of chunks per essay.
    Flattens the batch so the model sees a large batch of chunks.
    """

    def __call__(self, batch):
        # batch is a list of dicts from __getitem__

        input_ids_list = []
        attention_mask_list = []
        meta_features_list = []
        scores_list = []
        batch_ids_list = []  # To map chunk -> sample index in batch

        for i, sample in enumerate(batch):
            chunks_count = sample["num_chunks"]
            if chunks_count == 0:
                continue

            input_ids_list.append(torch.tensor(sample["input_ids"], dtype=torch.long))
            attention_mask_list.append(
                torch.tensor(sample["attention_mask"], dtype=torch.long)
            )

            # Repeat meta features and scores is NOT needed for model input usually,
            # but we need meta features per sample.
            # We will return meta_features as (Batch_Size, Features)
            meta_features_list.append(sample["meta_features"])
            scores_list.append(sample["score"])

            # Map these chunks to sample i
            batch_ids_list.append(torch.full((chunks_count,), i, dtype=torch.long))

        # Concatenate all chunks
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        batch_ids = torch.cat(batch_ids_list, dim=0)

        # Stack sample-level data
        meta_features = torch.tensor(np.array(meta_features_list), dtype=torch.float32)
        scores = torch.tensor(np.array(scores_list), dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "batch_ids": batch_ids,
            "meta_features": meta_features,
            "labels": scores,
        }


def get_dataloaders(
    train_batch_size: int = Config.TRAIN_BATCH_SIZE,
    valid_batch_size: int = Config.VALID_BATCH_SIZE,
    load_cached_data: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Main entry point to get DataLoaders.
    """
    logger.info("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Metadata
    logger.info("Loading Metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if Config.DEBUG:
        logger.info("DEBUG mode: Subsampling data...")
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:20]

    # Process Data
    logger.info("Processing Train Data...")
    train_data = preprocess_and_cache(train_df, tokenizer, "train", load_cached_data)

    logger.info("Processing Val Data...")
    val_data = preprocess_and_cache(val_df, tokenizer, "val", load_cached_data)

    logger.info("Processing Test Data...")
    test_data = preprocess_and_cache(test_df, tokenizer, "test", load_cached_data)

    # Create Datasets
    train_dataset = EssayDataset(train_data, len(train_df))
    val_dataset = EssayDataset(val_data, len(val_df))
    test_dataset = EssayDataset(test_data, len(test_df))

    # Create DataLoaders
    collate_fn = CollateFn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    return train_loader, val_loader, test_loader
