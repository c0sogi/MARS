import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class MarkdownRankDataset(Dataset):
    """
    PyTorch Dataset for the Dense Semantic Stream (Transformer).
    Tokenizes markdown text and serves (input_ids, attention_mask, label).
    """

    def __init__(self, df, tokenizer, max_len):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'source' and 'rank'.
            tokenizer (PreTrainedTokenizer): Transformer tokenizer.
            max_len (int): Maximum sequence length.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sources = self.df["source"].astype(str).tolist()
        self.ranks = self.df["rank"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.sources[idx]
        rank = self.ranks[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(rank, dtype=torch.float),
        }


def preprocess_data(df_meta, data_type="train", load_cached_data=True, debug=False):
    """
    Parses notebook JSONs to extract markdown cells and calculate normalized ranks.
    Implements caching via Parquet.

    Args:
        df_meta (pd.DataFrame): Metadata DataFrame containing 'id', 'filepath', and optionally 'cell_order'.
        data_type (str): 'train', 'val', or 'test'. Used for cache naming.
        load_cached_data (bool): Whether to load from cache if available.
        debug (bool): If True, processes a subset of data.

    Returns:
        pd.DataFrame: Processed dataframe with columns ['id', 'cell_id', 'source', 'rank', 'ancestor_id'].
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{data_type}_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {data_type} data from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if debug:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)
        return df

    print(f"Processing {data_type} data from scratch...")

    if debug:
        df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

    data_rows = []

    # Pre-check columns to determine mode
    has_ground_truth = "cell_order" in df_meta.columns

    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        rel_path = row["filepath"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Ancestor ID is useful for GroupKFold if we were doing CV here,
        # but we just pass it through. Default to nb_id if missing.
        ancestor_id = row.get("ancestor_id", nb_id)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                nb_json = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        if has_ground_truth:
            # Training/Val mode: Use ground truth order to calculate rank
            cell_order = row["cell_order"].split()
            total_cells = len(cell_order)

            for rank_idx, cell_id in enumerate(cell_order):
                ctype = cell_types.get(cell_id, "unknown")
                if ctype == "markdown":
                    # Normalized rank: 0.0 (top) to 1.0 (bottom)
                    # We use rank_idx / total_cells as per the idea description
                    norm_rank = rank_idx / total_cells

                    source_text = sources.get(cell_id, "")
                    data_rows.append(
                        {
                            "id": nb_id,
                            "cell_id": cell_id,
                            "source": source_text,
                            "rank": norm_rank,
                            "ancestor_id": ancestor_id,
                        }
                    )
        else:
            # Test mode: Extract all markdown cells. Rank is unknown (-1).
            # We iterate over the keys in the JSON.
            # Note: In the provided dataset, markdown cells are shuffled and placed after code cells.
            for cell_id, ctype in cell_types.items():
                if ctype == "markdown":
                    source_text = sources.get(cell_id, "")
                    data_rows.append(
                        {
                            "id": nb_id,
                            "cell_id": cell_id,
                            "source": source_text,
                            "rank": -1.0,
                            "ancestor_id": ancestor_id,
                        }
                    )

    df_processed = pd.DataFrame(data_rows)

    # Ensure source is string and handle NaNs
    df_processed["source"] = df_processed["source"].fillna("").astype(str)

    # Save to cache
    print(f"Saving processed {data_type} data to {cache_path}...")
    df_processed.to_parquet(cache_path, index=False)

    return df_processed


def get_test_anchors(df_test):
    """
    Extracts code cells from test notebooks to serve as fixed anchors.
    Assumes code cells in the JSON are in the correct relative order.

    Args:
        df_test (pd.DataFrame): Test metadata.

    Returns:
        dict: Mapping from notebook_id to list of code cell IDs.
    """
    anchors = {}

    for _, row in df_test.iterrows():
        nb_id = row["id"]
        rel_path = row["filepath"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                nb_json = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            anchors[nb_id] = []
            continue

        cell_types = nb_json.get("cell_type", {})

        # Extract code cells. We rely on the insertion order of the dictionary (Python 3.7+).
        # The task description implies code cells are in original order in the file.
        code_cells = [cid for cid, ctype in cell_types.items() if ctype == "code"]
        anchors[nb_id] = code_cells

    return anchors


def load_data_factory(load_cached_data=True):
    """
    High-level function to load all necessary data for training and inference.

    Returns:
        tuple: (df_train_processed, df_val_processed, df_test_processed, test_anchors)
    """
    # Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Process Data (Markdown extraction & Rank calculation)
    df_train = preprocess_data(
        df_train_meta,
        data_type="train",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )
    df_val = preprocess_data(
        df_val_meta,
        data_type="val",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )
    df_test = preprocess_data(
        df_test_meta,
        data_type="test",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # Get Anchors for Test set
    test_anchors = get_test_anchors(df_test_meta)

    return df_train, df_val, df_test, test_anchors
