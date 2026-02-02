import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from library.config import Config


def tokenize(sequence, vocab_map):
    """
    Converts a string sequence into a list of integer indices based on a vocabulary map.
    """
    return [vocab_map.get(char, 0) for char in sequence]


def get_structure_edges(structure_str):
    """
    Parses a dot-bracket structure string to identify base pairs.
    Returns a list of [source, target] lists for base pair edges (undirected/bidirectional).
    """
    stack = []
    edges = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Add bi-directional connection for the base pair
                edges.append([i, j])
                edges.append([j, i])
    return edges


def get_backbone_edges(seq_len):
    """
    Generates backbone edges for a sequence of given length.
    Returns:
        fwd_edges: List of [i, i+1]
        bwd_edges: List of [i+1, i]
    """
    fwd_edges = []
    bwd_edges = []
    for i in range(seq_len - 1):
        fwd_edges.append([i, i + 1])
        bwd_edges.append([i + 1, i])
    return fwd_edges, bwd_edges


def process_row(row, is_test=False):
    """
    Converts a single row from the dataframe into a PyTorch Geometric Data object.
    """
    seq_len = Config.SEQ_LEN  # Should be 107

    # 1. Node Features
    # Sequence embedding indices
    seq_idxs = tokenize(row["sequence"], Config.VOCAB_MAP_SEQ)
    # Structure embedding indices
    struct_idxs = tokenize(row["structure"], Config.VOCAB_MAP_STRUCT)
    # Loop type embedding indices
    loop_idxs = tokenize(row["predicted_loop_type"], Config.VOCAB_MAP_LOOP)

    # Stack features: (SeqLen, 3)
    # x columns: [sequence, structure, loop_type]
    x = torch.tensor([seq_idxs, struct_idxs, loop_idxs], dtype=torch.long).t()

    # 2. Edges
    # Backbone
    backbone_fwd, backbone_bwd = get_backbone_edges(seq_len)
    # Base pairs
    base_pairs = get_structure_edges(row["structure"])

    # Combine edges
    # We assign types: 0 for fwd, 1 for bwd, 2 for base_pair
    edge_indices = []
    edge_types = []

    if backbone_fwd:
        edge_indices.extend(backbone_fwd)
        edge_types.extend([0] * len(backbone_fwd))

    if backbone_bwd:
        edge_indices.extend(backbone_bwd)
        edge_types.extend([1] * len(backbone_bwd))

    if base_pairs:
        edge_indices.extend(base_pairs)
        edge_types.extend([2] * len(base_pairs))

    if not edge_indices:
        # Fallback for single node or weird case, though seq_len is 107
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long)

    # 3. Targets and Mask
    # Initialize targets with zeros
    y = torch.zeros((seq_len, Config.NUM_TARGETS), dtype=torch.float32)

    # Create mask: True for scored positions, False otherwise
    mask = torch.zeros(seq_len, dtype=torch.bool)
    seq_scored = row["seq_scored"]
    mask[:seq_scored] = True

    if not is_test:
        # Extract targets from the row
        # Targets are lists in the parquet dataframe
        for i, target_col in enumerate(Config.TARGET_COLS):
            vals = row[target_col]
            # Ensure we don't go out of bounds if data is malformed, though metadata checks passed
            length = min(len(vals), seq_len)
            y[:length, i] = torch.tensor(vals[:length], dtype=torch.float32)

    # 4. Create Data Object
    # We also store the ID for submission mapping
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)
    data.id = row["id"]

    return data


def load_data(split="train", load_cached_data=True):
    """
    Loads dataset for the given split.
    Checks cache first. If not found or forced reload, processes from Parquet.
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.TRAIN_CACHE_PATH
        is_test = False
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        cache_path = Config.VAL_CACHE_PATH
        is_test = False
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        cache_path = Config.TEST_CACHE_PATH
        is_test = True
    else:
        raise ValueError(f"Unknown split: {split}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            data_list = torch.load(cache_path)
            return data_list
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Metadata
    print(f"Processing {split} data from {meta_path}...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)

    # Debug mode: subset data
    if Config.DEBUG:
        df = df.head(50)
        print(f"DEBUG: Subsampled {split} to {len(df)} rows.")

    data_list = []
    for _, row in df.iterrows():
        data_obj = process_row(row, is_test=is_test)
        data_list.append(data_obj)

    # 3. Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    torch.save(data_list, cache_path)

    return data_list
