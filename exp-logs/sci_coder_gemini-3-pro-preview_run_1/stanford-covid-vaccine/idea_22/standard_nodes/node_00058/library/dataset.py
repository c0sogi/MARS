import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, sequences, loops, distances, targets=None, ids=None):
        self.sequences = sequences
        self.loops = loops
        self.distances = distances
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)
        dist = torch.tensor(self.distances[idx], dtype=torch.float32)

        item = {
            "seq": seq,
            "loop": loop,
            "dist": dist,
        }

        if self.targets is not None:
            # Targets are float32
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["target"] = target

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def parse_structure_distances(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to calculate signed pairing distances.
    For a pair (i, j) where i < j:
      - Distance at i is j - i (positive)
      - Distance at j is i - j (negative)
    Unpaired bases have distance 0.
    """
    dists = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Pair is (j, i) where j < i
                # Distance for j (opening) is i - j
                dists[j] = float(i - j)
                # Distance for i (closing) is j - i
                dists[i] = float(j - i)

    return dists


def process_dataframe(df, mode):
    """
    Process a dataframe into numeric arrays for the model.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    sequences = np.zeros((num_samples, seq_len), dtype=np.int64)
    loops = np.zeros((num_samples, seq_len), dtype=np.int64)
    distances = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Iterate and process
    for idx, row in df.iterrows():
        # Sequence Tokenization
        seq_str = row["sequence"]
        sequences[idx] = [Config.NUCLEOTIDE_MAP.get(c, 0) for c in seq_str]

        # Loop Tokenization
        loop_str = row["predicted_loop_type"]
        loops[idx] = [Config.LOOP_TYPE_MAP.get(c, 0) for c in loop_str]

        # Structure Distance
        struct_str = row["structure"]
        distances[idx] = parse_structure_distances(struct_str, seq_len)

    # Process Targets (Train/Val only)
    targets = None
    if mode in ["train", "val"]:
        # Initialize with zeros (padding for positions > 68)
        targets = np.zeros(
            (num_samples, seq_len, len(Config.TARGET_COLS)), dtype=np.float32
        )

        for t_i, col in enumerate(Config.TARGET_COLS):
            # Extract column data (series of lists)
            col_data = df[col].values
            # Convert list of lists to numpy array
            # Note: The metadata ensures these are lists of length 68
            col_data_list = col_data.tolist()
            col_array = np.array(col_data_list, dtype=np.float32)

            # Fill the first 68 positions
            pred_len = col_array.shape[1]
            targets[:, :pred_len, t_i] = col_array

    # Extract IDs
    ids = df["id"].values.tolist()

    return sequences, loops, distances, targets, ids


def load_or_process_data(mode, load_cached_data=True):
    """
    Loads data from cache or processes it from raw parquet files.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.npz")

    # Determine raw data path
    if mode == "train":
        raw_path = Config.TRAIN_DATA_PATH
    elif mode == "val":
        raw_path = Config.VAL_DATA_PATH
    elif mode == "test":
        raw_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {mode} data from cache: {cache_path}")
            data = np.load(cache_path)
            sequences = data["sequences"]
            loops = data["loops"]
            distances = data["distances"]

            if "targets" in data:
                targets = data["targets"]
            else:
                targets = None

            # Load IDs from parquet (fast, avoids pickle issues in npz)
            if os.path.exists(raw_path):
                df_ids = pd.read_parquet(raw_path, columns=["id"])
                ids = df_ids["id"].values.tolist()
            else:
                # Fallback if parquet missing but cache exists (unlikely in this setup)
                ids = [f"id_{i}" for i in range(len(sequences))]

            return sequences, loops, distances, targets, ids
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from {raw_path}...")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_parquet(raw_path)
    sequences, loops, distances, targets, ids = process_dataframe(df, mode)

    # 3. Save to cache
    save_dict = {"sequences": sequences, "loops": loops, "distances": distances}
    if targets is not None:
        save_dict["targets"] = targets

    print(f"Saving {mode} data to cache: {cache_path}")
    np.savez_compressed(cache_path, **save_dict)

    return sequences, loops, distances, targets, ids


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Returns training and validation DataLoaders.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Train Data
    t_seq, t_loop, t_dist, t_tgt, t_ids = load_or_process_data(
        "train", load_cached_data
    )
    train_dataset = RNADataset(t_seq, t_loop, t_dist, t_tgt, t_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Validation Data
    v_seq, v_loop, v_dist, v_tgt, v_ids = load_or_process_data("val", load_cached_data)
    val_dataset = RNADataset(v_seq, v_loop, v_dist, v_tgt, v_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Returns test DataLoader.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    te_seq, te_loop, te_dist, te_tgt, te_ids = load_or_process_data(
        "test", load_cached_data
    )
    test_dataset = RNADataset(te_seq, te_loop, te_dist, te_tgt, te_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
