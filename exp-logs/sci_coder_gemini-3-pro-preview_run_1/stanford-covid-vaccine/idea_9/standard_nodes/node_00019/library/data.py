import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Handles dynamic masking for the auxiliary reconstruction task.
    """

    def __init__(self, data, mode="train"):
        """
        Args:
            data (dict): Dictionary containing processed numpy arrays:
                         'sequence', 'loop_type', 'distance_map', 'targets' (optional).
            mode (str): 'train', 'val', or 'test'. Controls masking behavior.
        """
        self.sequences = data["sequence"]
        self.loop_types = data["loop_type"]
        self.distance_maps = data["distance_map"]
        self.ids = data["ids"]

        # Targets might not exist for test set
        self.targets = data.get("targets", None)

        self.mode = mode
        self.seq_len = Config.SEQ_LEN

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 1. Get base data
        seq_indices = self.sequences[idx].copy()  # Shape: (107,)
        loop_indices = self.loop_types[idx]  # Shape: (107,)
        dist_map = self.distance_maps[idx]  # Shape: (107,)
        sample_id = self.ids[idx]

        # 2. Dynamic Masking (Only for training)
        # The labels for reconstruction are the original sequence indices
        reconstruction_labels = torch.tensor(seq_indices, dtype=torch.long)

        if self.mode == "train":
            # Create a mask for 15% of tokens
            # We don't mask special tokens if we had them, but here vocab is just A,G,C,U
            prob_mask = torch.rand(self.seq_len)
            mask_bool = prob_mask < Config.MASK_PROB

            # Apply mask: Replace selected tokens with MASK_TOKEN_ID
            seq_input = torch.tensor(seq_indices, dtype=torch.long)
            seq_input[mask_bool] = Config.MASK_TOKEN_ID
        else:
            # For validation/test, we typically don't mask input for the primary task
            # However, if we want to evaluate reconstruction loss in val, we could mask.
            # Config implies inference is without masking.
            # We will pass clean sequence for inference.
            seq_input = torch.tensor(seq_indices, dtype=torch.long)

        # 3. Convert other features to tensors
        loop_input = torch.tensor(loop_indices, dtype=torch.long)
        dist_input = torch.tensor(dist_map, dtype=torch.float32)

        # 4. Prepare Targets
        item = {
            "id": sample_id,
            "seq_input": seq_input,
            "loop_input": loop_input,
            "dist_input": dist_input,
            "reconstruction_labels": reconstruction_labels,
        }

        if self.targets is not None:
            # Targets are (68, 5)
            target_vals = self.targets[idx]
            item["targets"] = torch.tensor(target_vals, dtype=torch.float32)

        return item


def _parse_structure_to_distance(structure_str, seq_len):
    """
    Parses a dot-bracket structure string into a signed distance map.
    If pair is (i, j), distance at i is j-i, distance at j is i-j.
    Unpaired bases have distance 0.
    """
    dist_array = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                # Signed distance
                dist = i - start_idx
                dist_array[start_idx] = dist  # Positive for opening
                dist_array[i] = -dist  # Negative for closing
        elif char == ".":
            dist_array[i] = 0.0

    return dist_array


def _tokenize_sequence(seq_str):
    return [Config.VOCAB_MAP[char] for char in seq_str]


def _tokenize_loop(loop_str):
    return [Config.LOOP_MAP[char] for char in loop_str]


def process_data(df, has_targets=True):
    """
    Converts DataFrame columns into numpy arrays for the dataset.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    sequences = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_types = np.zeros((num_samples, seq_len), dtype=np.int32)
    distance_maps = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Process inputs
    for i, row in df.iterrows():
        # Sequence
        sequences[i] = np.array(_tokenize_sequence(row["sequence"]))

        # Loop Type
        loop_types[i] = np.array(_tokenize_loop(row["predicted_loop_type"]))

        # Structure Distance
        distance_maps[i] = _parse_structure_to_distance(row["structure"], seq_len)

    # Process targets if they exist
    targets = None
    if has_targets:
        # Targets are lists of length 68 in the dataframe
        # We need to stack them: (N, 68, 5)
        target_cols = Config.TARGET_COLS
        # Extract lists and stack
        # df[col] is a Series of lists. np.vstack might fail if not careful,
        # but parquet loading usually keeps them as arrays/lists.

        # Shape: (N, 5, 68) -> transpose to (N, 68, 5)
        # Let's do it column by column
        target_arrays = []
        for col in target_cols:
            # stack lists into (N, 68)
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data)

        # Stack along last axis: (N, 68, 5)
        targets = np.stack(target_arrays, axis=-1).astype(np.float32)

    return {
        "sequence": sequences,
        "loop_type": loop_types,
        "distance_map": distance_maps,
        "ids": ids,
        "targets": targets,
    }


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Main function to load data, process/cache it, and return DataLoaders.
    """
    Config.create_dirs()

    # Define cache paths
    cache_train = os.path.join(Config.CACHE_DIR, "train_data.npz")
    cache_val = os.path.join(Config.CACHE_DIR, "val_data.npz")
    cache_test = os.path.join(Config.CACHE_DIR, "test_data.npz")

    train_data = None
    val_data = None
    test_data = None

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):
        print("Loading data from cache...")
        try:
            train_data = np.load(cache_train, allow_pickle=True)
            val_data = np.load(cache_val, allow_pickle=True)
            test_data = np.load(cache_test, allow_pickle=True)

            # Convert np.load result (NpzFile) back to dict for easier handling if needed,
            # but NpzFile acts like a dict.
            # Note: NpzFile stores arrays. 'targets' might be None (stored as None object or missing).
            # We need to handle the case where 'targets' key exists but is None or 0-d array.

            # Helper to convert NpzFile to dict
            def npz_to_dict(npz):
                d = {k: npz[k] for k in npz.files}
                if "targets" in d and (d["targets"].ndim == 0 or d["targets"] is None):
                    d["targets"] = None
                return d

            train_data = npz_to_dict(train_data)
            val_data = npz_to_dict(val_data)
            test_data = npz_to_dict(test_data)

        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            train_data = None

    # 2. Process from Scratch if needed
    if train_data is None:
        print("Processing data from Parquet files...")

        # Load Parquet
        df_train = pd.read_parquet(Config.TRAIN_FILE)
        df_val = pd.read_parquet(Config.VAL_FILE)
        df_test = pd.read_parquet(Config.TEST_FILE)

        # Debug subset
        if debug:
            print(f"DEBUG MODE: Using subset of {Config.DEBUG_SUBSET_SIZE} samples.")
            df_train = df_train.iloc[: Config.DEBUG_SUBSET_SIZE]
            df_val = df_val.iloc[: Config.DEBUG_SUBSET_SIZE]
            df_test = df_test.iloc[: Config.DEBUG_SUBSET_SIZE]

        # Process
        train_data = process_data(df_train, has_targets=True)
        val_data = process_data(df_val, has_targets=True)
        test_data = process_data(df_test, has_targets=False)

        # Save to cache
        # We use save (not savez_compressed) for speed, or savez_compressed for space.
        # Given 220GB RAM, space isn't huge issue, but let's compress lightly.
        np.savez(cache_train, **train_data)
        np.savez(cache_val, **val_data)
        np.savez(cache_test, **test_data)
        print("Data processed and cached.")

    # 3. Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
