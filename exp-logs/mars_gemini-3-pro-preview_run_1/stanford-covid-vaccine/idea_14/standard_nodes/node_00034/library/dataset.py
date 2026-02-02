import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Tokenization Maps
# =========================================================================
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
# Loop types: S: Stem, M: Multiloop, I: Internal, B: Bulge, H: Hairpin, E: End, X: External
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Serves preprocessed tensors for sequence, loop type, and geometric distance.
    """

    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing processed tensors.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.sequence = data_dict["sequence"]
        self.loop_type = data_dict["loop_type"]
        self.pair_dist = data_dict["pair_dist"]
        self.ids = data_dict["ids"]

        if mode != "test":
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        """
        Returns a single sample dictionary.
        """
        item = {
            "sequence": self.sequence[idx],
            "loop_type": self.loop_type[idx],
            "pair_dist": self.pair_dist[idx],
            "id": self.ids[idx],
        }

        if self.mode != "test":
            item["targets"] = self.targets[idx]

        return item


def parse_structure_to_distance(structure):
    """
    Parses a dot-bracket structure string into a signed pairing distance vector.

    Strategy: Explicit Geometric Encoding.
    For a pair (i, j), we encode the signed distance:
      - At position i (opening): value is j - i (positive)
      - At position j (closing): value is i - j (negative)
    Unpaired bases are assigned a distance of 0.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (L,) containing signed distances.
    """
    L = len(structure)
    distances = np.zeros(L, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = i
                prev_i = stack.pop()
                # Calculate signed distance
                # From opening bracket perspective: target is forward (positive dist)
                distances[prev_i] = float(j - prev_i)
                # From closing bracket perspective: target is backward (negative dist)
                distances[j] = float(prev_i - j)
            else:
                # Unbalanced closing bracket (should not happen in valid bpRNA, but handle safely)
                pass

    return distances


def process_dataframe(df, mode="train"):
    """
    Converts a pandas DataFrame into a dictionary of tensors.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing 'sequence', 'loop_type', 'pair_dist', 'ids',
              and optionally 'targets'.
    """
    # Reset index to ensure 0..N-1 iteration matches tensor indexing
    df = df.reset_index(drop=True)

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate tensors for efficiency
    sequences = torch.zeros((num_samples, seq_len), dtype=torch.long)
    loop_types = torch.zeros((num_samples, seq_len), dtype=torch.long)
    pair_dists = torch.zeros((num_samples, seq_len), dtype=torch.float32)

    if mode != "test":
        # Targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Shape: (N, 107, 3). We will fill the first 68 positions.
        targets = torch.zeros((num_samples, seq_len, 3), dtype=torch.float32)

    ids = df["id"].values.tolist()

    for i, row in df.iterrows():
        # 1. Sequence Tokenization
        seq_str = row["sequence"]
        # Map chars to indices, default to 0 (A) if unexpected (unlikely)
        seq_encoded = [NUCLEOTIDE_MAP.get(c, 0) for c in seq_str]
        sequences[i] = torch.tensor(seq_encoded, dtype=torch.long)

        # 2. Loop Type Tokenization
        loop_str = row["predicted_loop_type"]
        # Map chars to indices, default to 5 (E) if unexpected
        loop_encoded = [LOOP_TYPE_MAP.get(c, 5) for c in loop_str]
        loop_types[i] = torch.tensor(loop_encoded, dtype=torch.long)

        # 3. Geometric Distance Encoding
        struct_str = row["structure"]
        dists = parse_structure_to_distance(struct_str)
        pair_dists[i] = torch.tensor(dists, dtype=torch.float32)

        # 4. Targets (Train/Val only)
        if mode != "test":
            # Helper to extract array safely
            def get_target_array(col_name):
                arr = row[col_name]
                if isinstance(arr, (list, np.ndarray)):
                    return np.array(arr, dtype=np.float32)
                # Fallback for missing data (should not happen in clean metadata)
                return np.zeros(Config.PRED_LEN, dtype=np.float32)

            reactivity = get_target_array("reactivity")
            deg_Mg_pH10 = get_target_array("deg_Mg_pH10")
            deg_Mg_50C = get_target_array("deg_Mg_50C")

            # Stack selected targets
            # Note: Input lists are length 68 (Config.PRED_LEN)
            # We place them into the (107, 3) tensor.
            current_len = len(reactivity)

            # Shape (L_valid, 3)
            target_matrix = np.stack([reactivity, deg_Mg_pH10, deg_Mg_50C], axis=1)

            # Copy into main tensor
            # The loss function will handle slicing to the valid length
            targets[i, :current_len, :] = torch.tensor(
                target_matrix, dtype=torch.float32
            )

    data_dict = {
        "sequence": sequences,
        "loop_type": loop_types,
        "pair_dist": pair_dists,
        "ids": ids,
    }

    if mode != "test":
        data_dict["targets"] = targets

    return data_dict


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders. Handles caching logic.

    Args:
        load_cached_data (bool): If True, attempts to load .pt files from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Train Data
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(Config.TRAIN_CACHE):
        print(f"Loading train data from cache: {Config.TRAIN_CACHE}")
        train_data = torch.load(Config.TRAIN_CACHE)
    else:
        print(f"Processing train data from: {Config.TRAIN_FILE}")
        df_train = pd.read_parquet(Config.TRAIN_FILE)
        train_data = process_dataframe(df_train, mode="train")
        print(f"Saving train cache to: {Config.TRAIN_CACHE}")
        torch.save(train_data, Config.TRAIN_CACHE)

    train_dataset = RNADataset(train_data, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 2. Validation Data
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(Config.VAL_CACHE):
        print(f"Loading val data from cache: {Config.VAL_CACHE}")
        val_data = torch.load(Config.VAL_CACHE)
    else:
        print(f"Processing val data from: {Config.VAL_FILE}")
        df_val = pd.read_parquet(Config.VAL_FILE)
        val_data = process_dataframe(df_val, mode="val")
        print(f"Saving val cache to: {Config.VAL_CACHE}")
        torch.save(val_data, Config.VAL_CACHE)

    val_dataset = RNADataset(val_data, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Test Data
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(Config.TEST_CACHE):
        print(f"Loading test data from cache: {Config.TEST_CACHE}")
        test_data = torch.load(Config.TEST_CACHE)
    else:
        print(f"Processing test data from: {Config.TEST_FILE}")
        df_test = pd.read_parquet(Config.TEST_FILE)
        test_data = process_dataframe(df_test, mode="test")
        print(f"Saving test cache to: {Config.TEST_CACHE}")
        torch.save(test_data, Config.TEST_CACHE)

    test_dataset = RNADataset(test_data, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
