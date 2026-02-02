import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =========================================================================
# Tokenization Maps
# =========================================================================
NUC_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Helper Functions
# =========================================================================


def get_sinusoidal_encoding(positions, d_model):
    """
    Computes sinusoidal encoding for signed scalar positions.

    Args:
        positions (np.ndarray): Array of signed distances/positions. Shape (L,).
        d_model (int): The embedding dimension.

    Returns:
        np.ndarray: Sinusoidal encodings. Shape (L, d_model).
    """
    # Create the division term for the geometric progression
    # div_term shape: (d_model // 2,)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Initialize output array
    pe = np.zeros((len(positions), d_model), dtype=np.float32)

    # Calculate phase: positions * div_term
    # Broadcasting: (L, 1) * (d_model/2,) -> (L, d_model/2)
    phase = positions[:, None] * div_term

    # Apply Sine to even indices and Cosine to odd indices
    pe[:, 0::2] = np.sin(phase)
    pe[:, 1::2] = np.cos(phase)

    return pe


def parse_structure_dist(structure):
    """
    Parses a dot-bracket structure string to calculate signed pairing distances.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of signed distances. Unpaired bases are 0.
                    If i pairs with j, dist[i] = j - i.
    """
    n = len(structure)
    stack = []
    dists = np.zeros(n, dtype=np.float32)

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Calculate signed distance
                # For the opening bracket at j, pair is i (later). Dist = i - j (positive)
                # For the closing bracket at i, pair is j (earlier). Dist = j - i (negative)
                dists[j] = float(i - j)
                dists[i] = float(j - i)

    return dists


def process_dataframe(df, mode):
    """
    Extracts and processes features from the dataframe.
    """
    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values
    ids = df["id"].values

    N = len(df)
    L = Config.SEQ_LENGTH

    # Pre-allocate arrays
    X_seq = np.zeros((N, L), dtype=np.int32)
    X_loop = np.zeros((N, L), dtype=np.int32)
    X_dist = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        # Tokenize Sequence
        X_seq[i] = [NUC_MAP.get(c, 0) for c in sequences[i]]

        # Tokenize Loop Type
        X_loop[i] = [LOOP_MAP.get(c, 0) for c in loop_types[i]]

        # Parse Structure Distance
        X_dist[i] = parse_structure_dist(structures[i])

    Y = None
    if mode in ["train", "val"]:
        # Extract targets
        # Targets are stored as lists in the DataFrame columns.
        # We assume metadata ensures consistency.
        target_arrays = []
        for col in Config.TARGET_COLS:
            # Convert column of lists to 2D numpy array
            col_data = np.array(df[col].tolist(), dtype=np.float32)
            target_arrays.append(col_data)

        # Stack to shape (N, 68, 3)
        Y = np.stack(target_arrays, axis=2)

    return X_seq, X_loop, X_dist, Y, ids


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, seqs, loops, dists, targets=None, ids=None):
        """
        Args:
            seqs (np.ndarray): (N, L) Integer sequence tokens.
            loops (np.ndarray): (N, L) Integer loop type tokens.
            dists (np.ndarray): (N, L) Float signed structure distances.
            targets (np.ndarray, optional): (N, Scored_Len, Num_Targets).
            ids (np.ndarray, optional): Array of sample IDs.
        """
        self.seqs = seqs
        self.loops = loops
        self.dists = dists
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Inputs
        seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)

        # Compute Sinusoidal Encoding on the fly
        # dists[idx] is shape (L,) -> pair_enc is (L, Embed_Dim)
        dist_val = self.dists[idx]
        pair_enc_np = get_sinusoidal_encoding(dist_val, Config.EMBED_DIM_PAIR)
        pair_enc = torch.tensor(pair_enc_np, dtype=torch.float32)

        item = {"seq": seq, "loop": loop, "pair_enc": pair_enc}

        if self.ids is not None:
            item["id"] = str(self.ids[idx])

        if self.targets is not None:
            # Raw targets shape: (68, 3)
            tgt_raw = self.targets[idx]

            # Pad targets to full sequence length (107)
            # We use 0.0 for padding (masked out later)
            scored_len = tgt_raw.shape[0]
            total_len = Config.SEQ_LENGTH
            num_targets = tgt_raw.shape[1]

            tgt_padded = np.zeros((total_len, num_targets), dtype=np.float32)
            tgt_padded[:scored_len, :] = tgt_raw

            # Create a mask: 1 for scored positions, 0 for padding
            mask = np.zeros((total_len,), dtype=np.float32)
            mask[:scored_len] = 1.0

            item["target"] = torch.tensor(tgt_padded, dtype=torch.float32)
            item["mask"] = torch.tensor(mask, dtype=torch.float32)

        return item


# =========================================================================
# Main Data Loading Function
# =========================================================================


def get_dataloaders(load_cached_data=True):
    """
    Loads data, processes it (with caching), and returns DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed()
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    loaders = {}
    modes = ["train", "val", "test"]

    for mode in modes:
        cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.npz")
        data_loaded = False

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached {mode} data from {cache_path}...")
                cached = np.load(cache_path, allow_pickle=True)
                X_seq = cached["X_seq"]
                X_loop = cached["X_loop"]
                X_dist = cached["X_dist"]
                ids = cached["ids"]

                # Load Y if it exists in the archive
                if "Y" in cached:
                    Y = cached["Y"]
                else:
                    Y = None

                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}")
                data_loaded = False

        # 2. Process from Scratch if needed
        if not data_loaded:
            print(f"Processing {mode} data from scratch...")

            # Select correct input path
            if mode == "train":
                path = Config.TRAIN_DATA_PATH
            elif mode == "val":
                path = Config.VAL_DATA_PATH
            else:
                path = Config.TEST_DATA_PATH

            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")

            df = pd.read_parquet(path)

            # Process DataFrame
            X_seq, X_loop, X_dist, Y, ids = process_dataframe(df, mode)

            # Save to Cache
            print(f"Saving {mode} data to {cache_path}...")
            save_dict = {"X_seq": X_seq, "X_loop": X_loop, "X_dist": X_dist, "ids": ids}
            if Y is not None:
                save_dict["Y"] = Y

            np.savez_compressed(cache_path, **save_dict)

        # 3. Create Dataset and DataLoader
        dataset = RNADataset(X_seq, X_loop, X_dist, Y, ids)

        is_train = mode == "train"

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=is_train,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=is_train,
        )

        loaders[mode] = loader

    return loaders["train"], loaders["val"], loaders["test"]
