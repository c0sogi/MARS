import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_pair_map


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Attributes:
        X (torch.Tensor): Input features of shape (N, SEQ_LEN, INPUT_DIM).
        y (torch.Tensor): Target values of shape (N, PRED_LEN, NUM_CLASSES).
                          For the test set, this is a dummy tensor of zeros.
    """

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve input features
        features = self.X[idx]

        # Retrieve targets or return dummy for test set
        if self.y is not None:
            targets = self.y[idx]
        else:
            # Return dummy targets of the expected shape (68, 5)
            # We assume PRED_LEN=68 and NUM_CLASSES=5 based on Config
            targets = torch.zeros((68, 5), dtype=torch.float32)

        return features, targets


def one_hot_encode(seq, token_dict, length):
    """
    One-hot encodes a sequence string into a numpy array.

    Args:
        seq (str): Input sequence (nucleotides, structure, or loop type).
        token_dict (dict): Mapping from character to integer index.
        length (int): Fixed length of the sequence.

    Returns:
        np.ndarray: One-hot encoded array of shape (length, len(token_dict)).
    """
    arr = np.zeros((length, len(token_dict)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in token_dict:
            arr[i, token_dict[char]] = 1.0
    return arr


def process_data(df, config, has_targets=True):
    """
    Processes a dataframe into spatially augmented feature tensors and target tensors.

    Implements the 'Spatial Feature Augmentation' strategy:
    1. One-hot encodes Sequence, Structure, and Loop Type.
    2. Concatenates these to form base features.
    3. Uses the secondary structure to find paired bases.
    4. Concatenates the features of the paired base to the current base.

    Args:
        df (pd.DataFrame): Input dataframe containing sequences and metadata.
        config (Config): Configuration object with dimensions and mappings.
        has_targets (bool): Whether to extract target columns (True for Train/Val).

    Returns:
        tuple: (X_tensor, y_tensor)
               X_tensor: (N, SEQ_LEN, INPUT_DIM)
               y_tensor: (N, PRED_LEN, NUM_CLASSES) or None
    """
    X_list = []
    y_list = []

    # Mappings from Config
    seq_map = config.TOKEN_DICT_SEQ
    struct_map = config.TOKEN_DICT_STRUCT
    loop_map = config.TOKEN_DICT_LOOP

    for _, row in df.iterrows():
        # --- Feature Construction ---

        # 1. Base Features (One-Hot Encoding)
        seq_feat = one_hot_encode(row["sequence"], seq_map, config.SEQ_LEN)
        struct_feat = one_hot_encode(row["structure"], struct_map, config.SEQ_LEN)
        loop_feat = one_hot_encode(row["predicted_loop_type"], loop_map, config.SEQ_LEN)

        # Concatenate base features: Shape (SEQ_LEN, BASE_FEATURE_DIM)
        # BASE_FEATURE_DIM = 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
        base_feat = np.concatenate([seq_feat, struct_feat, loop_feat], axis=1)

        # 2. Spatial Augmentation
        # Get mapping of paired positions (index i -> index j)
        pair_map = get_pair_map(row["structure"])

        # Initialize augmented features: Shape (SEQ_LEN, INPUT_DIM)
        # INPUT_DIM = 28
        aug_feat = np.zeros((config.SEQ_LEN, config.INPUT_DIM), dtype=np.float32)

        # Fill first half with base features
        aug_feat[:, : config.BASE_FEATURE_DIM] = base_feat

        # Fill second half with features of the paired base
        for i in range(config.SEQ_LEN):
            partner = pair_map[i]
            if partner != -1:
                # If paired, copy the partner's features
                aug_feat[i, config.BASE_FEATURE_DIM :] = base_feat[partner]
            else:
                # If unpaired, leave as zeros (explicitly handled by initialization)
                pass

        X_list.append(aug_feat)

        # --- Target Extraction ---
        if has_targets:
            # Extract the 5 target columns
            # Each column in the dataframe is a list of floats of length PRED_LEN (68)
            t_list = []
            for col in config.TARGET_COLS:
                val = row[col]
                # Ensure we have a numpy array of floats
                if isinstance(val, (list, np.ndarray)):
                    t_list.append(np.array(val, dtype=np.float32))
                else:
                    # Fallback for missing data (should not happen in clean data)
                    t_list.append(np.zeros(config.PRED_LEN, dtype=np.float32))

            # Stack to shape (5, 68) then transpose to (68, 5)
            y_sample = np.stack(t_list, axis=1)
            y_list.append(y_sample)

    # Convert lists to PyTorch Tensors
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)

    if has_targets:
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
        return X_tensor, y_tensor
    else:
        return X_tensor, None


def get_dataloaders(config, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles caching of processed tensors to disk to optimize runtime.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): If True, attempts to load pre-processed tensors from disk.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # --- Train Data ---
    if load_cached_data and os.path.exists(config.TRAIN_CACHE):
        print(f"Loading cached train data from {config.TRAIN_CACHE}")
        train_data = torch.load(config.TRAIN_CACHE)
        X_train, y_train = train_data["X"], train_data["y"]
    else:
        print("Processing train data from metadata...")
        df_train = pd.read_parquet(config.TRAIN_PATH)
        X_train, y_train = process_data(df_train, config, has_targets=True)
        # Save to cache
        torch.save({"X": X_train, "y": y_train}, config.TRAIN_CACHE)

    # --- Validation Data ---
    if load_cached_data and os.path.exists(config.VAL_CACHE):
        print(f"Loading cached val data from {config.VAL_CACHE}")
        val_data = torch.load(config.VAL_CACHE)
        X_val, y_val = val_data["X"], val_data["y"]
    else:
        print("Processing validation data from metadata...")
        df_val = pd.read_parquet(config.VAL_PATH)
        X_val, y_val = process_data(df_val, config, has_targets=True)
        # Save to cache
        torch.save({"X": X_val, "y": y_val}, config.VAL_CACHE)

    # --- Test Data ---
    if load_cached_data and os.path.exists(config.TEST_CACHE):
        print(f"Loading cached test data from {config.TEST_CACHE}")
        test_data = torch.load(config.TEST_CACHE)
        X_test = test_data["X"]
        y_test = None
    else:
        print("Processing test data from metadata...")
        df_test = pd.read_parquet(config.TEST_PATH)
        X_test, _ = process_data(df_test, config, has_targets=False)
        # Save to cache
        torch.save({"X": X_test, "y": None}, config.TEST_CACHE)
        y_test = None

    # --- Create Datasets ---
    train_dataset = RNADataset(X_train, y_train)
    val_dataset = RNADataset(X_val, y_val)
    test_dataset = RNADataset(X_test, y_test)

    # --- Create DataLoaders ---
    # Use pinned memory if CUDA is available for faster transfer
    use_pin_memory = config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
