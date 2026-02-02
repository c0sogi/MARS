import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Yields:
        inputs: (Channels, SeqLen) tensor.
        partner_indices: (SeqLen,) tensor of integer indices.
        targets: (SeqScored, NumTargets) tensor (if available).
        id: Sample ID string.
    """

    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs shape: (SeqLen, Channels)
        # Transpose to (Channels, SeqLen) for PyTorch Conv1d compatibility
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).transpose(0, 1)

        # partner_indices shape: (SeqLen,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        sample = {"inputs": x, "partner_indices": p_idx}

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        if self.targets is not None:
            # targets shape: (SeqScored, NumTargets)
            # Targets are typically not transposed for loss calculation
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        return sample


def get_structure_map(structure):
    """
    Parses dot-bracket structure string to find base pairs.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (SeqLen,) where arr[i] is the index of the
                    base paired with i, or -1 if unpaired.
    """
    seq_len = len(structure)
    partner_map = np.full(seq_len, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_map[i] = j
                partner_map[j] = i

    return partner_map


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    seq_len = len(seq)
    encoding = np.zeros((seq_len, vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def process_dataframe(df, is_test=False):
    """
    Processes a pandas DataFrame into numpy arrays for model input.

    Performs:
    1. One-hot encoding of Sequence, Structure, and Loop Type.
    2. Generation of Partner Index Map.
    3. Creation of 'Partner Base Identity' feature.
    4. Parsing of stringified target lists.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH
    input_channels = Config.INPUT_CHANNELS

    # Pre-allocate arrays
    all_inputs = np.zeros((num_samples, seq_len, input_channels), dtype=np.float32)
    all_partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    scored_len = Config.SEQ_SCORED
    num_targets = Config.NUM_TARGETS
    all_targets = None
    if not is_test:
        all_targets = np.zeros((num_samples, scored_len, num_targets), dtype=np.float32)

    ids = df["id"].values

    # Mappings from Config
    seq_map = Config.TOKEN_TO_INDEX_SEQ
    struct_map = Config.TOKEN_TO_INDEX_STRUCT
    loop_map = Config.TOKEN_TO_INDEX_LOOP
    target_cols = Config.TARGET_COLS

    # Iterate over samples
    # Note: Using iterrows is slower but safe; given dataset size (~2k), it's acceptable.
    for idx, row in enumerate(df.itertuples(index=False)):
        # Access attributes by column name
        sequence = row.sequence
        structure = row.structure
        loop_type = row.predicted_loop_type

        # 1. Basic One-Hot Encodings
        seq_oh = one_hot_encode(sequence, seq_map, 4)
        struct_oh = one_hot_encode(structure, struct_map, 3)
        loop_oh = one_hot_encode(loop_type, loop_map, 7)

        # 2. Partner Map
        partner_map = get_structure_map(structure)
        all_partner_indices[idx] = partner_map

        # 3. Partner Base Identity Feature
        # If position i is paired with j, feature vector at i includes one-hot of sequence[j].
        partner_base_oh = np.zeros((seq_len, 4), dtype=np.float32)

        # Convert sequence to indices for fast lookup
        seq_indices = np.array([seq_map.get(c, -1) for c in sequence])

        # Identify paired positions
        paired_mask = partner_map != -1

        if np.any(paired_mask):
            # Get indices of the partners
            partners = partner_map[paired_mask]
            # Get the base identity index of the partners
            partner_base_indices = seq_indices[partners]

            # Rows to update: positions that are paired
            rows_to_update = np.where(paired_mask)[0]

            # Filter valid base indices (in case of unknown chars, though unlikely)
            valid_bases = partner_base_indices != -1

            final_rows = rows_to_update[valid_bases]
            final_cols = partner_base_indices[valid_bases]

            # Set the one-hot bit
            partner_base_oh[final_rows, final_cols] = 1.0

        # 4. Concatenate Inputs
        # Order: Seq (4) + Struct (3) + Loop (7) + PartnerBase (4) = 18
        sample_input = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_base_oh], axis=1
        )
        all_inputs[idx] = sample_input

        # 5. Parse Targets
        if not is_test:
            for t_i, col in enumerate(target_cols):
                val_str = getattr(row, col)
                try:
                    # Targets are stored as stringified lists "[0.1, ...]"
                    val_list = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val_list = [0.0] * scored_len

                val_arr = np.array(val_list, dtype=np.float32)

                # Ensure length matches scored_len
                if len(val_arr) > scored_len:
                    val_arr = val_arr[:scored_len]
                elif len(val_arr) < scored_len:
                    pad = np.zeros(scored_len - len(val_arr), dtype=np.float32)
                    val_arr = np.concatenate([val_arr, pad])

                all_targets[idx, :, t_i] = val_arr

    return all_inputs, all_partner_indices, all_targets, ids


def load_data_split(
    csv_path, cache_path, load_cached_data=True, is_test=False, debug=False
):
    """
    Loads data from cache if available, otherwise processes CSV and saves cache.
    """
    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            ids = data["ids"]
            targets = None
            if not is_test:
                targets = data["targets"]

            # Apply debug subsetting if requested
            if debug:
                subset = min(Config.DEBUG_SUBSET_SIZE, len(inputs))
                inputs = inputs[:subset]
                partner_indices = partner_indices[:subset]
                ids = ids[:subset]
                if targets is not None:
                    targets = targets[:subset]

            return inputs, partner_indices, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # If debug, we can process a smaller DF to save time,
    # but we should be careful not to overwrite the full cache with a debug cache.
    # For simplicity, we process full data then subset, unless strict runtime limits apply.
    # Given the instructions, we process full data to save valid cache, then subset.

    inputs, partner_indices, targets, ids = process_dataframe(df, is_test=is_test)

    # 3. Save Cache (Only if not in debug mode to avoid corrupting cache with partial data)
    # Or save to a separate debug cache. Here we assume standard operation.
    if not debug:
        print(f"Saving processed data to {cache_path}...")
        save_dict = {"inputs": inputs, "partner_indices": partner_indices, "ids": ids}
        if targets is not None:
            save_dict["targets"] = targets

        np.savez_compressed(cache_path, **save_dict)

    # Apply debug subsetting
    if debug:
        subset = min(Config.DEBUG_SUBSET_SIZE, len(inputs))
        inputs = inputs[:subset]
        partner_indices = partner_indices[:subset]
        ids = ids[:subset]
        if targets is not None:
            targets = targets[:subset]

    return inputs, partner_indices, targets, ids


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from .npz cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    debug = Config.DEBUG

    # --- Train ---
    train_inputs, train_p_idx, train_targets, train_ids = load_data_split(
        Config.TRAIN_CSV,
        Config.TRAIN_CACHE,
        load_cached_data,
        is_test=False,
        debug=debug,
    )
    train_dataset = RNADataset(train_inputs, train_p_idx, train_targets, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Validation ---
    val_inputs, val_p_idx, val_targets, val_ids = load_data_split(
        Config.VAL_CSV, Config.VAL_CACHE, load_cached_data, is_test=False, debug=debug
    )
    val_dataset = RNADataset(val_inputs, val_p_idx, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test ---
    test_inputs, test_p_idx, _, test_ids = load_data_split(
        Config.TEST_CSV, Config.TEST_CACHE, load_cached_data, is_test=True, debug=debug
    )
    test_dataset = RNADataset(test_inputs, test_p_idx, None, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
