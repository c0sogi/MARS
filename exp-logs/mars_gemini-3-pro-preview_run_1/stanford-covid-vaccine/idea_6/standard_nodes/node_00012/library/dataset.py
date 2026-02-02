import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.ids = data_dict["ids"]
        self.seq = data_dict["seq"]
        self.struct = data_dict["struct"]
        self.loop = data_dict["loop"]

        if mode != "test":
            self.targets = data_dict["targets"]
            self.masks = data_dict["masks"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Decode ID from bytes if necessary (np.savez stores strings as bytes often)
        sample_id = self.ids[idx]
        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode("utf-8")

        item = {
            "ids": sample_id,
            "sequence": torch.tensor(self.seq[idx], dtype=torch.long),
            "structure": torch.tensor(self.struct[idx], dtype=torch.long),
            "predicted_loop_type": torch.tensor(self.loop[idx], dtype=torch.long),
        }

        if self.mode != "test":
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["mask"] = torch.tensor(self.masks[idx], dtype=torch.bool)

        return item


def encode_text_column(series, vocab_map):
    """
    Encodes a pandas Series of strings into a numpy array of integers based on a vocab map.
    """
    # Convert series to list of lists of integers
    encoded = [[vocab_map[c] for c in seq] for seq in series]
    return np.array(encoded, dtype=np.int32)


def process_dataframe(df, mode="train"):
    """
    Process raw dataframe into numpy arrays.
    """
    # Encode inputs
    seq = encode_text_column(df["sequence"], Config.NUCLEOTIDE_MAP)
    struct = encode_text_column(df["structure"], Config.STRUCTURE_MAP)
    loop = encode_text_column(df["predicted_loop_type"], Config.LOOP_MAP)

    # Encode IDs as bytes to avoid pickle when saving to npz
    ids = df["id"].values.astype("S")

    data = {"ids": ids, "seq": seq, "struct": struct, "loop": loop}

    if mode != "test":
        # Process Targets
        # Targets are stored as lists in the dataframe columns.
        # We need to stack them: (N, 68) -> (N, 68, 5)
        target_arrays = []
        for col in Config.TARGET_COLS:
            # np.vstack converts the column of lists into a 2D array
            arr = np.vstack(df[col].values)  # Shape (N, 68)
            target_arrays.append(arr)

        # Stack along the last dimension -> (N, 68, 5)
        targets_68 = np.stack(target_arrays, axis=2)

        N = targets_68.shape[0]
        scored_len = targets_68.shape[1]  # Should be 68
        total_len = Config.SEQ_LEN  # 107

        # Pad targets to full sequence length (107)
        targets_107 = np.zeros((N, total_len, Config.NUM_TARGETS), dtype=np.float32)
        targets_107[:, :scored_len, :] = targets_68

        # Create mask (1 for scored positions, 0 for padded)
        masks = np.zeros((N, total_len), dtype=bool)
        masks[:, :scored_len] = True

        data["targets"] = targets_107
        data["masks"] = masks

    return data


def load_data(
    path, cache_file, load_cached_data, mode="train", debug=False, debug_size=100
):
    """
    Loads data from cache or processes from parquet.
    """
    cache_path = os.path.join(Config.CACHE_DIR, cache_file)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        # allow_pickle=False ensures we are strictly using npy format
        loaded = np.load(cache_path, allow_pickle=False)
        data = {k: loaded[k] for k in loaded.files}

        if debug:
            print(f"Debug mode: slicing first {debug_size} samples from cache")
            for k in data:
                data[k] = data[k][:debug_size]
        return data

    # 2. Process from scratch
    print(f"Processing data from {path}")
    df = pd.read_parquet(path)

    if debug:
        print(f"Debug mode: slicing first {debug_size} samples from dataframe")
        df = df.iloc[:debug_size]

    data = process_dataframe(df, mode=mode)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed data to {cache_path}")
    np.savez(cache_path, **data)

    return data


def get_dataloaders(
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load Train Data
    train_data = load_data(
        Config.TRAIN_DATA_PATH,
        "train_data.npz",
        load_cached_data,
        mode="train",
        debug=debug,
        debug_size=Config.DEBUG_SUBSET_SIZE,
    )
    train_dataset = RNADataset(train_data, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load Validation Data
    val_data = load_data(
        Config.VAL_DATA_PATH,
        "val_data.npz",
        load_cached_data,
        mode="train",  # Validation set has targets, so treat as train mode
        debug=debug,
        debug_size=Config.DEBUG_SUBSET_SIZE,
    )
    val_dataset = RNADataset(val_data, mode="train")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load Test Data
    test_data = load_data(
        Config.TEST_DATA_PATH,
        "test_data.npz",
        load_cached_data,
        mode="test",
        debug=debug,
        debug_size=Config.DEBUG_SUBSET_SIZE,
    )
    test_dataset = RNADataset(test_data, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
