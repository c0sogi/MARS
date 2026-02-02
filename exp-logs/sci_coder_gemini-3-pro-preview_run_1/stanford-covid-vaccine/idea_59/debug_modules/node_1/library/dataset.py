import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config, get_structure_distance


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data["seq"])

    def __getitem__(self, idx):
        # Retrieve data from the dictionary of arrays
        seq = self.data["seq"][idx]
        loop = self.data["loop"][idx]
        dist = self.data["dist"][idx]

        # Convert to torch tensors
        item = {
            "seq": torch.from_numpy(seq),
            "loop": torch.from_numpy(loop),
            "dist": torch.from_numpy(dist),
        }

        if self.mode in ["train", "val"]:
            targets = self.data["targets"][idx]
            mask = self.data["mask"][idx]
            item["targets"] = torch.from_numpy(targets)
            item["mask"] = torch.from_numpy(mask)
        else:
            item["id"] = self.data["id"][idx]

        return item


def preprocess_data(df, mode):
    """
    Converts DataFrame columns to numpy arrays for efficient storage and access.
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Initialize lists
    seqs = []
    loops = []
    dists = []
    ids = df["id"].values

    # Process sequences, loops, and structures
    for _, row in df.iterrows():
        # Sequence
        s = [seq_map.get(c, 0) for c in row["sequence"]]
        seqs.append(np.array(s, dtype=np.int64))

        # Loop
        l = [loop_map.get(c, 0) for c in row["predicted_loop_type"]]
        loops.append(np.array(l, dtype=np.int64))

        # Distance (using imported helper)
        d = get_structure_distance(row["structure"])
        dists.append(d)

    data = {
        "seq": np.stack(seqs),
        "loop": np.stack(loops),
        "dist": np.stack(dists),
        "id": ids,
    }

    # Process targets for training/validation sets
    if mode in ["train", "val"]:
        targets_list = []
        masks_list = []

        for _, row in df.iterrows():
            t_sample = []
            for col in Config.TARGET_COLS:
                val = np.array(row[col], dtype=np.float32)
                # Pad to SEQ_LEN if necessary
                if len(val) < Config.SEQ_LEN:
                    pad = np.zeros(Config.SEQ_LEN - len(val), dtype=np.float32)
                    val = np.concatenate([val, pad])
                t_sample.append(val)

            t_sample = np.stack(t_sample, axis=1)  # (SEQ_LEN, 3)
            targets_list.append(t_sample)

            # Create mask for scored positions
            m = np.zeros(Config.SEQ_LEN, dtype=np.float32)
            m[: Config.PRED_LEN] = 1.0
            masks_list.append(m)

        data["targets"] = np.stack(targets_list)
        data["mask"] = np.stack(masks_list)

    return data


def load_data(mode, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from metadata parquet files.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_file = os.path.join(Config.WORKING_DIR, f"cached_{mode}.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        loaded = np.load(cache_file, allow_pickle=True)
        # Convert NpzFile to dict
        data = {k: loaded[k] for k in loaded.files}
        return data

    # Process from scratch
    print(f"Processing {mode} data from metadata...")
    parquet_path = os.path.join(Config.METADATA_DIR, f"{mode}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    data = preprocess_data(df, mode)

    # Save to cache
    np.savez_compressed(cache_file, **data)
    print(f"Cached {mode} data to {cache_file}")

    return data


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    train_data = load_data("train", load_cached_data)
    val_data = load_data("val", load_cached_data)
    test_data = load_data("test", load_cached_data)

    train_ds = RNADataset(train_data, mode="train")
    val_ds = RNADataset(val_data, mode="val")
    test_ds = RNADataset(test_data, mode="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
