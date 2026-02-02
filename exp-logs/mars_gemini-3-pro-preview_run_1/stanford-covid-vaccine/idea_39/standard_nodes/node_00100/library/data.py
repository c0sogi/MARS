import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_pair_distance_vector, TOKEN_MAP, LOOP_TYPE_MAP


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA sequences.
    Holds pre-processed tensors for inputs and targets.
    """

    def __init__(self, data_dict, mode="train"):
        self.seqs = data_dict["seqs"]
        self.loops = data_dict["loops"]
        self.dists = data_dict["dists"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode != "test":
            self.targets = data_dict["targets"]
            self.masks = data_dict["masks"]
        else:
            self.targets = None
            self.masks = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "seq": self.seqs[idx],
            "loop": self.loops[idx],
            "dist": self.dists[idx],
            "id": self.ids[idx],
        }

        if self.mode != "test":
            item["targets"] = self.targets[idx]
            item["mask"] = self.masks[idx]

        return item


def process_data(df, mode="train", config=None):
    """
    Converts a pandas DataFrame into a dictionary of tensors.
    """
    if config is None:
        config = Config()

    # Pre-allocate lists
    seqs = []
    loops = []
    dists = []
    ids = df["id"].tolist()

    # Target processing lists
    targets_list = []
    masks_list = []

    seq_len = config.seq_len
    pred_len = config.pred_len
    target_cols = config.target_cols

    for _, row in df.iterrows():
        # 1. Sequence Tokenization
        seq_str = row["sequence"]
        seq_encoded = [TOKEN_MAP.get(c, 0) for c in seq_str]
        seqs.append(torch.tensor(seq_encoded, dtype=torch.long))

        # 2. Loop Type Tokenization
        loop_str = row["predicted_loop_type"]
        loop_encoded = [LOOP_TYPE_MAP.get(c, 0) for c in loop_str]
        loops.append(torch.tensor(loop_encoded, dtype=torch.long))

        # 3. Signed Distance Vector
        struct_str = row["structure"]
        dist_vec = get_pair_distance_vector(struct_str)
        # Keep as signed integers for the model to handle (e.g. sinusoidal encoding)
        dists.append(torch.tensor(dist_vec, dtype=torch.long))

        # 4. Targets (only for train/val)
        if mode != "test":
            # Extract specific target columns.
            # Each column in the parquet is a list/array of length pred_len (68).
            sample_targets = []
            for col in target_cols:
                val = row[col]
                # Ensure it's a list or array
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                sample_targets.append(val)

            # Stack to shape (3, 68) then transpose to (68, 3)
            t_tensor = torch.tensor(sample_targets, dtype=torch.float32).T

            # Pad to seq_len (107)
            # Create a full tensor of zeros (107, 3)
            full_targets = torch.zeros((seq_len, len(target_cols)), dtype=torch.float32)
            full_targets[:pred_len, :] = t_tensor
            targets_list.append(full_targets)

            # Create Mask: 1.0 for scored positions, 0.0 for others
            mask = torch.zeros(seq_len, dtype=torch.float32)
            mask[:pred_len] = 1.0
            masks_list.append(mask)

    # Stack lists into tensors
    data_dict = {
        "seqs": torch.stack(seqs),
        "loops": torch.stack(loops),
        "dists": torch.stack(dists),
        "ids": ids,
    }

    if mode != "test":
        data_dict["targets"] = torch.stack(targets_list)
        data_dict["masks"] = torch.stack(masks_list)

    return data_dict


def load_data(mode="train", config=None, load_cached_data=True):
    """
    Loads data from parquet, processes it, and caches it to disk.
    """
    if config is None:
        config = Config()

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    cache_path = os.path.join(config.working_dir, f"{mode}_data.pt")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data_dict = torch.load(cache_path)
            return RNADataset(data_dict, mode=mode)
        except Exception as e:
            # If load fails, fall through to processing
            pass

    # 2. Determine source file
    if mode == "train":
        file_path = config.train_file
    elif mode == "val":
        file_path = config.val_file
    elif mode == "test":
        file_path = config.test_file
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3. Load and Process
    df = pd.read_parquet(file_path)
    data_dict = process_data(df, mode=mode, config=config)

    # 4. Save to cache
    torch.save(data_dict, cache_path)

    return RNADataset(data_dict, mode=mode)


def get_dataloaders(config=None, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    if config is None:
        config = Config()

    train_dataset = load_data("train", config, load_cached_data)
    val_dataset = load_data("val", config, load_cached_data)
    test_dataset = load_data("test", config, load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )

    return train_loader, val_loader, test_loader
