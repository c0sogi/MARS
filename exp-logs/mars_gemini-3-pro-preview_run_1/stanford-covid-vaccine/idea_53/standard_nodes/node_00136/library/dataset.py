import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Holds pre-processed tensors for sequences, structures, and targets.
    """

    def __init__(self, data, mode="train"):
        self.sequences = data["sequences"]
        self.loops = data["loops"]
        self.distances = data["distances"]
        self.ids = data["ids"]
        self.mode = mode

        # Targets are only present for train and val sets
        self.targets = data.get("targets", None)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        loop = self.loops[idx]
        dist = self.distances[idx]
        sample_id = self.ids[idx]

        if self.mode in ["train", "val"]:
            target = self.targets[idx]
            return {
                "sequence": seq,
                "loop_type": loop,
                "pairing_distance": dist,
                "target": target,
                "id": sample_id,
            }
        else:
            return {
                "sequence": seq,
                "loop_type": loop,
                "pairing_distance": dist,
                "id": sample_id,
            }


def parse_structure_to_distance(structure_str):
    """
    Parses a dot-bracket structure string into a signed pairing distance array.
    Distance = paired_index - current_index. Unpaired = 0.
    """
    n = len(structure_str)
    distances = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Pair is (j, i) where j < i
                # Distance at j is i - j (positive)
                # Distance at i is j - i (negative)
                distances[j] = i - j
                distances[i] = j - i

    return distances


def process_dataframe(df, mode="train"):
    """
    Converts a pandas DataFrame into a dictionary of tensors.
    """
    # 1. Process Sequence
    # Convert 'AGCU' strings to lists of indices
    sequences = []
    for seq in df["sequence"]:
        sequences.append([SEQ_MAP.get(c, 0) for c in seq])
    sequences = torch.tensor(sequences, dtype=torch.long)

    # 2. Process Loop Type
    loops = []
    for loop in df["predicted_loop_type"]:
        loops.append([LOOP_MAP.get(c, 6) for c in loop])  # Default to X (6) if unknown
    loops = torch.tensor(loops, dtype=torch.long)

    # 3. Process Structure (Pairing Distance)
    distances = []
    for struct in df["structure"]:
        distances.append(parse_structure_to_distance(struct))
    distances = torch.tensor(np.array(distances), dtype=torch.float32)
    # Add channel dimension for embedding layer: (N, Seq) -> (N, Seq, 1)
    distances = distances.unsqueeze(-1)

    # 4. Process IDs
    ids = df["id"].tolist()

    data_dict = {
        "sequences": sequences,
        "loops": loops,
        "distances": distances,
        "ids": ids,
    }

    # 5. Process Targets (if applicable)
    if mode in ["train", "val"]:
        # Stack the target columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Each column in df is a list of floats (length 68)
        # Result shape: (N, 68, 3)
        target_arrays = []
        for col in Config.TARGET_COLS:
            # vstack converts the column of lists into a 2D array
            arr = np.vstack(df[col].values)
            target_arrays.append(arr)

        # Stack along the last dimension (channels)
        targets = np.stack(target_arrays, axis=2)
        data_dict["targets"] = torch.tensor(targets, dtype=torch.float32)

    return data_dict


def get_dataset(mode="train", load_cached_data=True):
    """
    Factory function to get an RNADataset. Handles caching.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        RNADataset: The instantiated dataset.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{mode}.pt")

    # Ensure cache directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data_dict = torch.load(cache_path)
            return RNADataset(data_dict, mode=mode)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Load raw data
    print(f"Processing {mode} data from source...")
    if mode == "train":
        file_path = Config.TRAIN_FILE
    elif mode == "val":
        file_path = Config.VAL_FILE
    else:
        file_path = Config.TEST_FILE

    df = pd.read_parquet(file_path)

    # Process data
    data_dict = process_dataframe(df, mode=mode)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_path}")
    torch.save(data_dict, cache_path)

    return RNADataset(data_dict, mode=mode)


def collate_fn(batch):
    """
    Custom collate function to stack batch items.
    """
    sequences = torch.stack([item["sequence"] for item in batch])
    loops = torch.stack([item["loop_type"] for item in batch])
    distances = torch.stack([item["pairing_distance"] for item in batch])
    ids = [item["id"] for item in batch]

    if "target" in batch[0]:
        targets = torch.stack([item["target"] for item in batch])
        return {
            "sequence": sequences,
            "loop_type": loops,
            "pairing_distance": distances,
            "target": targets,
            "id": ids,
        }
    else:
        return {
            "sequence": sequences,
            "loop_type": loops,
            "pairing_distance": distances,
            "id": ids,
        }


def get_dataloader(
    mode="train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
):
    """
    Creates a DataLoader for the specified mode.
    """
    dataset = get_dataset(mode, load_cached_data=load_cached_data)

    # For test set, we usually don't shuffle
    if mode == "test":
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
