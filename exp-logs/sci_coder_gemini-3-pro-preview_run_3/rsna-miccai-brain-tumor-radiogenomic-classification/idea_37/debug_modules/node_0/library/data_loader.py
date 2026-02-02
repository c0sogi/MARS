import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config, utils


class SiameseSNRDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Native-Resolution 2.5D Network.
    Holds pre-processed tensors for Even and Odd streams.
    """

    def __init__(self, X_even, X_odd, y=None, ids=None):
        self.X_even = X_even
        self.X_odd = X_odd
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X_even)

    def __getitem__(self, idx):
        # Convert numpy to torch tensor
        tensor_even = torch.from_numpy(self.X_even[idx])
        tensor_odd = torch.from_numpy(self.X_odd[idx])

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return tensor_even, tensor_odd, label
        else:
            return tensor_even, tensor_odd


def _process_data(df, split_name):
    """
    Internal function to process raw DICOMs into numpy arrays based on the
    Siamese SNR logic (32 slices -> 16 Even + 16 Odd).
    """
    print(f"Processing {split_name} data from scratch...")

    X_even_list = []
    X_odd_list = []
    y_list = []
    ids_list = []

    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Pre-calculate expected slices per view (Even/Odd) per modality
    # Total 32 slices -> 16 Even, 16 Odd
    slices_per_view = config.NUM_SLICES_PER_MODALITY // 2

    count = 0
    total = len(df)

    for _, row in df.iterrows():
        bra_id = row["BraTS21ID"]

        # Initialize lists to hold blocks for this subject
        subject_even_blocks = []
        subject_odd_blocks = []

        for mod in modalities:
            col_name = f"{mod}_paths"
            paths = row[col_name] if row[col_name] is not None else []

            # 1. Sort paths
            sorted_paths = utils.get_sorted_file_paths(list(paths))
            num_files = len(sorted_paths)

            if num_files == 0:
                # Handle missing modality with zeros
                # Shape: (16, 224, 224)
                zero_block = np.zeros(
                    (slices_per_view, config.IMG_SIZE, config.IMG_SIZE),
                    dtype=np.float32,
                )
                subject_even_blocks.append(zero_block)
                subject_odd_blocks.append(zero_block)
                continue

            # 2. High-Density Uniform Sampling (10%-90%)
            # We need 32 indices total
            target_count = config.NUM_SLICES_PER_MODALITY

            if num_files < target_count:
                # If fewer files than target, use linspace over the whole range
                indices = (
                    np.linspace(0, num_files - 1, target_count).round().astype(int)
                )
            else:
                # 10% to 90% depth
                start_idx = int(num_files * 0.1)
                end_idx = int(num_files * 0.9)

                # Safety check if range is too small
                if start_idx >= end_idx:
                    start_idx = 0
                    end_idx = num_files - 1

                indices = (
                    np.linspace(start_idx, end_idx, target_count).round().astype(int)
                )
                indices = np.clip(indices, 0, num_files - 1)

            # 3. Deterministic Strided Splitting
            # Even indices: 0, 2, 4...
            # Odd indices: 1, 3, 5...
            even_indices = indices[0::2]
            odd_indices = indices[1::2]

            even_paths = [sorted_paths[i] for i in even_indices]
            odd_paths = [sorted_paths[i] for i in odd_indices]

            # 4. Load and Process (includes View-Adaptive Normalization)
            block_even = utils.load_and_process_modality_block(even_paths)
            block_odd = utils.load_and_process_modality_block(odd_paths)

            subject_even_blocks.append(block_even)
            subject_odd_blocks.append(block_odd)

        # 5. Modality-Grouped Stacking
        # Concatenate blocks along channel dimension (axis 0)
        # Result shape: (64, 224, 224)
        X_even_subject = np.concatenate(subject_even_blocks, axis=0)
        X_odd_subject = np.concatenate(subject_odd_blocks, axis=0)

        X_even_list.append(X_even_subject)
        X_odd_list.append(X_odd_subject)
        ids_list.append(bra_id)

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

        count += 1
        if config.DEBUG and count >= config.DEBUG_SAMPLE_SIZE:
            break

    # Convert to numpy arrays
    X_even = np.array(X_even_list, dtype=np.float32)
    X_odd = np.array(X_odd_list, dtype=np.float32)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    return X_even, X_odd, y, ids


def get_dataloader(
    split, batch_size=config.BATCH_SIZE, load_cached_data=True, shuffle=False
):
    """
    Factory function to get a DataLoader with caching mechanism.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to attempt loading from cache.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Define cache file paths
    cache_X_even = os.path.join(config.CACHE_DIR, f"X_{split}_even.npy")
    cache_X_odd = os.path.join(config.CACHE_DIR, f"X_{split}_odd.npy")
    cache_y = os.path.join(config.CACHE_DIR, f"y_{split}.npy")
    cache_ids = os.path.join(config.CACHE_DIR, f"ids_{split}.npy")

    data_loaded = False
    X_even, X_odd, y, ids = None, None, None, None

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(cache_X_even)
            and os.path.exists(cache_X_odd)
            and os.path.exists(cache_ids)
        ):
            print(f"Loading {split} data from cache: {config.CACHE_DIR}")
            try:
                X_even = np.load(cache_X_even)
                X_odd = np.load(cache_X_odd)
                ids = np.load(cache_ids)
                if os.path.exists(cache_y):
                    y = np.load(cache_y)
                data_loaded = True
            except Exception as e:
                print(f"Error loading cache: {e}. Re-processing.")
                data_loaded = False
        else:
            print(f"Cache not found for {split}.")

    # 2. Process from scratch if needed
    if not data_loaded:
        # Load metadata
        if split == "train":
            meta_path = config.TRAIN_META_PATH
        elif split == "val":
            meta_path = config.VAL_META_PATH
        elif split == "test":
            meta_path = config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_parquet(meta_path)

        # Process
        X_even, X_odd, y, ids = _process_data(df, split)

        # Save to cache
        print(f"Saving {split} data to cache...")
        np.save(cache_X_even, X_even)
        np.save(cache_X_odd, X_odd)
        np.save(cache_ids, ids)
        if y is not None:
            np.save(cache_y, y)

    # 3. Create Dataset and DataLoader
    dataset = SiameseSNRDataset(X_even, X_odd, y, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return loader
