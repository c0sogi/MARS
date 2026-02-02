import os
import torch
import numpy as np
import pandas as pd
import bisect
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Dict, Optional, Tuple

from library.config import Config
from library.utils import load_sensor_geometry
from library.feature_engineering import preprocess_batch

# ==========================================
# Caching Utilities
# ==========================================


class MemmapLRUCache:
    """
    A simple LRU cache for numpy memmap objects to avoid hitting file handle limits.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.cache: Dict[str, np.memmap] = {}
        self.access_order: List[str] = []

    def get(self, path: str) -> np.memmap:
        if path in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(path)
            self.access_order.append(path)
            return self.cache[path]

        # Load new file
        if len(self.cache) >= self.capacity:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
            # Garbage collection will eventually close the file handle

        try:
            mmap = np.load(path, mmap_mode="r")
            self.cache[path] = mmap
            self.access_order.append(path)
            return mmap
        except Exception as e:
            raise RuntimeError(f"Failed to load memmap at {path}: {e}")


# Global cache instance per worker process
_MEMMAP_CACHE: Optional[MemmapLRUCache] = None


def get_memmap_from_cache(path: str) -> np.memmap:
    global _MEMMAP_CACHE
    if _MEMMAP_CACHE is None:
        # Initialize cache per worker.
        # Capacity 64 is reasonable for 12 workers (12 * 64 = 768 < 1024 ulimit usually)
        _MEMMAP_CACHE = MemmapLRUCache(capacity=64)
    return _MEMMAP_CACHE.get(path)


# ==========================================
# Dataset Implementation
# ==========================================


def _preprocess_worker(args):
    """
    Helper function for parallel preprocessing.
    """
    batch_id, meta_subset, geometry = args
    # We call preprocess_batch with load_cached_data=True.
    # If the file exists, it returns quickly. If not, it computes it.
    preprocess_batch(
        batch_id,
        meta_subset,
        geometry,
        output_dir=os.path.join(Config.WORKING_DIR, "cache"),
        load_cached_data=True,
    )
    return batch_id


class IceCubeDataset(Dataset):
    def __init__(self, mode: str = "train", limit_batches: Optional[int] = None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            limit_batches (int, optional): Limit usage to N batches (for debugging).
        """
        self.mode = mode
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Load Metadata
        if mode == "train":
            meta_path = Config.TRAIN_META_PATH
        elif mode == "val":
            meta_path = Config.VAL_META_PATH
        elif mode == "test":
            meta_path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"[{mode.upper()}] Loading metadata from {meta_path}...")
        self.meta_df = pd.read_parquet(meta_path)

        # 2. Determine Batches
        unique_batches = self.meta_df["batch_id"].unique()
        unique_batches.sort()

        # Handle Debugging / Limits
        if Config.DEBUG:
            print(
                f"[{mode.upper()}] DEBUG mode enabled. Limiting to {Config.DEBUG_SAMPLE_SIZE} events approx."
            )
            # Estimate batches needed
            # Assuming ~200k events per batch, 1 batch is usually enough for debug
            limit_batches = 1 if limit_batches is None else limit_batches

        if limit_batches is not None:
            unique_batches = unique_batches[:limit_batches]
            # Filter metadata to match
            self.meta_df = self.meta_df[
                self.meta_df["batch_id"].isin(unique_batches)
            ].copy()
            print(f"[{mode.upper()}] Limited to {len(unique_batches)} batches.")

        self.batch_ids = unique_batches

        # 3. Preprocess Data (Ensure Cache Exists)
        print(
            f"[{mode.upper()}] Verifying/Preprocessing {len(self.batch_ids)} batches..."
        )

        # Load geometry once for the main process
        geometry = load_sensor_geometry()

        # Identify missing batches to avoid unnecessary overhead
        tasks = []
        for bid in self.batch_ids:
            path_X = os.path.join(self.cache_dir, f"batch_{bid}_X.npy")
            # If path doesn't exist, we must process.
            # Even if it exists, we pass it to worker to be safe?
            # No, checking existence here saves pickling overhead.
            if not os.path.exists(path_X):
                # Slice metadata for this batch
                batch_meta = self.meta_df[self.meta_df["batch_id"] == bid].copy()
                tasks.append((bid, batch_meta, geometry))

        if tasks:
            print(
                f"[{mode.upper()}] Preprocessing {len(tasks)} missing batches using {Config.NUM_WORKERS} workers..."
            )
            with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
                list(executor.map(_preprocess_worker, tasks))
        else:
            print(f"[{mode.upper()}] All batches already cached.")

        # 4. Build Index Mapping
        # We need to map global index -> (batch_id, local_index)
        # We assume the metadata is sorted by batch_id (it usually is, but we enforce sort above)
        # We'll compute the number of events per batch.

        # Group by batch_id and count
        counts = self.meta_df.groupby("batch_id").size()
        # Ensure order matches self.batch_ids
        counts = counts.reindex(self.batch_ids).fillna(0).astype(int).values

        self.batch_counts = counts
        self.batch_offsets = np.cumsum(counts)  # Cumulative sum for bisect
        # Insert 0 at the beginning for range calculations
        self.batch_offsets = np.insert(self.batch_offsets, 0, 0)

        self.total_events = self.batch_offsets[-1]
        print(f"[{mode.upper()}] Dataset ready. Total events: {self.total_events}")

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_events:
            raise IndexError(
                f"Index {idx} out of range for dataset size {self.total_events}"
            )

        # 1. Find Batch
        # self.batch_offsets = [0, N1, N1+N2, ...]
        # bisect_right returns insertion point.
        # If idx is in first batch (0 to N1-1), bisect_right returns 1.
        batch_idx = bisect.bisect_right(self.batch_offsets, idx) - 1

        batch_id = self.batch_ids[batch_idx]
        local_idx = idx - self.batch_offsets[batch_idx]

        # 2. Get File Paths
        path_X = os.path.join(self.cache_dir, f"batch_{batch_id}_X.npy")
        path_priors = os.path.join(self.cache_dir, f"batch_{batch_id}_priors.npy")
        path_y = os.path.join(self.cache_dir, f"batch_{batch_id}_y.npy")
        path_ids = os.path.join(self.cache_dir, f"batch_{batch_id}_ids.npy")

        # 3. Load Data from Memmap
        X_mmap = get_memmap_from_cache(path_X)
        priors_mmap = get_memmap_from_cache(path_priors)

        # Copy to tensor (this triggers the actual read from disk)
        X = torch.from_numpy(np.array(X_mmap[local_idx]))
        priors = torch.from_numpy(np.array(priors_mmap[local_idx]))

        # Load Targets
        if self.mode in ["train", "val"]:
            y_mmap = get_memmap_from_cache(path_y)
            y = torch.from_numpy(np.array(y_mmap[local_idx]))
        else:
            # For test set, return dummy targets or zeros
            y = torch.zeros(2, dtype=torch.float32)

        # Load Event ID (useful for submission)
        ids_mmap = get_memmap_from_cache(path_ids)
        event_id = int(ids_mmap[local_idx])

        return X, priors, y, event_id


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    limit_train_batches=None,
    limit_val_batches=None,
):
    """
    Factory function to create DataLoaders.
    """
    # Train Loader
    train_ds = IceCubeDataset(mode="train", limit_batches=limit_train_batches)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Validation Loader
    val_ds = IceCubeDataset(mode="val", limit_batches=limit_val_batches)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Factory function for Test DataLoader.
    """
    test_ds = IceCubeDataset(mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader
