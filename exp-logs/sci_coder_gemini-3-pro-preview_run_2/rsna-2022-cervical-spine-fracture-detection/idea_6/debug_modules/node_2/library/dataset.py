import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from library.config import Config
from library.utils import load_dicom_slice, get_transforms


class CervicalSpineDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        transform: A.Compose = None,
        load_cached_data: bool = True,
        debug: bool = Config.DEBUG,
    ):
        """
        Dataset for 2.5D Cervical Spine Fracture Detection.

        Args:
            split (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform pipeline.
            load_cached_data (bool): Whether to load/save metadata cache.
            debug (bool): If True, subsamples the dataset for debugging.
        """
        self.split = split
        self.transform = transform if transform else get_transforms(split)
        self.debug = debug
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load metadata with file paths
        self.df = self._load_metadata(split, load_cached_data)

        # Subsample for debugging
        if self.debug:
            self.df = self.df.iloc[:20].reset_index(drop=True)
            print(f"DEBUG MODE: Reduced {split} dataset to {len(self.df)} samples.")

        # Prepare targets
        self.targets = None
        if split != "test":
            target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
            self.targets = self.df[target_cols].values.astype(np.float32)

    def _load_metadata(self, split: str, load_cached_data: bool) -> pd.DataFrame:
        """
        Loads metadata and scans directories for slice files. Caches result to Parquet.
        """
        cache_path = os.path.join(self.cache_dir, f"{split}_paths_cache.parquet")

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # pandas read_parquet handles list columns automatically with pyarrow
                df = pd.read_parquet(cache_path)

                # Validate schema for training/validation
                if split != "test":
                    required_cols = [
                        "C1",
                        "C2",
                        "C3",
                        "C4",
                        "C5",
                        "C6",
                        "C7",
                        "patient_overall",
                    ]
                    if not all(col in df.columns for col in required_cols):
                        raise ValueError("Cache missing required target columns")

                print(f"Loaded cached metadata for {split} from {cache_path}")
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")

        print(f"Generating metadata cache for {split}...")

        # Determine source file
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df = pd.read_csv(meta_path)

        # Helper to get sorted slice files
        def get_sorted_slices(rel_path):
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                return []
            try:
                files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                # Sort numerically by filename (e.g., '10.dcm' > '2.dcm')
                files.sort(key=lambda x: int(os.path.splitext(x)[0]))
                return files
            except OSError:
                return []

        # Apply to dataframe
        # Note: This might take a minute for the full dataset, but it's one-time per cache
        df["slice_files"] = df["image_path"].apply(get_sorted_slices)

        # Filter empty studies for training/validation
        if split != "test":
            initial_count = len(df)
            df = df[df["slice_files"].map(len) > 0].reset_index(drop=True)
            dropped = initial_count - len(df)
            if dropped > 0:
                print(f"Dropped {dropped} studies with no DICOM files.")

        # Save cache
        try:
            df.to_parquet(cache_path, index=False)
            print(f"Saved metadata cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slice_files = row["slice_files"]
        num_slices = len(slice_files)

        # Handle edge case of empty study (mostly for test set robustness)
        if num_slices == 0:
            dummy_img = torch.zeros(
                (Config.SEQ_LEN, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                dtype=torch.float32,
            )
            dummy_target = torch.zeros(8, dtype=torch.float32)
            return dummy_img, dummy_target

        # 1. Uniform Sampling
        # Select Config.SEQ_LEN indices evenly spaced
        indices = np.linspace(0, num_slices - 1, Config.SEQ_LEN).round().astype(int)

        base_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        stacked_images = []  # List of (H, W, 3) arrays

        # 2. 2.5D Stacking
        for i in indices:
            # Determine neighbors (clamp to boundaries)
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(num_slices - 1, i + 1)

            # Construct paths
            p_prev = os.path.join(base_path, slice_files[idx_prev])
            p_curr = os.path.join(base_path, slice_files[idx_curr])
            p_next = os.path.join(base_path, slice_files[idx_next])

            # Load slices (returns H, W in [0, 1])
            s_prev = load_dicom_slice(p_prev)
            s_curr = load_dicom_slice(p_curr)
            s_next = load_dicom_slice(p_next)

            # Stack to (H, W, 3)
            img_25d = np.stack([s_prev, s_curr, s_next], axis=-1)
            stacked_images.append(img_25d)

        # 3. Augmentation
        final_tensors = []

        if self.split == "train":
            # Volumetric Consistency: Apply same geometric params to all slices
            # Apply to first frame
            first_res = self.transform(image=stacked_images[0])
            final_tensors.append(first_res["image"])  # (3, H, W)
            replay_params = first_res["replay"]

            # Replay on subsequent frames
            for img in stacked_images[1:]:
                res = A.ReplayCompose.replay(replay_params, image=img)
                final_tensors.append(res["image"])
        else:
            # Independent application (deterministic for val/test)
            for img in stacked_images:
                res = self.transform(image=img)
                final_tensors.append(res["image"])

        # Stack sequence: (SEQ_LEN, 3, H, W)
        sequence_tensor = torch.stack(final_tensors)

        # 4. Targets
        if self.split == "test":
            target = torch.zeros(8, dtype=torch.float32)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return sequence_tensor, target
