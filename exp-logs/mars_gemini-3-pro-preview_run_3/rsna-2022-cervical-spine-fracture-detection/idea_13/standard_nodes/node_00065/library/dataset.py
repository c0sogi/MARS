import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2

from library.config import Config
from library import utils


class RSNADataset(Dataset):
    def __init__(self, subset="train", n_samples=None):
        """
        Dataset for RSNA Cervical Spine Fracture Detection.

        Args:
            subset (str): One of 'train', 'val', 'test'.
            n_samples (int, optional): Limit the number of samples for debugging.
        """
        self.subset = subset
        # Map subset to data_type expected by utils.get_transforms
        self.data_type = "train" if subset == "train" else "valid"

        # Load Metadata
        if subset == "train":
            meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
        elif subset == "val":
            meta_path = os.path.join(Config.METADATA_DIR, "val_metadata.csv")
        elif subset == "test":
            meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.csv")
        else:
            raise ValueError(f"Invalid subset: {subset}")

        self.df = pd.read_csv(meta_path)

        # Debugging sample limit
        if n_samples is not None:
            self.df = self.df.iloc[:n_samples]
        elif Config.N_SAMPLES is not None:
            self.df = self.df.iloc[: Config.N_SAMPLES]

        # Setup Transforms
        # utils.get_transforms returns A.Compose.
        # Train: [ShiftScaleRotate, Resize]
        # Valid: [Resize]
        transforms_list = utils.get_transforms(self.data_type).transforms

        if subset == "train":
            # Separate geometric augmentation for consistent application across the volume
            # We use ReplayCompose to record parameters on a dummy image and replay on all slices
            self.geo_transform = A.ReplayCompose([transforms_list[0]])
            self.resize_transform = transforms_list[1]
        else:
            self.geo_transform = None
            self.resize_transform = transforms_list[0]

        # Local stream uses a Center Crop to preserve high-frequency details
        self.center_crop = A.CenterCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE)

        # ImageNet Normalization Constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _load_volume(self, study_id, image_path_rel):
        """
        Loads volume, processing from DICOM if not cached.

        Cache Format: .npy file, uint8 (0-255) to save space/IO.
        Return Format: float32 (0.0-1.0).
        """
        cache_file = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

        # 1. Try loading from cache
        if os.path.exists(cache_file):
            try:
                vol_uint8 = np.load(cache_file)
                # Verify cache consistency against runtime configuration (Cite debug_lesson_2)
                # Ensure cached volume dimensions are sufficient for the requested crop size
                if (
                    vol_uint8.shape[1] >= Config.IMAGE_SIZE
                    and vol_uint8.shape[2] >= Config.IMAGE_SIZE
                ):
                    return vol_uint8.astype(np.float32) / 255.0
            except Exception:
                pass  # Fallback to load from source if cache is corrupt or incompatible

        # 2. Load from Source
        # image_path_rel is e.g. "train_images/1.2.3..."
        # utils.load_dicom_volume expects the root dir and the study_id separately
        full_path = os.path.join(Config.INPUT_DIR, image_path_rel)
        parent_dir = os.path.dirname(full_path)

        vol = utils.load_dicom_volume(study_id, parent_dir)

        if vol.shape[0] == 0:
            return vol

        # Windowing (Bone Window) -> Normalizes to 0-1
        vol = utils.window_image(vol, Config.WINDOW_LEVEL, Config.WINDOW_WIDTH)

        # Save to cache (convert to uint8 to maximize throughput)
        np.save(cache_file, (vol * 255).astype(np.uint8))

        return vol

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]
        image_path = row["image_path"]

        # 1. Load Volume
        volume = self._load_volume(study_id, image_path)

        # Handle empty volume edge case
        if volume.shape[0] == 0:
            return (
                torch.zeros(
                    (Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                ),
                torch.zeros(
                    (Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                ),
                torch.zeros((Config.NUM_CLASSES,)),
            )

        # 2. Uniform Sampling
        depth = volume.shape[0]
        indices = np.linspace(0, depth - 1, Config.NUM_SLICES).astype(int)

        # 3. Prepare Augmentation (Consistent across the bag)
        replay_params = None
        if self.geo_transform is not None:
            # Generate params using a dummy image once per bag
            # Assuming standard 512x512 input size for transformation calculation
            dummy = np.zeros((512, 512, 3), dtype=np.uint8)
            res = self.geo_transform(image=dummy)
            replay_params = res["replay"]

        image_stack = []

        # 4. Iterate slices to generate inputs
        for i in indices:
            # 2.5D Construction (z-1, z, z+1)
            # Clamp indices to volume boundaries
            idx_prev = max(0, i - 1)
            idx_next = min(depth - 1, i + 1)

            # Stack channels -> (H, W, 3)
            slice_25d = np.stack(
                [volume[idx_prev], volume[i], volume[idx_next]], axis=-1
            )

            # Apply Geometric Transform (Shift/Scale/Rotate)
            # Must be identical for all slices in the bag
            if self.geo_transform is not None:
                augmented = A.ReplayCompose.replay(replay_params, image=slice_25d)[
                    "image"
                ]
            else:
                augmented = slice_25d

            # Resize to model input size
            img = self.resize_transform(image=augmented)["image"]

            # Normalize (ImageNet stats)
            img = (img - self.mean) / self.std

            # Transpose to Channel-First (C, H, W)
            img = img.transpose(2, 0, 1)

            image_stack.append(img)

        # Stack into tensors (Batch, C, H, W) where Batch = NUM_SLICES
        inputs = torch.tensor(np.stack(image_stack), dtype=torch.float32)

        # 5. Targets
        if self.subset == "test":
            targets = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        else:
            # Columns: C1..C7, patient_overall
            cols = [f"C{k}" for k in range(1, 8)] + ["patient_overall"]
            targets = torch.tensor(
                row[cols].values.astype(np.float32), dtype=torch.float32
            )

        return inputs, targets
