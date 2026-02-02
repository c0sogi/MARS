import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import window_dicom


def load_volume(study_id, study_dir, cache_dir, load_cached_data=True):
    """
    Loads a 3D volume for a study with caching mechanism.

    Args:
        study_id (str): The StudyInstanceUID.
        study_dir (str): The directory containing the DICOM files for this study.
        cache_dir (str): Directory to store/retrieve cached .npy files.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: A float32 array of shape (N, H, W) with values in [0, 1].
    """
    cache_path = os.path.join(cache_dir, f"{study_id}.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            # Validate shape: Must be (N, H, W) and H, W must match Config.IMAGE_SIZE
            if volume.ndim == 3 and volume.shape[1:] == Config.IMAGE_SIZE:
                return volume
            else:
                print(
                    f"Invalid cache shape {volume.shape} for {study_id}. Recomputing."
                )
        except Exception as e:
            print(f"Failed to load cache for {study_id}: {e}. Recomputing.")

    # 2. Compute from scratch
    if not os.path.exists(study_dir):
        # Fallback: return a zero volume if directory missing (should not happen with valid metadata)
        print(f"Warning: Directory not found {study_dir}")
        return np.zeros((1, *Config.IMAGE_SIZE), dtype=np.float32)

    files = glob.glob(os.path.join(study_dir, "*.dcm"))
    if not files:
        print(f"Warning: No DICOM files in {study_dir}")
        return np.zeros((1, *Config.IMAGE_SIZE), dtype=np.float32)

    # Sort files by instance number (filename integer)
    # Assumes filenames are like '1.dcm', '10.dcm'
    try:
        files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    except ValueError:
        # Fallback if filenames are not integers
        files.sort()

    processed_slices = []

    for f in files:
        try:
            ds = pydicom.dcmread(f)

            # Convert to float and Apply Rescale Slope/Intercept if present to get HU
            img = ds.pixel_array.astype(np.float32)

            slope = getattr(ds, "RescaleSlope", 1.0)
            intercept = getattr(ds, "RescaleIntercept", 0.0)

            if slope != 1.0:
                img = img * slope
            if intercept != 0.0:
                img = img + intercept

            # Apply Windowing (returns [0, 1] float32)
            img = window_dicom(img, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)

            # Resize
            if img.shape != Config.IMAGE_SIZE:
                img = cv2.resize(img, Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

            processed_slices.append(img)
        except Exception as e:
            # Skip corrupted slices
            continue

    if not processed_slices:
        return np.zeros((1, *Config.IMAGE_SIZE), dtype=np.float32)

    volume = np.stack(processed_slices, axis=0)  # (N, H, W)

    # 3. Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    try:
        np.save(cache_path, volume)
    except Exception as e:
        print(f"Failed to save cache for {study_id}: {e}")

    return volume


class CervicalSpineDataset(Dataset):
    def __init__(self, df, mode="train", transforms=None, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            transforms (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Ensure transforms support replay for volumetric consistency
        if transforms is not None and not isinstance(transforms, A.ReplayCompose):
            # Wrap in ReplayCompose if it's a standard Compose
            # Note: A.Compose has a 'transforms' attribute which is a list
            if hasattr(transforms, "transforms"):
                self.transforms = A.ReplayCompose(transforms.transforms)
            else:
                self.transforms = A.ReplayCompose([transforms])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # 1. Load Volume
        volume = load_volume(
            study_id, full_path, Config.CACHE_DIR, self.load_cached_data
        )
        num_slices = volume.shape[0]

        # 2. Sample Slices (Uniformly)
        # We need exactly BAG_SIZE slices.
        if num_slices > 0:
            indices = np.linspace(0, num_slices - 1, Config.BAG_SIZE).astype(int)
        else:
            indices = np.zeros(Config.BAG_SIZE, dtype=int)

        # 3. Construct 2.5D Stacks and Apply Augmentation
        bag_images = []
        bag_positions = []

        # Prepare Replay parameters for consistent augmentation across the bag
        replay_params = None
        if self.transforms:
            # Generate params using a dummy image or the first slice
            dummy_data = np.zeros((*Config.IMAGE_SIZE, 3), dtype=np.float32)
            res = self.transforms(image=dummy_data)
            replay_params = res["replay"]

        for i in indices:
            # 2.5D Stacking: (z-1, z, z+1)
            # Handle boundary conditions by clamping
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(num_slices - 1, i + 1)

            s_prev = volume[idx_prev]
            s_curr = volume[idx_curr]
            s_next = volume[idx_next]

            # Stack to (H, W, 3) for Albumentations
            stack = np.stack([s_prev, s_curr, s_next], axis=-1)

            # Apply Augmentation
            if self.transforms and replay_params:
                augmented = self.transforms.replay(replay_params, image=stack)
                stack = augmented["image"]

            # Convert to Tensor (C, H, W)
            if isinstance(stack, torch.Tensor):
                # ToTensorV2 already converts to CHW
                pass
            else:
                # Manual conversion if ToTensorV2 not in pipeline
                stack = torch.from_numpy(stack.transpose(2, 0, 1))

            bag_images.append(stack)

            # Positional Encoding (Normalized Depth)
            d_z = i / max(1, num_slices)
            bag_positions.append(d_z)

        # Stack into bag tensors
        bag_images = torch.stack(bag_images)  # (Bag_Size, C, H, W)
        bag_positions = torch.tensor(bag_positions, dtype=torch.float32).unsqueeze(
            1
        )  # (Bag_Size, 1)

        # 4. Prepare Targets
        if self.mode != "test":
            # Labels: C1-C7 (7) + Patient Overall (1)
            # Columns are C1, C2, ..., C7, patient_overall
            c_labels = [row[f"C{k}"] for k in range(1, 8)]
            overall_label = row["patient_overall"]

            # Combine: [C1, C2, ..., C7, patient_overall]
            targets = torch.tensor(c_labels + [overall_label], dtype=torch.float32)
        else:
            # Dummy targets for test
            targets = torch.zeros(8, dtype=torch.float32)

        return bag_images, bag_positions, targets, study_id
