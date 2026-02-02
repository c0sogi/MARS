import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library import config, utils


class BraTSDataset(Dataset):
    """
    Dataset class implementing the Siamese Hybrid-Sampling strategy.
    Generates two 12-channel views:
    1. ROI View: Centered on the max intensity FLAIR slice.
    2. Geometric View: Slices at 25%, 50%, and 75% depth.
    """

    def __init__(self, df, phase="train", transform=None):
        self.df = df
        self.phase = phase
        self.transform = transform
        self.modalities = config.MODALITIES
        self.img_size = config.IMG_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Determine Indices
        # Retrieve cached best index (ROI center) and slice count
        best_idx = int(row.get("best_flair_index", 0))
        num_slices = int(row.get("num_flair_slices", 0))

        # Handle edge case where folder might be empty
        if num_slices < 1:
            num_slices = 1
            best_idx = 0

        # ROI View Indices: [best-1, best, best+1]
        roi_indices = [best_idx - 1, best_idx, best_idx + 1]
        # Clamp indices to valid range
        roi_indices = [max(0, min(i, num_slices - 1)) for i in roi_indices]

        # 2. Load Images
        roi_channels = []

        for mod in self.modalities:
            # Construct path to modality folder
            dir_path = os.path.join(config.INPUT_DIR, row[f"path_{mod}"])

            # Get sorted files using util
            files = utils.get_sorted_files(dir_path)
            mod_slices = len(files)

            if mod_slices == 0:
                # Missing modality: Fill with zeros
                zeros = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                for _ in range(3):
                    roi_channels.append(zeros)
                continue

            # Helper to map FLAIR index to current modality index
            def get_slice_img(target_idx_flair_space):
                if num_slices > 0:
                    # Map index proportionally if slice counts differ
                    ratio = target_idx_flair_space / num_slices
                    idx_mod = int(ratio * mod_slices)
                else:
                    idx_mod = 0

                idx_mod = max(0, min(idx_mod, mod_slices - 1))
                f_name = files[idx_mod]
                full_path = os.path.join(dir_path, f_name)
                # Load and normalize slice
                return utils.load_dicom_slice(full_path, size=self.img_size)

            # Load ROI slices
            for i in roi_indices:
                roi_channels.append(get_slice_img(i))

        # Stack: (C, H, W) where C = 12 (3 slices * 4 modalities)
        roi_tensor = np.stack(roi_channels, axis=0)

        # 3. Augmentations
        if self.transform:
            # Albumentations expects HWC format
            roi_hwc = np.transpose(roi_tensor, (1, 2, 0))

            # Apply transform
            res_roi = self.transform(image=roi_hwc)
            roi_hwc = res_roi["image"]

            # Convert back to CHW
            roi_tensor = np.transpose(roi_hwc, (2, 0, 1))

        # Convert to torch tensor
        roi_tensor = torch.from_numpy(roi_tensor).float()

        # Return Logic
        if self.phase in ["train", "val"]:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return roi_tensor, target
        else:
            return roi_tensor


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    Strictly geometric: Rotate, Flip. No destructive regularization.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
            ]
        )
    return None


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached_data=True
):
    """
    Orchestrates metadata loading, processing (best slice computation),
    and DataLoader creation.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 1. Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Compute Best Slices (with Caching)
    # Using the utility function provided in library.utils to scan FLAIR images
    # and cache results to parquet files.
    print("Preparing Training Data...")
    df_train = utils.compute_best_slices(
        df_train, cache_name="train", load_cached_data=load_cached_data
    )

    print("Preparing Validation Data...")
    df_val = utils.compute_best_slices(
        df_val, cache_name="val", load_cached_data=load_cached_data
    )

    print("Preparing Test Data...")
    df_test = utils.compute_best_slices(
        df_test, cache_name="test", load_cached_data=load_cached_data
    )

    # 3. Create Datasets
    train_ds = BraTSDataset(df_train, phase="train", transform=get_transforms("train"))
    val_ds = BraTSDataset(df_val, phase="val", transform=None)
    test_ds = BraTSDataset(df_test, phase="test", transform=None)

    # 4. Create Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Helps with BatchNorm stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
