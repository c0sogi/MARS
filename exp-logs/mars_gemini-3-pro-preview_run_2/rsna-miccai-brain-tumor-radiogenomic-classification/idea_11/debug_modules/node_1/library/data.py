import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import (
    read_dicom_robust,
    preprocess_image,
    get_roi_indices,
    get_image_plane,
)


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(Config.SEED)


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------
class MGMTDataset(Dataset):
    """
    Dataset for MGMT Promoter Methylation Prediction.

    Constructs a 12-channel volume:
    - 4 Modalities: FLAIR, T1w, T1wCE, T2w
    - 3 Slices per modality: Anchor-Stride, Anchor, Anchor+Stride

    The order of channels is Modality-Major (FLAIR_slices, T1w_slices, ...)
    to align with the Grouped Convolution stem of the model.
    """

    def __init__(self, df, roi_map, transform=None):
        self.df = df
        self.roi_map = roi_map
        self.transform = transform
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.stride = Config.STRIDE
        # Offsets for the 3 slices: [Anchor-5, Anchor, Anchor+5]
        self.offsets = [-self.stride, 0, self.stride]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = row["BraTS21ID"]

        # Retrieve the calculated anchor slice index for this subject
        # Default to 0 if not found (though get_roi_indices should cover all)
        anchor_idx = self.roi_map.get(brats_id, 0)

        channels = []

        # Iterate through modalities to build the 12-channel input
        for mod in self.modalities:
            # Construct path to the modality directory
            mod_path_rel = row[f"path_{mod}"]
            full_mod_path = os.path.join(Config.INPUT_DIR, mod_path_rel)

            # List and sort DICOM files anatomically
            files = []
            if os.path.exists(full_mod_path):
                raw_files = os.listdir(full_mod_path)
                # Filter for valid DICOMs
                files = [f for f in raw_files if "Image-" in f]
                files.sort(key=get_image_plane)

            num_files = len(files)

            # Retrieve the specific slices for this modality
            for offset in self.offsets:
                target_idx = anchor_idx + offset

                img = None

                if num_files > 0:
                    # Clamp index to valid range (Boundary Handling)
                    safe_idx = max(0, min(target_idx, num_files - 1))

                    file_name = files[safe_idx]
                    file_path = os.path.join(full_mod_path, file_name)

                    # Robust Read & Preprocess
                    # read_dicom_robust handles tail-reads for corrupt headers
                    raw_img = read_dicom_robust(file_path)

                    # preprocess_image handles Float32 conversion, Resizing (Area), and Normalization
                    img = preprocess_image(
                        raw_img, target_size=(Config.IMG_SIZE, Config.IMG_SIZE)
                    )

                # Fallback for empty folders or failed reads
                if img is None:
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

                channels.append(img)

        # Stack channels to create (H, W, 12) volume
        image = np.stack(channels, axis=-1)

        # Apply Geometric Augmentations (if any)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert to PyTorch Tensor: (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Get Target Label
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32).unsqueeze(0)
        else:
            # Dummy label for test set
            label = torch.tensor(-1.0, dtype=torch.float32)

        return image, label


# ------------------------------------------------------------------------------
# Data Loading Factory
# ------------------------------------------------------------------------------
def get_dataloader(
    split, batch_size=None, num_workers=None, load_cached_data=True, debug=False
):
    """
    Creates a DataLoader for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size. Defaults to Config.BATCH_SIZE.
        num_workers (int): Number of worker threads. Defaults to Config.NUM_WORKERS.
        load_cached_data (bool): Whether to load ROI indices from cache.
        debug (bool): If True, uses a small subset of data for debugging.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Select Metadata File
    if split == "train":
        meta_path = Config.TRAIN_METADATA
    elif split == "val":
        meta_path = Config.VAL_METADATA
    elif split == "test":
        meta_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Invalid split: {split}")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Debugging Subset
    if debug:
        df = df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Compute/Load ROI Indices (Signal-Fidelity Pipeline)
    # This ensures we are looking at the tumor region
    roi_map = get_roi_indices(df, load_cached_data=load_cached_data)

    # Define Augmentations
    # Train: Rotation (+/- 15 deg), HFlip, VFlip
    # Val/Test: None (Preprocessing handles resizing)
    if split == "train":
        transform = A.Compose(
            [
                A.Rotate(limit=Config.ROTATION_DEGREES, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )
    else:
        transform = None

    # Instantiate Dataset
    dataset = MGMTDataset(df, roi_map, transform=transform)

    # Create DataLoader
    # Shuffle only for training
    shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(
            split == "train"
        ),  # Drop incomplete batch in training to maintain stats
    )

    return loader
