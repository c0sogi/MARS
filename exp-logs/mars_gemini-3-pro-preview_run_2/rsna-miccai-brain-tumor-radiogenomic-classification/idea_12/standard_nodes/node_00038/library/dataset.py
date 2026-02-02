import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import provided utility functions
from library.utils import read_dicom_robust, generate_roi_cache, seed_everything


class MGMTDataset(Dataset):
    def __init__(self, df, root_dir, roi_cache, phase="train", transform=None):
        """
        Dataset for GLP prediction with Stochastic-Stride Stacking.

        Args:
            df (pd.DataFrame): Metadata dataframe containing BraTS21ID and paths.
            root_dir (str): Root directory of the dataset.
            roi_cache (dict): Dictionary mapping BraTS21ID to the optimal anchor slice index.
            phase (str): 'train', 'valid', or 'test'. Controls stride strategy.
            transform (A.Compose): Albumentations transform pipeline.
        """
        self.df = df
        self.root_dir = root_dir
        self.roi_cache = roi_cache
        self.phase = phase
        self.transform = transform
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = row["BraTS21ID"]

        # Retrieve anchor index from cache, default to 0 if missing (fallback)
        anchor_idx = self.roi_cache.get(brats_id, 0)

        # Fixed Stride Selection
        # Using a fixed stride ensures geometric consistency for the 2.5D model (Cite solution_lesson_node_00037)
        stride = 5

        # Define relative offsets: [Previous, Anchor, Next]
        offsets = [-stride, 0, stride]

        channels = []

        # Iterate over modalities in fixed order: FLAIR -> T1w -> T1wCE -> T2w
        for mod in self.modalities:
            # Get relative path from dataframe (e.g., "train/00000/FLAIR")
            rel_path = row.get(f"path_{mod}")
            if not rel_path:
                # Fallback if path column missing
                rel_path = os.path.join(self.phase, f"{brats_id:05d}", mod)

            mod_dir = os.path.join(self.root_dir, rel_path)

            for off in offsets:
                # Calculate target slice index
                slice_idx = anchor_idx + off

                # Construct filename assuming "Image-{i}.dcm" format (1-based index)
                # If the file does not exist (e.g., index < 1 or > num_slices),
                # read_dicom_robust will return a zero-filled array.
                file_num = slice_idx + 1
                dcm_path = os.path.join(mod_dir, f"Image-{file_num}.dcm")

                # Load image: returns float32 array (224, 224)
                img = read_dicom_robust(dcm_path, size=(224, 224))

                # Instance-wise Min-Max Normalization to [0, 1]
                # Essential for float32 inputs to neural networks
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)

                channels.append(img)

        # Stack all channels: (H, W, 12)
        # 4 modalities * 3 slices = 12 channels
        image = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Becomes Tensor (12, H, W) via ToTensorV2
        else:
            # Manual conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Return image and target (if available)
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32).unsqueeze(0)
            return image, target
        else:
            return image


def get_transforms(phase):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Limit rotation to +/- 10 degrees to respect anatomical orientation
                A.Rotate(limit=10, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloader(
    metadata_df,
    phase,
    batch_size=32,
    num_workers=4,
    input_root="./input",
    cache_dir="./working/idea_12",
):
    """
    Factory function to create a DataLoader with ROI caching and reproducibility.

    Args:
        metadata_df (pd.DataFrame): Dataframe for the split.
        phase (str): 'train', 'valid', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.
        input_root (str): Path to input data.
        cache_dir (str): Directory to store ROI cache.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Ensure reproducibility
    seed_everything(42)

    # Generate or load ROI cache
    # Note: This computes ROIs for the subjects in metadata_df and saves to cache_dir
    roi_cache = generate_roi_cache(
        metadata_df, load_cached_data=True, cache_dir=cache_dir, input_root=input_root
    )

    # Get transforms
    transform = get_transforms(phase)

    # Initialize Dataset
    dataset = MGMTDataset(
        df=metadata_df,
        root_dir=input_root,
        roi_cache=roi_cache,
        phase=phase,
        transform=transform,
    )

    # Initialize DataLoader
    # Shuffle only for training
    shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == "train"),  # Drop incomplete batches during training
    )

    return loader
