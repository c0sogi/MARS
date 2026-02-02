import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_processed_slice, generate_slice_cache


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the BraTS21 Glioblastoma classification task.
    Implements a Deterministic Strided Sampling strategy for a 2.5D Siamese Network.

    For each subject, it retrieves slices at specific depths (45%, 50%, 55%) for
    three modalities (FLAIR, T1wCE, T2w), constructing a (3, 3, H, W) input tensor.
    """

    def __init__(self, split="train", load_cached_data=True, debug_limit=None):
        """
        Args:
            split (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load file paths from cache or recompute.
            debug_limit (int, optional): Limit the dataset size for debugging purposes.
        """
        self.split = split
        self.img_size = Config.IMG_SIZE
        self.depths = Config.SLICE_DEPTHS  # [0.45, 0.50, 0.55]
        self.modalities = Config.SELECTED_MODALITIES  # ["FLAIR", "T1wCE", "T2w"]

        # 1. Load Metadata based on split
        if split == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        df_metadata = pd.read_csv(metadata_path)

        # 2. Generate or Load Cached Slice Paths
        # This returns a DataFrame with columns like 'FLAIR_0.45', 'T1wCE_0.5', etc.
        self.df = generate_slice_cache(
            metadata_df=df_metadata, split_name=split, load_cached_data=load_cached_data
        )

        # 3. Apply Debug Limit if requested
        if debug_limit is not None and debug_limit > 0:
            self.df = self.df.iloc[:debug_limit].reset_index(drop=True)

        # 4. Define Augmentations
        # Applied independently to each of the 3 views during training
        if self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.OneOf(
                        [
                            A.ElasticTransform(
                                alpha=120,
                                sigma=120 * 0.05,
                                alpha_affine=120 * 0.03,
                                p=0.5,
                            ),
                            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                        ],
                        p=0.6,
                    ),
                    # RandomBrightnessContrast is safe as we normalized to [0,1] but
                    # albumentations handles float32 images correctly
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        views = []

        # Iterate through the 3 depths: 0.45, 0.50, 0.55
        # Each depth corresponds to one "view" or input branch of the Siamese network
        for depth in self.depths:
            channels = []

            # Construct the 3-channel image for this depth
            for mod in self.modalities:
                # Column name in cached dataframe, e.g., "FLAIR_0.45"
                col_name = f"{mod}_{depth}"
                file_path = row[col_name]

                # Load, resize, and normalize (min-max) the slice
                # Returns (H, W) float32 array in [0, 1]
                img_slice = load_processed_slice(file_path, target_size=self.img_size)
                channels.append(img_slice)

            # Stack modalities to create (H, W, 3) image
            # Channel order: FLAIR, T1wCE, T2w
            composite_img = np.stack(channels, axis=-1)

            # Apply Augmentations (only for training)
            if self.transform:
                augmented = self.transform(image=composite_img)
                composite_img = augmented["image"]

            # Transpose to PyTorch format (C, H, W) -> (3, H, W)
            composite_img = composite_img.transpose(2, 0, 1)
            views.append(composite_img)

        # Stack the 3 views to create the final input tensor
        # Shape: (3, 3, H, W) -> (Num_Views, Channels, Height, Width)
        input_tensor = np.stack(views, axis=0)
        input_tensor = torch.from_numpy(input_tensor).float()

        # Get Target Label
        if "MGMT_value" in row and not np.isnan(row["MGMT_value"]):
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            # For test set where label might not exist
            target = torch.tensor(-1.0, dtype=torch.float32)

        return input_tensor, target
