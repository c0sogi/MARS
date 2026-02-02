import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.data_processing import load_volumetric_slab


class BraTSDataset(Dataset):
    """
    Dataset class for Volumetric Glioblastoma Classification.
    Constructs a 9-channel input tensor based on Centroid-Aligned Weight-Inflated Volumetric strategy.
    """

    def __init__(
        self, df, centroids_df, input_dir=Config.INPUT_DIR, transform=None, mode="train"
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing BraTS21ID and paths.
            centroids_df (pd.DataFrame): Dataframe containing computed centroids for each subject.
            input_dir (str): Path to the input directory.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        # Index centroids by BraTS21ID for O(1) lookup
        if "BraTS21ID" in centroids_df.columns:
            self.centroids_df = centroids_df.set_index("BraTS21ID")
        else:
            self.centroids_df = centroids_df

        self.input_dir = input_dir
        self.transform = transform
        self.mode = mode
        self.modalities = ["flair", "t1wce", "t2w"]
        self.stride = Config.STRIDE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]

        # Retrieve centroids for this subject
        # Handle case where subject might be missing from centroids cache (fallback to 0/middle handled later)
        if sid in self.centroids_df.index:
            subject_centroids = self.centroids_df.loc[sid]
        else:
            # Fallback dictionary if ID not found
            subject_centroids = {f"{m}_centroid": 0 for m in self.modalities}

        # Load slabs for each modality
        # load_volumetric_slab returns (H, W, 3) corresponding to [z-delta, z, z+delta]
        slabs = {}
        for mod in self.modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(self.input_dir, rel_path)

            # Glob files for this modality
            # Note: load_volumetric_slab handles sorting, we just need to provide paths
            if os.path.exists(full_path):
                file_paths = glob.glob(os.path.join(full_path, "*.dcm"))
            else:
                file_paths = []

            # Get centroid index
            centroid_idx = int(subject_centroids.get(f"{mod}_centroid", 0))
            if not file_paths and f"{mod}_centroid" not in subject_centroids:
                # If no files and no centroid info, logic inside load_volumetric_slab handles empty list
                pass
            elif file_paths and f"{mod}_centroid" not in subject_centroids:
                # Fallback if specific centroid missing but files exist
                centroid_idx = len(file_paths) // 2

            # Load the 3-slice slab
            # Returns (H, W, 3) -> [z-stride, z, z+stride]
            slabs[mod] = load_volumetric_slab(
                file_paths,
                centroid_idx,
                delta=self.stride,
                image_size=Config.IMAGE_SIZE,
            )

        # Construct 9-channel tensor
        # Requirement:
        # Ch 0-2: [FLAIR, T1wCE, T2w] at z-delta
        # Ch 3-5: [FLAIR, T1wCE, T2w] at z
        # Ch 6-8: [FLAIR, T1wCE, T2w] at z+delta

        channels = []

        # Iterate through depths: 0 (z-d), 1 (z), 2 (z+d)
        for depth_idx in range(3):
            for mod in self.modalities:
                # Extract the specific depth slice from the modality slab
                # slabs[mod] is (H, W, 3)
                channels.append(slabs[mod][:, :, depth_idx])

        # Stack to create (H, W, 9)
        img_stack = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img_stack)
            img_tensor = augmented["image"]  # (9, H, W) via ToTensorV2
        else:
            # Manual conversion if no transform provided
            img_tensor = torch.tensor(img_stack.transpose(2, 0, 1), dtype=torch.float32)

        # Return based on mode
        if self.mode == "test":
            return img_tensor, sid
        else:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return img_tensor, target


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    Strictly excludes Translation and Scaling to preserve Centroid Alignment.
    """
    if mode == "train":
        return A.Compose(
            [
                # Rotation is allowed (preserves center relative to image if crop is handled,
                # but here we resize first so center of image is center of brain)
                A.Rotate(limit=15, p=0.5),
                # Elastic and Grid distortions for shape variance
                A.GridDistortion(p=0.5),
                A.ElasticTransform(
                    p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                # Regularization
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5),
                # Convert to Tensor (H, W, C) -> (C, H, W)
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])
