import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.data_processing import read_dicom


class BraTSDataset(Dataset):
    """
    Dataset class for Volumetric Glioblastoma Classification.
    Constructs a 3-channel input tensor using Centroid-Aligned Slice Selection.
    Cite solution_lesson_node_00009: Avoid naive channel stacking of depth.
    Cite solution_lesson_node_00015: Deterministic geometric heuristics.
    """

    def __init__(
        self, df, centroids_df, input_dir=Config.INPUT_DIR, transform=None, mode="train"
    ):
        self.df = df.reset_index(drop=True)
        if "BraTS21ID" in centroids_df.columns:
            self.centroids_df = centroids_df.set_index("BraTS21ID")
        else:
            self.centroids_df = centroids_df

        self.input_dir = input_dir
        self.transform = transform
        self.mode = mode
        self.modalities = ["flair", "t1wce", "t2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]

        if sid in self.centroids_df.index:
            subject_centroids = self.centroids_df.loc[sid]
        else:
            subject_centroids = {f"{m}_centroid": 0 for m in self.modalities}

        channels = []
        for mod in self.modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(self.input_dir, rel_path)

            if os.path.exists(full_path):
                # We need sorted files to index correctly
                files = sorted(
                    glob.glob(os.path.join(full_path, "*.dcm")),
                    key=lambda x: int(x.split("-")[-1].split(".")[0]),
                )
            else:
                files = []

            if not files:
                img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
            else:
                # Select centroid slice
                centroid_idx = int(
                    subject_centroids.get(f"{mod}_centroid", len(files) // 2)
                )
                centroid_idx = max(0, min(centroid_idx, len(files) - 1))

                img = read_dicom(files[centroid_idx], Config.IMAGE_SIZE)

                # Cite solution_lesson_node_00023: Channel-Independent Normalization
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)

            channels.append(img)

        # Stack to create (H, W, 3)
        img_stack = np.stack(channels, axis=-1)

        if self.transform:
            augmented = self.transform(image=img_stack)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.tensor(img_stack.transpose(2, 0, 1), dtype=torch.float32)

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
