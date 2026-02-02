import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.utils import seed_everything
from library.data_processing import LungDataProcessor


def get_transforms(mode="train", img_size=224):
    """
    Returns the Albumentations transform pipeline.
    Strictly spatial augmentations for training; normalization for all.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return A.Compose(
            [
                # Spatial Augmentations Only
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Ensure size is correct (safety)
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class LungDataset(Dataset):
    def __init__(self, df, tabular_features, processor, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing Patient, Weeks, etc.
            tabular_features (np.array): Pre-processed tabular features matching df rows.
            processor (LungDataProcessor): Instance to handle image loading/caching.
            transforms (albumentations.Compose): Transforms to apply to images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.tabular_features = tabular_features
        self.processor = processor
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]
        dicom_dir = row["dicom_dir"]

        # 1. Load Images (Axial and Coronal)
        # The processor handles caching logic internally.
        # It returns (224, 224, 3) numpy arrays in uint8 [0-255]
        img_ax, img_cor = self.processor.process_images(
            patient_id, dicom_dir, load_cached_data=True
        )

        # 2. Apply Transforms
        # We apply transforms independently to axial and coronal views
        # as they are orthogonal and spatial consistency isn't strictly 1:1 in pixel space.
        if self.transforms:
            res_ax = self.transforms(image=img_ax)
            img_ax_tensor = res_ax["image"]

            res_cor = self.transforms(image=img_cor)
            img_cor_tensor = res_cor["image"]
        else:
            # Fallback to simple ToTensor if no transforms provided
            img_ax_tensor = torch.from_numpy(img_ax.transpose(2, 0, 1)).float() / 255.0
            img_cor_tensor = (
                torch.from_numpy(img_cor.transpose(2, 0, 1)).float() / 255.0
            )

        # 3. Tabular Features
        tab_vec = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        # 4. Meta Inputs for Residual Anchor
        # Model needs: Baseline_FVC and Weeks_From_Baseline (time delta)
        baseline_fvc = float(row["Baseline_FVC"])
        weeks_from_baseline = float(row["Weeks_From_Baseline"])

        meta_tensor = torch.tensor(
            [baseline_fvc, weeks_from_baseline], dtype=torch.float32
        )

        data = {
            "img_ax": img_ax_tensor,
            "img_cor": img_cor_tensor,
            "tab": tab_vec,
            "meta": meta_tensor,
            "patient_week": str(
                row.get("Patient_Week", "")
            ),  # Useful for submission mapping
        }

        # 5. Target (if available)
        if self.mode in ["train", "val"]:
            target = float(row["FVC"])
            data["target"] = torch.tensor(target, dtype=torch.float32)

        return data


def get_dataloaders(
    metadata_dir="./metadata",
    cache_dir="./working/idea_13/",
    batch_size=32,
    num_workers=4,
    img_size=224,
):
    """
    Factory function to prepare datasets and dataloaders.
    """
    seed_everything(42)

    # 1. Load Metadata
    try:
        train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        raise

    # 2. Initialize Processor
    processor = LungDataProcessor(cache_dir=cache_dir)

    # 3. Prepare Tabular Features
    # This adds 'Weeks_From_Baseline' to dfs and returns normalized feature matrices
    train_proc, train_feats, val_proc, val_feats, test_proc, test_feats = (
        processor.prepare_tabular_features(train_df, val_df, test_df)
    )

    # 4. Create Datasets
    train_ds = LungDataset(
        train_proc,
        train_feats,
        processor,
        transforms=get_transforms("train", img_size),
        mode="train",
    )

    val_ds = LungDataset(
        val_proc,
        val_feats,
        processor,
        transforms=get_transforms("val", img_size),
        mode="val",
    )

    test_ds = LungDataset(
        test_proc,
        test_feats,
        processor,
        transforms=get_transforms("test", img_size),
        mode="test",
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
