import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(policy_name, image_size=Config.IMAGE_SIZE):
    """
    Creates an Albumentations transform pipeline based on the specified policy.

    Args:
        policy_name (str): The name of the augmentation policy (e.g., 'Texture', 'Feature', 'Balanced', 'Validation').
        image_size (tuple): Target (height, width) for resizing.

    Returns:
        A.Compose: The composition of transforms.
    """
    transforms = []

    # 1. Resize (Always applied)
    # Ensure strictly 256 (H) x 640 (W)
    transforms.append(A.Resize(height=image_size[0], width=image_size[1]))

    # 2. Augmentations (Only for Training Policies)
    if policy_name in Config.STRATIFIED_POLICIES:
        policy = Config.STRATIFIED_POLICIES[policy_name]

        # Horizontal Flip (Time Inversion)
        transforms.append(A.HorizontalFlip(p=0.5))

        # CoarseDropout (Cutout)
        # Retrieve params from config
        cutout_params = policy.get("cutout_params", {})
        cutout_prob = policy.get("cutout_prob", 0.0)

        transforms.append(
            A.CoarseDropout(
                max_holes=cutout_params.get("max_holes", 1),
                max_height=cutout_params.get("max_height", 10),
                max_width=cutout_params.get("max_width", 10),
                min_holes=1,
                min_height=1,
                min_width=1,
                fill_value=0,  # Fill with black
                p=cutout_prob,
            )
        )

    # 3. Normalization (Always applied)
    # ImageNet Mean and Std
    transforms.append(
        A.Normalize(mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0)
    )

    # 4. Convert to Tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles dynamic loading of spectrograms, channel replication, and label retrieval.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'rec_id', 'file_path', and label columns.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Used for logging or specific logic if needed.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Identify label columns
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Pre-calculate paths to avoid string ops in __getitem__
        self.image_paths = []
        for rel_path in self.df["file_path"]:
            # Convert WAV path to Spectrogram BMP path
            # Input: essential_data/src_wavs/filename.wav
            # Output: full/path/to/supplemental_data/spectrograms/filename.bmp
            basename = os.path.basename(rel_path)
            filename_no_ext = os.path.splitext(basename)[0]
            bmp_filename = f"{filename_no_ext}.bmp"
            full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)
            self.image_paths.append(full_path)

        # Extract labels as float32 for BCEWithLogitsLoss
        # If test set (hidden labels), these will be 0s, which is fine as we don't calculate loss on test
        if all(col in self.df.columns for col in self.label_cols):
            self.labels = self.df[self.label_cols].values.astype(np.float32)
        else:
            # Fallback if columns missing (shouldn't happen with provided metadata)
            self.labels = np.zeros((len(self.df), Config.NUM_CLASSES), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]

        # Load as grayscale (single channel spectrogram)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images (robustness)
            # Create a black image of expected size if file read fails
            # Note: EDA showed no missing files, but good practice.
            # Assuming roughly 256x640 based on EDA, but resize transform handles exacts.
            image = np.zeros((256, 640), dtype=np.uint8)

        # 2. Channel Replication
        # Copy the single-channel spectrogram to R, G, and B
        image = cv2.merge([image, image, image])

        # 3. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 4. Get Label
        label = self.labels[idx]

        return image, label


def get_dataloaders(
    train_df,
    val_df,
    test_df,
    pseudo_labels_df=None,
    teacher_policy=None,
    student_policy=Config.STUDENT_POLICY_NAME,
    batch_size=Config.BATCH_SIZE,
    workers=Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        train_df (pd.DataFrame): Labeled training data (Fold 0).
        val_df (pd.DataFrame): Validation data (Fold 0).
        test_df (pd.DataFrame): Test data (Fold 1).
        pseudo_labels_df (pd.DataFrame, optional): DataFrame containing pseudo-labels for the test set.
                                                   If provided, creates a combined Student dataset.
        teacher_policy (str, optional): Augmentation policy for Teacher training.
        student_policy (str, optional): Augmentation policy for Student training.
        batch_size (int): Batch size.
        workers (int): Number of worker processes.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """

    # --- Debug Mode ---
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        test_df = test_df.head(Config.DEBUG_SAMPLES)
        if pseudo_labels_df is not None:
            pseudo_labels_df = pseudo_labels_df.head(Config.DEBUG_SAMPLES)

    # --- Determine Training Configuration ---

    if pseudo_labels_df is not None:
        # === Student Mode ===
        # We merge the pseudo-labels into the test dataframe and concatenate with train

        # Ensure pseudo_labels_df has the same columns as expected
        # We assume pseudo_labels_df has 'rec_id' and 'species_0'...'species_18'

        # 1. Prepare Pseudo-labeled Test Set
        # We take the file paths from test_df, but labels from pseudo_labels_df
        # Merge on rec_id
        label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Drop existing zero-labels from test_df to avoid collision
        cols_to_use = [c for c in test_df.columns if c not in label_cols]
        test_metadata_only = test_df[cols_to_use]

        # Merge
        pseudo_labeled_test_df = pd.merge(
            test_metadata_only, pseudo_labels_df, on="rec_id", how="inner"
        )

        # 2. Concatenate with Ground Truth Train Set
        # Ensure column order matches
        combined_train_df = pd.concat(
            [train_df, pseudo_labeled_test_df], axis=0, ignore_index=True
        )

        # 3. Select Policy
        train_policy_name = student_policy
        final_train_df = combined_train_df

    else:
        # === Teacher Mode ===
        if teacher_policy is None:
            raise ValueError(
                "teacher_policy must be specified when pseudo_labels_df is None"
            )

        train_policy_name = teacher_policy
        final_train_df = train_df

    # --- Create Transforms ---
    train_transforms = get_transforms(train_policy_name)
    val_transforms = get_transforms("Validation")  # Validation/Test just Resize+Norm

    # --- Create Datasets ---
    train_dataset = BirdDataset(
        final_train_df, transforms=train_transforms, mode="train"
    )
    val_dataset = BirdDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = BirdDataset(test_df, transforms=val_transforms, mode="test")

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,  # Useful for Batch Normalization stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
