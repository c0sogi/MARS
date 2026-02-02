import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(data, img_size):
    """
    Returns the transformation pipeline for training or validation/testing.

    Strategy:
    - Train: Strong Geometric Augmentation (ShiftScaleRotate, RandomFlip, Transpose).
      Strictly excludes Spatial Occlusion (Cutout) and Photometric Augmentation.
    - Valid/Test: Resize and Normalize only.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Strong Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(
                    p=0.5
                ),  # RandomFlip implies both unless restricted. Strategy says "RandomFlip".
                # TTA restricts Vertical, but training allows it implies general robustness.
                # However, strategy text says "leaves have gravity priors" in TTA section.
                # To be safe and consistent with "RandomFlip" usually meaning H+V or just H,
                # and the TTA restriction, we will include V-flip in training to force invariance
                # if possible, or stick to H-flip if gravity is strict.
                # Given "Strong Geometric Augmentation", we include V-flip here.
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7
                ),
                A.Transpose(p=0.5),
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    Dataset class for Apple Disease Detection.
    Handles image loading and target generation for Multi-Task Learning.
    """

    def __init__(self, df, transform=None, output_aux_targets=True, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            transform (albumentations.Compose): Augmentation pipeline.
            output_aux_targets (bool): Whether to output auxiliary targets for MTL.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = df
        self.transform = transform
        self.output_aux_targets = output_aux_targets
        self.is_test = is_test

        # Pre-compute paths to avoid joining strings in the loop
        # Metadata file_path is relative: "images/Train_0.jpg"
        # Config.INPUT_DIR is "./input"
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        if not self.is_test:
            # Extract labels
            self.labels = df[Config.CLASS_LABELS].values.astype(np.float32)

            # Map column indices for auxiliary task generation
            # Config.CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
            self.idx_healthy = Config.CLASS_LABELS.index("healthy")
            self.idx_multiple = Config.CLASS_LABELS.index("multiple_diseases")
            self.idx_rust = Config.CLASS_LABELS.index("rust")
            self.idx_scab = Config.CLASS_LABELS.index("scab")
        else:
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing images (should not happen given metadata validation)
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return for Test Set
        if self.is_test:
            return image, self.image_ids[idx]

        # Generate Targets for Training/Validation
        main_target = self.labels[idx]  # One-hot vector

        targets = {"main": torch.tensor(main_target, dtype=torch.float32)}

        if self.output_aux_targets:
            # Decoupled Multi-Task Targets
            # Rust: True if Rust OR Multiple
            has_rust = max(main_target[self.idx_rust], main_target[self.idx_multiple])

            # Scab: True if Scab OR Multiple
            has_scab = max(main_target[self.idx_scab], main_target[self.idx_multiple])

            # Healthy: True if Healthy
            is_healthy = main_target[self.idx_healthy]

            targets["aux_rust"] = torch.tensor([has_rust], dtype=torch.float32)
            targets["aux_scab"] = torch.tensor([has_scab], dtype=torch.float32)
            targets["aux_healthy"] = torch.tensor([is_healthy], dtype=torch.float32)

        return image, targets


def get_class_weights(load_cached_data=True):
    """
    Calculates Inverse Frequency Class Weights for the main loss.
    Implements caching mechanism.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "class_weights.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached class weights from {cache_path}")
        return np.load(cache_path)

    print("Computing class weights from scratch...")
    # Load train metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Get labels
    labels = train_df[Config.CLASS_LABELS].values
    # Convert one-hot to class indices for counting
    class_indices = np.argmax(labels, axis=1)

    classes, counts = np.unique(class_indices, return_counts=True)
    num_classes = len(Config.CLASS_LABELS)
    total_samples = len(train_df)

    # Calculate weights: N / (C * count_c)
    # Initialize with 1.0
    weights = np.ones(num_classes, dtype=np.float32)

    for cls_idx, count in zip(classes, counts):
        weights[cls_idx] = total_samples / (num_classes * count)

    print(f"Computed Class Weights: {weights}")

    # Save to cache
    np.save(cache_path, weights)

    return weights


def load_data(debug=False):
    """
    Loads train, validation, and test dataframes.
    Handles debug mode by subsampling.
    """
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        print("DEBUG MODE: Subsampling datasets.")
        train_df = train_df.sample(
            n=min(100, len(train_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(50, len(val_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        # Keep test full or subsample? Usually keep test full to ensure pipeline works,
        # but for speed debug we can subsample.
        test_df = test_df.sample(
            n=min(20, len(test_df)), random_state=Config.SEED
        ).reset_index(drop=True)

    return train_df, val_df, test_df


def get_loaders(train_df, val_df, img_size, batch_size, num_workers):
    """
    Creates DataLoaders for training and validation.
    """
    train_dataset = AppleDataset(
        train_df,
        transform=get_transforms("train", img_size),
        output_aux_targets=True,
        is_test=False,
    )

    val_dataset = AppleDataset(
        val_df,
        transform=get_transforms("valid", img_size),
        output_aux_targets=True,  # We validate on main metric, but model might output aux
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(test_df, img_size, batch_size, num_workers):
    """
    Creates DataLoader for inference (test set).
    """
    test_dataset = AppleDataset(
        test_df,
        transform=get_transforms("test", img_size),
        output_aux_targets=False,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
