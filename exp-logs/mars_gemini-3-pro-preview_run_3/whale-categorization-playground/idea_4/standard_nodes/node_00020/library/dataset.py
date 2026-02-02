import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from library.config import Config


class WhaleDataset(Dataset):
    """
    Custom Dataset for Whale Identification.
    Handles image loading, preprocessing, and target formatting.
    """

    def __init__(
        self,
        df,
        transforms=None,
        root_dir=Config.input_dir,
        split="train",
        label_encoder=None,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'Image', 'Id', and 'file_path'.
            transforms (albumentations.Compose): Transformations to apply.
            root_dir (str): Root directory containing image folders.
            split (str): 'train', 'val', 'gallery', or 'test'.
            label_encoder (LabelEncoder): Fitted sklearn LabelEncoder for mapping string Ids to ints.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root_dir = root_dir
        self.split = split
        self.label_encoder = label_encoder

        # Pre-check file existence to avoid runtime errors
        # (Optional optimization: could be skipped if metadata is trusted)
        self.file_paths = [
            os.path.join(self.root_dir, fp) for fp in self.df["file_path"].values
        ]

        # Prepare targets if not test
        if self.split != "test":
            self.ids = self.df["Id"].values
            if self.label_encoder is None:
                raise ValueError(
                    "LabelEncoder must be provided for train/val/gallery splits."
                )
            # Transform labels to integers
            # Note: We assume the df has already been filtered to only contain known classes
            # or classes present in the encoder.
            try:
                self.targets = self.label_encoder.transform(self.ids)
            except ValueError as e:
                # Fallback for debugging or if validation set has unseen classes (should be filtered out)
                print(f"Warning: Unseen labels found in {split} set. Mapping to -1.")
                # Create a map for safe transformation
                valid_classes = set(self.label_encoder.classes_)
                self.targets = np.array(
                    [
                        (
                            self.label_encoder.transform([x])[0]
                            if x in valid_classes
                            else -1
                        )
                        for x in self.ids
                    ]
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for corrupt images (create black image)
            image = np.zeros((Config.input_size, Config.input_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on split
        if self.split == "test":
            # Return image and filename for submission
            image_name = self.df.iloc[idx]["Image"]
            return image, image_name
        else:
            # Return image and integer label
            label = self.targets[idx]
            return image, label


def get_transforms(split="train"):
    """
    Returns Albumentations transforms for the specific split.
    Strategy: Geometric augmentations only for training.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose(
            [
                A.Resize(Config.input_size, Config.input_size),
                # Geometric Augmentations (No occlusion)
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=20,
                    p=0.7,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Affine(shear=(-10, 10), p=0.5, mode=cv2.BORDER_CONSTANT, cval=0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Gallery / Test
        return A.Compose(
            [
                A.Resize(Config.input_size, Config.input_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    debug=False,
    load_cached_data=True,
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
):
    """
    Prepares DataLoaders for Train, Gallery, Validation, and Test.

    Args:
        debug (bool): If True, subsamples data for quick testing.
        load_cached_data (bool): If True, attempts to load cached LabelEncoder.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.

    Returns:
        train_loader, gallery_loader, val_loader, test_loader, encoder
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.train_csv_path)
    val_df = pd.read_csv(Config.val_csv_path)
    test_df = pd.read_csv(Config.test_csv_path)

    # 2. Filter 'new_whale' if configured
    # Strategy: Train only on known identities.
    if Config.exclude_new_whale:
        train_df = train_df[train_df["Id"] != "new_whale"].copy()
        # Also filter from Val for Metric Learning evaluation (MAP@5 on knowns)
        val_df = val_df[val_df["Id"] != "new_whale"].copy()

    # Debug Mode: Subsample
    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]

    # 3. Label Encoding with Caching
    encoder_cache_path = os.path.join(Config.working_dir, "label_encoder_classes.npy")
    encoder = LabelEncoder()

    loaded_cache = False
    if load_cached_data and os.path.exists(encoder_cache_path):
        try:
            classes = np.load(encoder_cache_path, allow_pickle=True)
            encoder.classes_ = classes
            loaded_cache = True
            # print(f"Loaded LabelEncoder from {encoder_cache_path}")
        except Exception as e:
            print(f"Failed to load label encoder cache: {e}")

    if not loaded_cache:
        # Fit on training data
        encoder.fit(train_df["Id"])
        # Save cache
        os.makedirs(os.path.dirname(encoder_cache_path), exist_ok=True)
        np.save(encoder_cache_path, encoder.classes_)
        # print(f"Fitted LabelEncoder and saved to {encoder_cache_path}")

    # 4. Create Datasets
    # Train: Augmented
    train_dataset = WhaleDataset(
        train_df,
        transforms=get_transforms("train"),
        split="train",
        label_encoder=encoder,
    )

    # Gallery: Training data, but NO augmentation (Deterministic)
    # Used for building the retrieval database
    gallery_dataset = WhaleDataset(
        train_df,
        transforms=get_transforms("gallery"),
        split="gallery",
        label_encoder=encoder,
    )

    # Val: Query set, NO augmentation
    val_dataset = WhaleDataset(
        val_df, transforms=get_transforms("val"), split="val", label_encoder=encoder
    )

    # Test: NO augmentation, returns image names
    test_dataset = WhaleDataset(
        test_df, transforms=get_transforms("test"), split="test", label_encoder=None
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,  # Order matters for indexing
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, gallery_loader, val_loader, test_loader, encoder
