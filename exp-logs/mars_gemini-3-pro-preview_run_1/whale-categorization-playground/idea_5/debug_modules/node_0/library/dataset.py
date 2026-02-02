import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Whale Species Prediction.
    """

    def __init__(self, df, root_dir, transform=None, label_encoder=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Directory containing images.
            transform (albumentations.Compose): Transformations to apply.
            label_encoder (dict): Dictionary mapping class IDs to integers.
            is_test (bool): If True, returns image_id instead of label.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = label_encoder
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["Image"]

        # Construct full path.
        # Note: metadata 'file_path' includes the folder (e.g., 'train/img.jpg'),
        # but Config.INPUT_DIR is './input'.
        # We can use the 'file_path' from metadata directly combined with INPUT_DIR.
        # Alternatively, rely on root_dir passed in.
        # The metadata 'file_path' is relative to INPUT_DIR.

        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a black image of expected size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            return image, image_name
        else:
            label_str = row["Id"]
            label = self.label_encoder[label_str]
            return image, torch.tensor(label, dtype=torch.long)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                # Conservative Affine: Rotation +/- 20, Scale 0.9-1.1
                A.ShiftScaleRotate(
                    shift_limit=0.0,  # No shifting
                    scale_limit=0.1,  # 0.9 to 1.1
                    rotate_limit=20,  # +/- 20 degrees
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Photometric Augmentations
                # Explicitly excluding Hue/Saturation as per requirements
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_label_encoder(train_df, load_cached_data=True):
    """
    Generates or loads the class-to-index mapping.

    Args:
        train_df (pd.DataFrame): Training metadata containing all classes.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        dict: mapping string Id -> int index
        np.array: array of class names where index corresponds to the int index
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        classes = np.load(cache_path)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return class_to_idx, classes

    # Compute from scratch
    # We must ensure 'new_whale' and all other IDs are included.
    # The train_df from metadata is guaranteed to have all classes (including singletons).
    unique_classes = sorted(train_df["Id"].unique().tolist())

    # Save to cache
    np.save(cache_path, np.array(unique_classes))

    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    return class_to_idx, np.array(unique_classes)


def get_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached label encoder.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Prepare Label Encoder
    class_to_idx, class_names = get_label_encoder(
        df_train, load_cached_data=load_cached_data
    )

    # Verify consistency
    if len(class_names) != Config.NUM_CLASSES:
        # If the generated classes differ from Config, we should warn or update.
        # Ideally, Config.NUM_CLASSES should match len(class_names).
        # For this implementation, we proceed with the data-derived length.
        pass

    # Create Datasets
    train_dataset = WhaleDataset(
        df=df_train,
        root_dir=Config.TRAIN_IMG_DIR,
        transform=get_transforms(phase="train"),
        label_encoder=class_to_idx,
        is_test=False,
    )

    val_dataset = WhaleDataset(
        df=df_val,
        root_dir=Config.TRAIN_IMG_DIR,  # Val images are in train folder
        transform=get_transforms(phase="val"),
        label_encoder=class_to_idx,
        is_test=False,
    )

    test_dataset = WhaleDataset(
        df=df_test,
        root_dir=Config.TEST_IMG_DIR,
        transform=get_transforms(phase="test"),
        label_encoder=None,
        is_test=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
