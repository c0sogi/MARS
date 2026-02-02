import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import get_hierarchy_dicts


class PlantDataset(Dataset):
    """
    PyTorch Dataset for Plant Classification.
    Handles image loading and hierarchical target generation.
    """

    def __init__(self, df, transforms=None, hierarchy_dicts=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'category_id' (if not test).
            transforms (A.Compose): Albumentations transforms to apply.
            hierarchy_dicts (tuple): (species_to_genus, species_to_family) mappings. Required if is_test=False.
            is_test (bool): If True, returns (image, image_id). If False, returns (image, target_dict).
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.is_test = is_test

        if not self.is_test:
            if hierarchy_dicts is None:
                raise ValueError(
                    "hierarchy_dicts must be provided for training/validation"
                )
            self.species_to_genus, self.species_to_family = hierarchy_dicts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in metadata is relative to input directory (e.g., "train_images/...")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt images (though metadata generation checked existence)
            # Create a black image to prevent crashing
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Basic to tensor if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.is_test:
            # For test set, return image and ID for submission
            return image, str(row["image_id"])
        else:
            # For train/val, return image and hierarchical targets
            species_id = int(row["category_id"])

            # Map to parent taxa
            genus_id = self.species_to_genus.get(species_id, -1)
            family_id = self.species_to_family.get(species_id, -1)

            targets = {
                "species": torch.tensor(species_id, dtype=torch.long),
                "genus": torch.tensor(genus_id, dtype=torch.long),
                "family": torch.tensor(family_id, dtype=torch.long),
            }
            return image, targets


def get_transforms(split, image_size):
    """
    Returns Albumentations transforms for the specified split and image size.

    Args:
        split (str): 'train' or 'val'/'test'.
        image_size (int): Target image resolution (e.g., 224 or 320).
    """
    if split == "train":
        return A.Compose(
            [
                # Strong Data Augmentation
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.08, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Deterministic preprocessing for validation/test
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(stage=1, debug=False):
    """
    Creates DataLoaders for training and validation based on the training stage.

    Args:
        stage (int): 1 or 2. Determines image size and batch size from Config.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader
    """
    # Determine parameters based on stage
    if stage == 1:
        image_size = Config.STAGE1_IMAGE_SIZE
        batch_size = Config.STAGE1_BATCH_SIZE
    elif stage == 2:
        image_size = Config.STAGE2_IMAGE_SIZE
        batch_size = Config.STAGE2_BATCH_SIZE
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # Debug Sampling
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        print(
            f"DEBUG MODE: Sampled {len(df_train)} train and {len(df_val)} val images."
        )

    # Get Hierarchy Mappings
    hierarchy_dicts = get_hierarchy_dicts(load_cached_data=True)

    # Create Datasets
    train_dataset = PlantDataset(
        df=df_train,
        transforms=get_transforms("train", image_size),
        hierarchy_dicts=hierarchy_dicts,
        is_test=False,
    )

    val_dataset = PlantDataset(
        df=df_val,
        transforms=get_transforms("val", image_size),
        hierarchy_dicts=hierarchy_dicts,
        is_test=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(image_size, batch_size):
    """
    Creates DataLoader for the test set.

    Args:
        image_size (int): Resolution for inference.
        batch_size (int): Batch size for inference.

    Returns:
        test_loader
    """
    df_test = pd.read_csv(Config.TEST_CSV)

    test_dataset = PlantDataset(
        df=df_test,
        transforms=get_transforms("test", image_size),
        hierarchy_dicts=None,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
