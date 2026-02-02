import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# Import configuration and utility functions
from library.config import Config
from library.utils import process_hierarchy_mappings


def get_hierarchy_map(json_path, cache_dir, load_cached_data=True):
    """
    Retrieves the taxonomic hierarchy mapping (Category -> Genus -> Family).
    Delegates to the utility function which handles caching logic.

    Args:
        json_path (str): Path to the train_metadata.json file.
        cache_dir (str): Directory to store/load the cached parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing category_id, genus_id, and family_id.
    """
    return process_hierarchy_mappings(json_path, cache_dir, load_cached_data)


def get_transforms(mode="train"):
    """
    Returns the Albumentations transforms for the specified mode.

    Args:
        mode (str): 'train' or 'val'.

    Returns:
        A.Compose: Composed transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                # Strong Data Augmentation
                A.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE), scale=(0.8, 1.0)
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Standard Resize and Normalize
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class PlantDataset(Dataset):
    def __init__(self, df, hierarchy_map=None, transforms=None, mode="train"):
        """
        Custom Dataset for Plant Classification.

        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            hierarchy_map (pd.DataFrame): DataFrame mapping category_id to genus_id and family_id.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Pre-process hierarchy lookup for fast access during training/validation
        self.genus_lookup = None
        self.family_lookup = None

        if self.mode in ["train", "val"] and hierarchy_map is not None:
            # Create numpy arrays for O(1) lookup of hierarchy labels
            # Assuming category_id is integer and can be used as index
            max_cat_id = hierarchy_map["category_id"].max()

            # Initialize with -100 (ignore index for CrossEntropyLoss)
            self.genus_lookup = np.full(max_cat_id + 1, -100, dtype=np.int64)
            self.family_lookup = np.full(max_cat_id + 1, -100, dtype=np.int64)

            # Fill lookup arrays
            cats = hierarchy_map["category_id"].values
            genera = hierarchy_map["genus_id"].values
            families = hierarchy_map["family_id"].values

            self.genus_lookup[cats] = genera
            self.family_lookup[cats] = families

            # Ensure any -1 values (e.g. from cached metadata) are converted to -100
            self.genus_lookup[self.genus_lookup == -1] = -100
            self.family_lookup[self.family_lookup == -1] = -100

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt/missing images to prevent crash
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode in ["train", "val"]:
            # Retrieve labels
            species_id = int(row["category_id"])

            # Lookup hierarchical labels
            if self.genus_lookup is not None:
                genus_id = self.genus_lookup[species_id]
                family_id = self.family_lookup[species_id]
            else:
                genus_id = -100
                family_id = -100

            return (
                image,
                torch.tensor(species_id, dtype=torch.long),
                torch.tensor(genus_id, dtype=torch.long),
                torch.tensor(family_id, dtype=torch.long),
            )
        else:
            # Test mode: Return image and image_id for submission generation
            image_id = row["image_id"]
            return image, str(image_id)


def get_loaders(load_cached_data=True):
    """
    Constructs and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to load hierarchy mapping from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Load Hierarchy Mapping
    hierarchy_df = get_hierarchy_map(
        Config.TRAIN_METADATA_JSON,
        Config.WORKING_DIR,
        load_cached_data=load_cached_data,
    )

    # 3. Apply Debug Sampling
    if Config.DEBUG:
        train_df = train_df.sample(
            n=min(len(train_df), 2000), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), 500), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), 500), random_state=Config.SEED
        ).reset_index(drop=True)

    # 4. Prepare Transforms
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # 5. Instantiate Datasets
    train_dataset = PlantDataset(
        train_df, hierarchy_map=hierarchy_df, transforms=train_transforms, mode="train"
    )

    val_dataset = PlantDataset(
        val_df, hierarchy_map=hierarchy_df, transforms=val_transforms, mode="val"
    )

    test_dataset = PlantDataset(
        test_df, hierarchy_map=None, transforms=val_transforms, mode="test"
    )

    # 6. Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
