import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.taxonomy import TaxonomyManager


class HerbariumDataset(Dataset):
    """
    PyTorch Dataset for the Herbarium 2021 competition.
    Handles loading images, applying transforms, and retrieving hierarchical labels.
    """

    def __init__(self, csv_path, mode="train", transform=None, taxonomy_map=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            taxonomy_map (pd.DataFrame): DataFrame mapping category_id to family_id and order_id.
        """
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Debugging: Subsample if configured
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE).copy()
            print(f"[DEBUG] Subsampled {self.mode} dataset to {len(self.df)} samples.")

        # Merge taxonomy info for train/val modes
        if self.mode in ["train", "val"]:
            if taxonomy_map is not None:
                # Merge taxonomy mapping on category_id
                # taxonomy_map columns: ['category_id', 'family_id', 'order_id']
                self.df = pd.merge(self.df, taxonomy_map, on="category_id", how="left")

                # Check for missing mappings
                if self.df["family_id"].isnull().any():
                    print(
                        f"Warning: Some categories in {mode} set missing taxonomy info."
                    )
                    self.df.fillna({"family_id": -1, "order_id": -1}, inplace=True)
            else:
                # If no map provided, fill with dummy values (though this shouldn't happen in this pipeline)
                self.df["family_id"] = -1
                self.df["order_id"] = -1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # metadata file_path is relative to input dir, e.g., "train/images/..."
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt images: return black image
            # In a real scenario, we might log this.
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            t = A.Compose(
                [
                    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = t(image=image)["image"]

        # Return based on mode
        if self.mode == "test":
            image_id = row["image_id"]
            return image, image_id
        else:
            # Targets
            species_id = torch.tensor(row["category_id"], dtype=torch.long)
            family_id = torch.tensor(int(row["family_id"]), dtype=torch.long)
            order_id = torch.tensor(int(row["order_id"]), dtype=torch.long)

            return image, species_id, family_id, order_id

    def get_labels(self):
        """Returns the list of category_ids for sampling weights calculation."""
        if "category_id" in self.df.columns:
            return self.df["category_id"].values
        return None


def get_transforms(mode="train", image_size=224):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.8, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        # Resize to slightly larger then center crop is standard,
        # but for efficiency and matching the idea description, we can do Resize or Resize+Crop.
        # Let's do Resize(256) -> CenterCrop(224) for standard evaluation protocol.
        crop_size = image_size
        resize_size = int(image_size * 256 / 224)
        return A.Compose(
            [
                A.Resize(height=resize_size, width=resize_size),
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(stage=1):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        stage (int): 1 for Instance-Balanced Sampling (Representation Learning).
                     2 for Class-Balanced Sampling (Classifier Re-balancing).

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        int: num_families
        int: num_orders
    """
    # 1. Prepare Taxonomy
    tax_manager = TaxonomyManager()
    taxonomy_map, num_families, num_orders = tax_manager.build_mappings(
        load_cached_data=True
    )

    # 2. Define Transforms
    train_transform = get_transforms(mode="train", image_size=Config.IMAGE_SIZE)
    val_transform = get_transforms(mode="val", image_size=Config.IMAGE_SIZE)

    # 3. Create Datasets
    train_dataset = HerbariumDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        transform=train_transform,
        taxonomy_map=taxonomy_map,
    )

    val_dataset = HerbariumDataset(
        csv_path=Config.VAL_CSV,
        mode="val",
        transform=val_transform,
        taxonomy_map=taxonomy_map,
    )

    test_dataset = HerbariumDataset(
        csv_path=Config.TEST_CSV,
        mode="test",
        transform=val_transform,  # Use val transform (deterministic) for test
        taxonomy_map=None,
    )

    # 4. Configure Sampler for Training
    train_sampler = None
    shuffle = True

    if stage == 2:
        print(
            "Configuring Class-Balanced Sampler (WeightedRandomSampler) for Stage 2..."
        )
        # Get all labels
        labels = train_dataset.get_labels()

        # Calculate class counts
        class_counts = pd.Series(labels).value_counts().sort_index()

        # Calculate weight per class (inverse frequency)
        # We use a safe division
        class_weights = 1.0 / class_counts

        # Map weights to samples
        # This can be slow for 1M+ images, optimized via pandas map
        # Convert labels to series to map
        sample_weights = pd.Series(labels).map(class_weights).values

        # Convert to tensor
        sample_weights = torch.from_numpy(sample_weights).double()

        # Create sampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False  # Shuffle is mutually exclusive with sampler
        print("Sampler configured.")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        sampler=train_sampler,
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return (
        {"train": train_loader, "val": val_loader, "test": test_loader},
        num_families,
        num_orders,
    )
