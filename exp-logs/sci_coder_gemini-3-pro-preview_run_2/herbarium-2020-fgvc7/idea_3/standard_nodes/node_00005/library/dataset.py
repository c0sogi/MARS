import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.utils import set_seed
from library.taxonomy import get_taxonomy_mappings

# ImageNet normalization constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PlantDataset(Dataset):
    """
    Dataset class for Plant Species Classification.
    Returns image and hierarchical labels (Species, Genus, Family).
    """

    def __init__(
        self,
        df,
        root_dir,
        transform=None,
        label_map=None,
        taxonomy_maps=None,
        is_test=False,
    ):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map  # dict: category_id -> contiguous_idx
        self.is_test = is_test

        # Unpack taxonomy maps if provided
        if taxonomy_maps:
            self.species_to_genus, self.species_to_family = taxonomy_maps
        else:
            self.species_to_genus, self.species_to_family = None, None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = ToTensorV2()(image=image)["image"]

        # Test mode: return image and ID for submission
        if self.is_test:
            return image, row["image_id"]

        # Train/Val mode: return image and hierarchical labels
        category_id = row["category_id"]

        # Map sparse category_id to contiguous index
        if self.label_map:
            species_label = self.label_map[category_id]
        else:
            species_label = category_id

        # Retrieve auxiliary labels (Genus, Family)
        # These IDs are already contiguous from library.taxonomy
        genus_label = self.species_to_genus.get(category_id, -1)
        family_label = self.species_to_family.get(category_id, -1)

        return {
            "image": image,
            "species": torch.tensor(species_label, dtype=torch.long),
            "genus": torch.tensor(genus_label, dtype=torch.long),
            "family": torch.tensor(family_label, dtype=torch.long),
            "category_id": torch.tensor(category_id, dtype=torch.long),
        }


def get_label_map(train_csv_path, load_cached_data=True):
    """
    Creates or loads a mapping from original category_id to contiguous integers.
    """
    cache_dir = "./working/idea_3"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "label_map.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            label_map = np.load(cache_file, allow_pickle=True).item()
            return label_map
        except Exception:
            pass  # Proceed to recompute if load fails

    # Compute mapping
    df = pd.read_csv(train_csv_path)
    unique_cats = sorted(df["category_id"].unique())
    label_map = {cat: idx for idx, cat in enumerate(unique_cats)}

    # Save to cache
    np.save(cache_file, label_map)
    return label_map


def get_dataloaders(
    input_dir="./input",
    batch_size=32,
    image_size=380,
    num_workers=4,
    debug_size=None,
    load_cached_data=True,
):
    """
    Prepares DataLoaders for training and validation.
    """
    set_seed(42)

    train_csv = os.path.join(input_dir, "metadata/train.csv")
    val_csv = os.path.join(input_dir, "metadata/val.csv")
    train_meta_json = os.path.join(input_dir, "nybg2020/train/metadata.json")

    # 1. Get Label Map (Species ID -> Contiguous Index)
    label_map = get_label_map(train_csv, load_cached_data=load_cached_data)
    num_classes = len(label_map)

    # 2. Get Taxonomy Mappings (Species ID -> Genus ID, Family ID)
    species_to_genus, species_to_family, num_genera, num_families = (
        get_taxonomy_mappings(
            metadata_path=train_meta_json, load_cached_data=load_cached_data
        )
    )
    taxonomy_maps = (species_to_genus, species_to_family)

    # 3. Define Transforms
    # Train: Random Resized Crop + Flip
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

    # Val: Resize (preserve aspect ratio) + Center Crop
    val_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=int(image_size * 1.15)),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

    # 4. Load DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if debug_size:
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]

    # 5. Create Datasets
    train_dataset = PlantDataset(
        train_df,
        input_dir,
        transform=train_transform,
        label_map=label_map,
        taxonomy_maps=taxonomy_maps,
        is_test=False,
    )

    val_dataset = PlantDataset(
        val_df,
        input_dir,
        transform=val_transform,
        label_map=label_map,
        taxonomy_maps=taxonomy_maps,
        is_test=False,
    )

    # 6. Create DataLoaders
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
    )

    return train_loader, val_loader, num_classes, num_genera, num_families, label_map


def get_test_dataloader(
    input_dir="./input", batch_size=32, image_size=380, num_workers=4
):
    """
    Prepares DataLoader for the test set.
    """
    test_csv = os.path.join(input_dir, "metadata/test.csv")
    test_df = pd.read_csv(test_csv)

    # Test Transform: Same as Validation
    transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=int(image_size * 1.15)),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

    dataset = PlantDataset(
        test_df,
        input_dir,
        transform=transform,
        label_map=None,
        taxonomy_maps=None,
        is_test=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
