import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.utils import set_seed


class TaxonomyProcessor:
    """
    Handles the processing of hierarchical taxonomy data (Family -> Genus -> Species).
    Manages caching of mappings and encoders to ensure consistency and speed.
    """

    def __init__(
        self, metadata_json_path=Config.TRAIN_METADATA_JSON, cache_dir=Config.CACHE_DIR
    ):
        self.metadata_json_path = metadata_json_path
        self.cache_dir = cache_dir
        self.mapping_cache_path = os.path.join(cache_dir, "taxonomy_mapping.parquet")
        self.counts_cache_path = os.path.join(cache_dir, "taxonomy_counts.json")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_taxonomy(self, load_cached_data=True):
        """
        Loads taxonomy metadata, encodes Family and Genus strings to integers,
        and returns a mapping DataFrame and count metadata.

        Returns:
            mapping_df (pd.DataFrame): DataFrame with columns [category_id, genus_id, family_id].
            counts (dict): Dictionary containing num_families and num_genera.
        """
        # 1. Try to load from cache
        if (
            load_cached_data
            and os.path.exists(self.mapping_cache_path)
            and os.path.exists(self.counts_cache_path)
        ):
            try:
                mapping_df = pd.read_parquet(self.mapping_cache_path)
                with open(self.counts_cache_path, "r") as f:
                    counts = json.load(f)
                return mapping_df, counts
            except Exception as e:
                print(f"Failed to load cached taxonomy data: {e}. Recomputing...")

        # 2. Compute from scratch
        if not os.path.exists(self.metadata_json_path):
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_json_path}"
            )

        with open(self.metadata_json_path, "r") as f:
            meta = json.load(f)

        # Extract categories list: expected structure list of dicts with family, genus, category_id
        if "categories" not in meta:
            raise ValueError("Invalid metadata format: 'categories' key missing.")

        categories = meta["categories"]
        df = pd.DataFrame(categories)

        # Ensure we have the required columns
        required_cols = ["category_id", "genus", "family"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"Metadata categories missing required columns. Found: {df.columns}"
            )

        # Encode Family and Genus
        le_family = LabelEncoder()
        df["family_id"] = le_family.fit_transform(df["family"])

        le_genus = LabelEncoder()
        df["genus_id"] = le_genus.fit_transform(df["genus"])

        # Select relevant columns for the mapping
        mapping_df = df[["category_id", "genus_id", "family_id"]].copy()
        mapping_df["category_id"] = mapping_df["category_id"].astype(int)
        mapping_df["genus_id"] = mapping_df["genus_id"].astype(int)
        mapping_df["family_id"] = mapping_df["family_id"].astype(int)

        # Sort by category_id for faster lookups if needed
        mapping_df = mapping_df.sort_values("category_id").reset_index(drop=True)

        counts = {
            "num_families": int(df["family_id"].max() + 1),
            "num_genera": int(df["genus_id"].max() + 1),
            "num_species": int(df["category_id"].max() + 1),
        }

        # 3. Save to cache
        mapping_df.to_parquet(self.mapping_cache_path, index=False)
        with open(self.counts_cache_path, "w") as f:
            json.dump(counts, f)

        return mapping_df, counts


class PlantDataset(Dataset):
    """
    PyTorch Dataset for Plant Classification.
    Returns image and hierarchical labels (Species, Genus, Family).
    """

    def __init__(self, df, taxonomy_map_df, transform=None, base_path=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'image_path' and 'label' (category_id).
            taxonomy_map_df (pd.DataFrame): Mapping from category_id to genus_id and family_id.
            transform (callable, optional): Optional transform to be applied on a sample.
            base_path (str): Base directory for images.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.base_path = base_path

        # Create efficient lookup dicts
        # Assuming taxonomy_map_df has unique category_ids
        self.genus_lookup = dict(
            zip(taxonomy_map_df["category_id"], taxonomy_map_df["genus_id"])
        )
        self.family_lookup = dict(
            zip(taxonomy_map_df["category_id"], taxonomy_map_df["family_id"])
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image Loading
        # row['image_path'] is relative to input dir, e.g., "train_images/..."
        img_path = os.path.join(self.base_path, row["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for missing/corrupt images (should be rare given validation)
            # Return a black image or raise error depending on strictness.
            # Here we create a blank image to prevent crashing the loader.
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        # Target Loading
        species_label = int(row["label"])
        genus_label = self.genus_lookup.get(species_label, -1)
        family_label = self.family_lookup.get(species_label, -1)

        # Return tuple of targets for multi-task learning
        return image, (species_label, genus_label, family_label)


class TestDataset(Dataset):
    """
    PyTorch Dataset for Inference. Returns image and image_id.
    """

    def __init__(self, df, transform=None, base_path=Config.INPUT_DIR):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, row["image_path"])
        image_id = row["image_id"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, image_id


def get_dataloaders(
    train_csv_path=Config.TRAIN_CSV,
    val_csv_path=Config.VAL_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=False,
    data_subset_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Constructs Train and Validation DataLoaders with transforms and sampling.

    Returns:
        train_loader, val_loader, taxonomy_counts (dict)
    """
    set_seed(Config.SEED)

    # 1. Prepare Taxonomy Mapping
    processor = TaxonomyProcessor()
    mapping_df, counts = processor.process_taxonomy(load_cached_data=load_cached_data)

    # 2. Load DataFrames
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # Debugging / Subsetting
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), data_subset_size), random_state=Config.SEED
        )
        df_val = df_val.sample(
            n=min(len(df_val), data_subset_size), random_state=Config.SEED
        )

    # 3. Define Transforms
    # Training: Resize -> RandAugment -> Tensor -> Normalize
    train_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Validation: Resize -> Tensor -> Normalize
    val_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 4. Create Datasets
    train_dataset = PlantDataset(df_train, mapping_df, transform=train_transform)
    val_dataset = PlantDataset(df_val, mapping_df, transform=val_transform)

    # 5. Imbalanced Data Handling (WeightedRandomSampler)
    # Calculate weights for each sample based on the inverse frequency of its class
    class_counts = df_train["label"].value_counts().sort_index()
    # Ensure all classes in the subset are accounted for; map label to count
    count_map = class_counts.to_dict()

    # Calculate weight per class
    # We use a smoothed inverse frequency or just 1/N
    weights = []
    for label in df_train["label"]:
        c = count_map.get(label, 0)
        if c > 0:
            weights.append(1.0 / c)
        else:
            weights.append(0.0)

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Sampler implies shuffle=False
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

    return train_loader, val_loader, counts


def get_test_dataloader(
    test_csv_path=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Constructs the Test DataLoader.
    """
    df_test = pd.read_csv(test_csv_path)

    test_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = TestDataset(df_test, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
