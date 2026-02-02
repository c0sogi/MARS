import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_9"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_dataframes(load_cached_data: bool = True):
    """
    Loads metadata dataframes with caching logic strictly following requirements.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    datasets = {}
    modes = ["train", "val", "test"]

    for mode in modes:
        cache_path = os.path.join(CACHE_DIR, f"{mode}_meta.parquet")
        csv_path = os.path.join("./metadata", f"{mode}.csv")

        df = None

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # print(f"Loaded {mode} metadata from cache.")
            except Exception:
                # print(f"Failed to load {mode} cache, reloading from source.")
                pass

        # 2. If loading failed or not requested, process from scratch
        if df is None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Metadata file not found: {csv_path}")

            df = pd.read_csv(csv_path)

            # Save to cache
            df.to_parquet(cache_path, index=False)
            # print(f"Saved {mode} metadata to cache.")

        datasets[mode] = df

    return datasets["train"], datasets["val"], datasets["test"]


def get_transforms(resolution: int, mode: str = "train"):
    """
    Returns transforms based on resolution and mode.
    Uses Bicubic interpolation and context-preserving augmentations.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (resolution, resolution),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        # Validation / Test
        return transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )


class PetDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(INPUT_DIR, row["filepath"])

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images (though dataset analysis showed none)
            # Return a black image to prevent crash
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.mode in ["train", "val"]:
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
        else:
            # For test, return image and id
            img_id = row["id"]
            return image, img_id


def get_dataloaders(
    batch_size: int = 32,
    resolution: int = 224,
    num_workers: int = 4,
    load_cached_data: bool = True,
    debug_subset: int = None,
):
    """
    Constructs dataloaders for the specific resolution pipeline.

    Args:
        batch_size: Batch size for training/inference.
        resolution: Input resolution (224, 256, or 320).
        num_workers: Number of subprocesses for data loading.
        load_cached_data: Whether to use cached metadata.
        debug_subset: If set, limits dataset size for debugging.
    """
    seed_everything(42)

    # Load Metadata
    train_df, val_df, test_df = get_dataframes(load_cached_data=load_cached_data)

    if debug_subset:
        train_df = train_df.iloc[:debug_subset]
        val_df = val_df.iloc[:debug_subset]
        test_df = test_df.iloc[:debug_subset]

    # Define Transforms
    train_transform = get_transforms(resolution, mode="train")
    val_transform = get_transforms(resolution, mode="val")

    # Create Datasets
    train_dataset = PetDataset(train_df, transform=train_transform, mode="train")
    val_dataset = PetDataset(val_df, transform=val_transform, mode="val")
    test_dataset = PetDataset(test_df, transform=val_transform, mode="test")

    # Create Dataloaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
