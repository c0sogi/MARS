import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from library.config import Config


def get_transforms(phase: str, img_size: int = 224):
    """
    Returns the data augmentation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
        img_size (int): Target image size.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Validation or Test
        # Resize to slightly larger than target (256 for 224 target), then center crop
        resize_dim = int(img_size * 256 / 224)
        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_data_and_classes(load_cached_data: bool = True, debug: bool = False):
    """
    Loads the metadata, processes class mappings, and handles caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.
        debug (bool): If True, returns a small subset of data and disables caching.

    Returns:
        tuple: (full_train_df, test_df, class_to_idx, classes_list)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    classes_path = os.path.join(cache_dir, "classes.parquet")
    train_meta_path = os.path.join(cache_dir, "train_meta.parquet")
    test_meta_path = os.path.join(cache_dir, "test_meta.parquet")

    # If debug is ON, we force re-computation (no cache loading) and do not save cache
    if debug:
        load_cached_data = False

    # 1. Try loading cached data
    if (
        load_cached_data
        and os.path.exists(classes_path)
        and os.path.exists(train_meta_path)
        and os.path.exists(test_meta_path)
    ):
        try:
            classes_df = pd.read_parquet(classes_path)
            full_train_df = pd.read_parquet(train_meta_path)
            test_df = pd.read_parquet(test_meta_path)

            classes = classes_df["breed"].tolist()
            class_to_idx = {cls: i for i, cls in enumerate(classes)}

            return full_train_df, test_df, class_to_idx, classes
        except Exception as e:
            # Fallback to recomputing if cache load fails
            pass

    # 2. Compute from scratch
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(Config.METADATA_DIR, "val.csv")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Metadata not found at {train_csv}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Combine train and val for Cross-Validation
    full_train_df = pd.concat([df_train, df_val], ignore_index=True)

    # Extract and sort classes to ensure deterministic mapping
    classes = sorted(full_train_df["breed"].unique().tolist())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Debugging: Sample data
    if debug:
        full_train_df = full_train_df.sample(
            n=min(len(full_train_df), Config.DEBUG_SAMPLE_SIZE),
            random_state=Config.SEED,
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # Do not save cache in debug mode
        return full_train_df, test_df, class_to_idx, classes

    # 3. Save to cache
    classes_df = pd.DataFrame({"breed": classes})
    classes_df.to_parquet(classes_path)
    full_train_df.to_parquet(train_meta_path)
    test_df.to_parquet(test_meta_path)

    return full_train_df, test_df, class_to_idx, classes


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    """

    def __init__(self, df, class_to_idx=None, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            class_to_idx (dict): Mapping from breed name to integer index. Required for train/val.
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "train/id.jpg"
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for corrupted/missing images
            # Return a black image to prevent crashing
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mode in ["train", "val"]:
            breed = row["breed"]
            label = self.class_to_idx[breed]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Test mode: return image and ID
            return image, row["id"]
