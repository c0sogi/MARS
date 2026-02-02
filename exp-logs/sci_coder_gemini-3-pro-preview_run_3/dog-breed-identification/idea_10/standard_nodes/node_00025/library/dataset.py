import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset")


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads a deterministic mapping between breed names and integer indices.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        dict: A dictionary mapping breed names (str) to indices (int).
        list: A list of breed names sorted by index.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_classes = pd.read_parquet(cache_path)
            class_to_idx = {
                row["breed"]: row["idx"] for _, row in df_classes.iterrows()
            }
            classes = df_classes.sort_values("idx")["breed"].tolist()
            logger.info(f"Loaded class mapping from {cache_path}")
            return class_to_idx, classes
        except Exception as e:
            logger.warning(f"Failed to load class cache: {e}. Recomputing...")

    # 2. Compute from scratch
    logger.info("Computing class mapping from training metadata...")
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_CSV}")

    df_train = pd.read_csv(Config.TRAIN_CSV)
    unique_breeds = sorted(df_train["breed"].unique().tolist())

    class_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_classes = pd.DataFrame(
        {"breed": list(class_to_idx.keys()), "idx": list(class_to_idx.values())}
    )
    df_classes.to_parquet(cache_path, index=False)
    logger.info(f"Saved class mapping to {cache_path}")

    return class_to_idx, unique_breeds


def get_transforms(phase):
    """
    Returns the data augmentation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    mean = Config.MEAN
    std = Config.STD
    img_size = Config.IMG_SIZE

    if phase == "train":
        # Proposed Solution: RRC -> HorizontalFlip -> RandAugment
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation/Test: Resize(256) -> CenterCrop(224)
        # This maintains aspect ratio better than direct resize
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    """

    def __init__(
        self, csv_path, class_to_idx, transform=None, debug=False, is_test=False
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            class_to_idx (dict): Mapping from breed name to integer index.
            transform (callable, optional): Optional transform to be applied on a sample.
            debug (bool): If True, use a small subset of the data.
            is_test (bool): If True, handle test set (no labels expected).
        """
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_test = is_test
        self.input_dir = Config.INPUT_DIR

        if debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)
            logger.info(f"Debug mode: Dataset reduced to {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains relative paths like 'train/id.jpg'
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error to prevent crash, though unlikely with validated metadata
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        # Apply Transforms
        if self.transform:
            image = self.transform(image)

        # Get ID
        image_id = row["id"]

        # Get Label
        if self.is_test:
            label = -1  # Dummy label for test set
        else:
            breed = row["breed"]
            label = self.class_to_idx[breed]

        return image, label, image_id


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, runs on a small subset of data.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders and 'class_mapping'.
    """
    # 1. Get Class Mapping
    class_to_idx, classes = get_class_mapping(load_cached_data=True)

    # 2. Define Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")  # Used for both val and test

    # 3. Create Datasets
    train_dataset = DogDataset(
        csv_path=Config.TRAIN_CSV,
        class_to_idx=class_to_idx,
        transform=train_transform,
        debug=debug,
        is_test=False,
    )

    val_dataset = DogDataset(
        csv_path=Config.VAL_CSV,
        class_to_idx=class_to_idx,
        transform=val_transform,
        debug=debug,
        is_test=False,
    )

    test_dataset = DogDataset(
        csv_path=Config.TEST_CSV,
        class_to_idx=class_to_idx,  # Not used for labels, but passed for consistency
        transform=val_transform,
        debug=debug,
        is_test=True,
    )

    # 4. Create DataLoaders
    # Pin memory speeds up host to device copy
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_to_idx": class_to_idx,
        "idx_to_class": classes,
    }
