import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="data")


class DogDataset(Dataset):
    """
    Custom Dataset for Dog Breed Classification.
    Handles image loading, label encoding, and transformations.
    """

    def __init__(self, df, class_to_idx=None, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'breed' (for train/val).
            class_to_idx (dict, optional): Mapping from breed name to integer index.
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.class_to_idx = class_to_idx

        # Pre-calculate full paths to avoid overhead in __getitem__
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].tolist()
        ]

        if self.mode != "test":
            self.labels = df["breed"].tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        # Load image (RGB)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately.
            # For this task, we assume data integrity based on metadata validation.
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            return image, self.df.iloc[idx]["id"]

        # Encode label
        label_str = self.labels[idx]
        label = self.class_to_idx[label_str]

        return image, torch.tensor(label, dtype=torch.long)


def get_train_transforms():
    """
    Returns the training augmentation pipeline enforcing Geometric Diversity.
    Sequence: RandomResizedCrop -> RandomHorizontalFlip -> RandAugment -> Norm
    """
    return T.Compose(
        [
            T.RandomResizedCrop(
                size=Config.IMG_SIZE,
                scale=Config.AUG_CROP_SCALE,
                ratio=Config.AUG_CROP_RATIO,
            ),
            T.RandomHorizontalFlip(p=Config.AUG_FLIP_PROB),
            T.RandAugment(
                num_ops=Config.AUG_RANDAUG_NUM_OPS,
                magnitude=Config.AUG_RANDAUG_MAGNITUDE,
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_valid_transforms():
    """
    Returns the validation/test transformation pipeline.
    Sequence: Resize (256) -> CenterCrop (224) -> Norm
    """
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(Config.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_loaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Prepares and returns DataLoaders for training, validation, and testing.

    Args:
        load_cached_data (bool): If True, attempts to load class mapping from cache.
        debug (bool): If True, subsets the data for rapid debugging.

    Returns:
        train_loader, val_loader, test_loader, class_list
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "classes.parquet")

    # 1. Load Metadata
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # 2. Handle Class Mapping (Caching Logic)
    class_list = None

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached class list from {cache_path}")
        try:
            df_classes = pd.read_parquet(cache_path)
            class_list = df_classes["breed"].tolist()
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing.")
            class_list = None

    if class_list is None:
        logger.info("Computing class list from training data.")
        # Get unique breeds from training data and sort them
        class_list = sorted(df_train["breed"].unique().tolist())

        # Save to cache
        logger.info(f"Saving class list to {cache_path}")
        pd.DataFrame({"breed": class_list}).to_parquet(cache_path, index=False)

    if debug:
        logger.info(
            f"Debug mode: Subsetting data to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Create mapping
    class_to_idx = {breed: idx for idx, breed in enumerate(class_list)}

    logger.info(f"Number of classes: {len(class_list)}")

    # 3. Create Datasets
    train_dataset = DogDataset(
        df_train,
        class_to_idx=class_to_idx,
        transform=get_train_transforms(),
        mode="train",
    )

    val_dataset = DogDataset(
        df_val, class_to_idx=class_to_idx, transform=get_valid_transforms(), mode="val"
    )

    test_dataset = DogDataset(
        df_test,
        class_to_idx=None,  # Test set has no labels
        transform=get_valid_transforms(),
        mode="test",
    )

    # 4. Create DataLoaders
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

    logger.info(
        f"DataLoaders created. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches."
    )

    return train_loader, val_loader, test_loader, class_list
