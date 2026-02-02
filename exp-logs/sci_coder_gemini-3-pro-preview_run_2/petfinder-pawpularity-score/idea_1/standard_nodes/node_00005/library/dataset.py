import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from library.config import Config

# Fix random seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class PawpularityDataset(Dataset):
    """
    Dataset class for Pawpularity prediction.
    Loads images, applies transforms, and extracts metadata and targets.
    """

    def __init__(self, df, image_dir, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            image_dir (str): Root directory for images (usually Config.INPUT_DIR).
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Affects return values.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        # Pre-convert metadata to float32 numpy array for speed
        self.metadata = self.df[Config.METADATA_COLS].values.astype(np.float32)

        # Pre-convert targets if available
        if self.mode != "test":
            self.targets = self.df["Pawpularity"].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        # file_path in df is relative to input_dir (e.g., "train/id.jpg")
        rel_path = self.df.loc[idx, "file_path"]
        img_path = os.path.join(self.image_dir, rel_path)

        # Use OpenCV to load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen based on EDA)
            # Create a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            image = self.transform(image)

        # 3. Get Metadata
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)

        # 4. Get Target
        if self.mode != "test":
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, metadata, target
        else:
            # For test set, return a dummy target or just image/meta
            # Keeping signature consistent: (image, metadata, target)
            # Target is 0.0 for test
            return image, metadata, torch.tensor(0.0, dtype=torch.float32)


def get_dataloaders(debug=None, batch_size=None):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool, optional): Override Config.DEBUG.
        batch_size (int, optional): Override Config.BATCH_SIZE.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Use Config values if overrides not provided
    is_debug = debug if debug is not None else Config.DEBUG
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE

    # Define Transforms
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    # Validation/Test transform is usually the same for this simple baseline
    # (Resize + Normalize, no augmentation)
    val_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    # Load DataFrames
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Handle Debug Mode
    if is_debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        df_train = df_train.iloc[:subset_size]
        df_val = df_val.iloc[:subset_size]
        df_test = df_test.iloc[:subset_size]
        print(f"DEBUG MODE: Subsetting datasets to {subset_size} samples.")

    # Instantiate Datasets
    train_dataset = PawpularityDataset(
        df_train, Config.INPUT_DIR, transform=train_transform, mode="train"
    )

    val_dataset = PawpularityDataset(
        df_val, Config.INPUT_DIR, transform=val_transform, mode="val"
    )

    test_dataset = PawpularityDataset(
        df_test, Config.INPUT_DIR, transform=val_transform, mode="test"
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
