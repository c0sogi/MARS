import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import Tokenizer


def get_transforms(image_size: int):
    """
    Returns the image transformations for the dataset.

    Args:
        image_size (int): The target size (height and width) for resizing.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI molecule images.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        tokenizer: Tokenizer,
        transform: A.Compose = None,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing image_id, InChI, and file_path.
            root_dir (str): Root directory containing the image folders.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (A.Compose, optional): Transformations to apply to the images.
        """
        self.df = df
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # The metadata file_path is relative to input dir, e.g., "train/0/0/0/id.png"
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (though validation showed none)
            # Create a black image of expected size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Process Label
        text = row["InChI"]
        # Convert text to tensor sequence
        label_seq = self.tokenizer.text_to_sequence(text)

        return image, label_seq


def get_dataloaders(
    tokenizer: Tokenizer,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    debug_subset_size: int = None,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        tokenizer (Tokenizer): The tokenizer instance.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker threads.
        debug_subset_size (int, optional): If provided, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    # We expect these files to exist from the metadata generation step
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debugging: Subset the data if requested
    if debug_subset_size is not None:
        train_df = train_df.iloc[:debug_subset_size]
        val_df = val_df.iloc[:debug_subset_size]
        test_df = test_df.iloc[:debug_subset_size]
        print(f"Debug mode: Subsetting datasets to {debug_subset_size} samples.")

    # 2. Define Transforms
    transforms = get_transforms(Config.IMAGE_SIZE)

    # 3. Initialize Datasets
    train_dataset = InChiDataset(
        df=train_df,
        root_dir=Config.INPUT_DIR,
        tokenizer=tokenizer,
        transform=transforms,
    )

    val_dataset = InChiDataset(
        df=val_df,
        root_dir=Config.INPUT_DIR,
        tokenizer=tokenizer,
        transform=transforms,
    )

    test_dataset = InChiDataset(
        df=test_df,
        root_dir=Config.INPUT_DIR,
        tokenizer=tokenizer,
        transform=transforms,
    )

    # 4. Create DataLoaders
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
