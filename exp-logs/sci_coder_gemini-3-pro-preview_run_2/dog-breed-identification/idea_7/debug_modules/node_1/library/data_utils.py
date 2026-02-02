import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from library.config import Config


class DogDataset(Dataset):
    """
    Dataset class for the Dog Breed Prediction Task.
    Loads images and labels/IDs based on metadata CSVs.
    Returns raw PIL images to allow for multi-view processing downstream.
    """

    def __init__(
        self, metadata_path, input_dir, mode="train", class_to_idx=None, debug=False
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            input_dir (str): Root directory for image files.
            mode (str): One of 'train', 'val', 'test'.
            class_to_idx (dict, optional): Dictionary mapping breed names to integers. Required for 'train' and 'val'.
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        self.metadata_path = metadata_path
        self.input_dir = input_dir
        self.mode = mode
        self.class_to_idx = class_to_idx
        self.debug = debug

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Apply Debug Limit
        if self.debug:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)

        # Validation
        if self.mode in ["train", "val"]:
            if "breed" not in self.df.columns:
                raise ValueError(
                    f"Metadata for {self.mode} must contain 'breed' column."
                )
            if self.class_to_idx is None:
                raise ValueError(
                    "class_to_idx must be provided for train/val datasets."
                )
        elif self.mode == "test":
            if "id" not in self.df.columns:
                raise ValueError("Metadata for test must contain 'id' column.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image (PIL.Image): Raw RGB image.
            target (int or str): Label index (train/val) or Image ID (test).
        """
        row = self.df.iloc[idx]

        # Construct file path
        # Metadata contains relative path e.g. "train/xxx.jpg"
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load Image
        # Convert to RGB to ensure 3 channels (handles Grayscale/RGBA)
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image at {full_path}: {e}")

        # Return Target
        if self.mode in ["train", "val"]:
            breed = row["breed"]
            target = self.class_to_idx[breed]
            return image, target
        else:
            image_id = row["id"]
            return image, image_id


def collate_pil(batch):
    """
    Custom collate function to handle batches of PIL images.
    Standard default_collate fails on PIL images.

    Args:
        batch: List of tuples (image, target)

    Returns:
        images: List of PIL.Image objects
        targets: Tensor of labels (int) or Tuple of IDs (str)
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Check target type to distinguish between train/val (int) and test (str)
    if isinstance(targets[0], int):
        targets = torch.tensor(targets, dtype=torch.long)
    else:
        targets = tuple(targets)

    return images, targets


def get_dataloaders(debug=Config.DEBUG, batch_size=None, num_workers=None):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        debug (bool): Whether to run in debug mode (subset of data).
        batch_size (int, optional): Batch size override.
        num_workers (int, optional): Number of workers override.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    # Resolve parameters
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # 1. Establish Class Mapping (Deterministic)
    # We read the full training set metadata to ensure all classes are captured and sorted.
    train_meta_path = Config.TRAIN_METADATA_PATH
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Training metadata not found at {train_meta_path}")

    full_train_df = pd.read_csv(train_meta_path)
    classes = sorted(full_train_df["breed"].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # 2. Instantiate Datasets
    train_dataset = DogDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        input_dir=Config.INPUT_DIR,
        mode="train",
        class_to_idx=class_to_idx,
        debug=debug,
    )

    val_dataset = DogDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        input_dir=Config.INPUT_DIR,
        mode="val",
        class_to_idx=class_to_idx,
        debug=debug,
    )

    test_dataset = DogDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        input_dir=Config.INPUT_DIR,
        mode="test",
        class_to_idx=None,
        debug=debug,
    )

    # 3. Instantiate DataLoaders
    # Note: We use collate_pil because Dataset returns PIL images, not Tensors.
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_pil,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_pil,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_pil,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, class_to_idx
