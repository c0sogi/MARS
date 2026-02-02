import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(split: str, img_size: int):
    """
    Returns the torchvision transform pipeline based on the split and image size.
    Implements the Dynamic Fidelity strategy (varying img_size) and
    PIL-native augmentations.

    Args:
        split (str): 'train' or 'val'/'test'.
        img_size (int): Target spatial dimension (e.g., 224 or 384).

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose(
            [
                # Geometric Augmentations
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                # Photometric Augmentations (Automated)
                transforms.RandAugment(num_ops=2, magnitude=9),
                # Conversion and Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Validation/Test: Deterministic resizing
        return transforms.Compose(
            [
                # Resize the smaller edge to img_size, maintaining aspect ratio
                transforms.Resize(img_size),
                # Crop the center square
                transforms.CenterCrop(img_size),
                # Conversion and Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease Classification.
    Strictly uses PIL for image loading to align with torchvision transforms.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, output_label: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label' columns.
            transforms (callable, optional): Transform pipeline to apply to the image.
            output_label (bool): Whether to return the label (True for train/val, False optional for test).
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.output_label = output_label
        self.root_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input dir (e.g., "train_images/123.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using PIL (Native RGB)
        # Using .convert('RGB') ensures consistency for 1-channel or 4-channel images
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of corruption to prevent crash
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        # Return data
        if self.output_label:
            label = torch.tensor(row["label"], dtype=torch.long)
            return image, label
        else:
            return image


def load_metadata(split: str, debug: bool = False):
    """
    Loads the metadata DataFrame for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, returns a small subset of the data for debugging.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA
    elif split == "val":
        path = Config.VAL_METADATA
    elif split == "test":
        path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    if debug:
        # Sample a small subset for debugging
        debug_size = min(100, len(df))
        df = df.sample(n=debug_size, random_state=Config.SEED).reset_index(drop=True)
        print(f"[DEBUG] Loaded subset of {len(df)} samples for split '{split}'")

    return df
