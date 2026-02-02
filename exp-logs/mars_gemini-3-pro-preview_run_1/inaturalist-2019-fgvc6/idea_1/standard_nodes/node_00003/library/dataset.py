import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(stage: str = "train"):
    """
    Returns the data transformation pipeline for a specific stage.

    Args:
        stage (str): One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composed transformations.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_list = []

    if stage == "train":
        # Stronger augmentation for fine-grained classification (Cite solution_lesson_node_00002)
        transform_list.append(
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.5, 1.0))
        )
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        )
    else:
        # Standard resizing for validation/test
        transform_list.append(transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)))

    # Convert PIL image to Tensor (HWC -> CHW, [0, 255] -> [0.0, 1.0])
    transform_list.append(transforms.ToTensor())

    # Normalize
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


class INatDataset(Dataset):
    """
    PyTorch Dataset for the iNaturalist 2019 competition data.
    Reads image paths and labels from metadata CSVs.
    """

    def __init__(self, csv_path, mode="train", transform=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (train, val, or test).
            mode (str): 'train', 'val', or 'test'. Determines return values.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.data = pd.read_csv(csv_path)

        # Pre-construct full file paths to avoid overhead in __getitem__
        # Using vectorized string operation for speed
        self.file_paths = (Config.INPUT_ROOT + os.sep + self.data["file_path"]).tolist()

        if self.mode != "test":
            self.labels = self.data["category_id"].tolist()
        else:
            self.image_ids = self.data["image_id"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple:
                - (image, target) where target is class_index if mode is 'train' or 'val'.
                - (image, image_id) if mode is 'test'.
        """
        img_path = self.file_paths[idx]

        # Open image and convert to RGB (handles grayscale or RGBA images)
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, IOError) as e:
            # Fallback for corrupt images or create a black image to prevent crashing
            print(f"Warning: Could not load image {img_path}: {e}")
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            image_id = self.image_ids[idx]
            return image, image_id
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
