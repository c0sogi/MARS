import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from library.config import INPUT_DIR, IMG_SIZE, SEED
from library.utils import seed_everything


def get_transforms(img_size=IMG_SIZE):
    """
    Creates the transformation pipeline for image preprocessing.

    Args:
        img_size (int): The target height and width for resizing.

    Returns:
        torchvision.transforms.Compose: The composed transforms.
    """
    # Standard ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class PlantDataset(Dataset):
    """
    A custom Dataset class for loading plant images and labels/IDs based on metadata CSVs.
    """

    def __init__(
        self,
        csv_file,
        root_dir=INPUT_DIR,
        transform=None,
        test_mode=False,
        label_map=None,
    ):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels/ids.
            root_dir (str): Root directory containing the images (default: INPUT_DIR).
            transform (callable, optional): Optional transform to be applied on a sample.
            test_mode (bool): Flag to indicate if the dataset is for testing (returns image_id)
                              or training/validation (returns label).
            label_map (dict, optional): Dictionary mapping raw labels to contiguous indices.
        """
        # Set seed for reproducibility
        seed_everything(SEED)

        self.root_dir = root_dir
        self.transform = transform
        self.test_mode = test_mode
        self.label_map = label_map

        # Load the metadata dataframe
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_file}")

        self.df = pd.read_csv(csv_file)

        # Validation of dataframe columns
        if self.test_mode:
            if "image_id" not in self.df.columns:
                # Ensure image_id column exists for test set
                raise ValueError(
                    f"Column 'image_id' missing in {csv_file} for testing."
                )
        else:
            if "label" not in self.df.columns:
                raise ValueError(
                    f"Column 'label' missing in {csv_file} for training/validation."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        # Combine root directory with relative path from CSV
        img_path = os.path.join(self.root_dir, row["image_path"])

        try:
            # Load image and convert to RGB (handles grayscale or CMYK)
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for corrupt or missing images to prevent crashing
            # Create a black image of the expected size
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.test_mode:
            # Return image and image_id for submission mapping
            return image, row["image_id"]
        else:
            # Return image and class label for training/evaluation
            label = int(row["label"])
            if self.label_map is not None:
                label = self.label_map[label]
            return image, label
