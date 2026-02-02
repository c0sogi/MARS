import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from library.config import Config


def get_transforms(mode="train", image_size=Config.IMAGE_SIZE):
    """
    Returns the data transformation pipeline based on the mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        image_size (int): The input resolution for the model (e.g., 380 for B4).

    Returns:
        torchvision.transforms.Compose: The composed transform pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Scale up slightly before cropping to maintain object context
    # EfficientNet usually uses a specific crop ratio, but a general rule of thumb
    # is resizing to ~1.15x the crop size.
    resize_dim = int(image_size * 1.15)

    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test: Deterministic Center Crop
        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class INatDataset(Dataset):
    """
    PyTorch Dataset for the iNaturalist 2019 competition data.
    """

    def __init__(
        self,
        metadata_file,
        root_dir=Config.INPUT_DIR,
        transform=None,
        mode="train",
        debug=Config.DEBUG,
    ):
        """
        Args:
            metadata_file (str): Path to the CSV file containing metadata.
            root_dir (str): Root directory containing the image files.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines return values.
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        # Load metadata
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.df = pd.read_csv(metadata_file)

        # Handle Debugging
        if debug:
            subset_size = min(len(self.df), Config.DEBUG_SUBSET_SIZE)
            self.df = self.df.iloc[:subset_size].reset_index(drop=True)
            # print(f"DEBUG MODE: Loaded {len(self.df)} samples from {metadata_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # metadata 'file_name' is relative to input directory
        img_path = os.path.join(self.root_dir, row["file_name"])

        # Load image
        # Using cv2 for robustness, then converting to PIL for torchvision transforms
        image = cv2.imread(img_path)
        if image is None:
            # Fallback or error handling: create a black image if file is corrupt/missing
            # In a strict competition setting, we might want to raise an error,
            # but for training stability, a placeholder can be used.
            # However, the metadata script verified paths, so this should be rare.
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for transforms
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return based on mode
        if self.mode in ["train", "val"]:
            # Return image and label
            label = row["category_id"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Test mode: Return image and image_id (for submission)
            image_id = row["image_id"]
            return image, str(image_id)
