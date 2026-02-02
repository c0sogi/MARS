import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from library.config import Config


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing leaf images.
    """

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Open image and convert to RGB to ensure 3 channels (ResNet expectation)
        # This handles binary/grayscale inputs correctly.
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image in case of error to prevent crash, though unlikely with verified metadata
            img = Image.new("RGB", (Config.IMG_WIDTH, Config.IMG_HEIGHT))

        if self.transform:
            img = self.transform(img)

        return img


class DeepFeatureExtractor:
    """
    Extracts high-level semantic features from images using a pre-trained ResNet18.
    """

    def __init__(self):
        set_seed(Config.RANDOM_SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize ResNet18 with ImageNet weights
        # We use the modern weights API
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)

        # Remove the classification head (fc layer)
        # ResNet18 structure: ... -> avgpool -> flatten -> fc
        # Replacing fc with Identity allows us to get the 512-dim vector after flatten
        self.model.fc = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # Define the preprocessing pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
                transforms.ToTensor(),
                # Normalize with ImageNet mean and std
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract(self, image_paths, cache_name=None, load_cached_data=True):
        """
        Extracts features for a list of image paths.

        Args:
            image_paths (list): List of file paths to images.
            cache_name (str, optional): Name of the cache file (without extension).
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Array of shape (N, 512) containing extracted features.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        cache_path = None
        if cache_name:
            cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.npy")

        # 1. Try to load from cache
        if load_cached_data and cache_path and os.path.exists(cache_path):
            print(f"Loading cached deep features from {cache_path}")
            try:
                features = np.load(cache_path)
                if features.shape[0] == len(image_paths):
                    return features
                else:
                    print(
                        f"Cache size mismatch ({features.shape[0]} vs {len(image_paths)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(
            f"Extracting deep features for {len(image_paths)} images using {self.device}..."
        )

        dataset = ImageDataset(image_paths, transform=self.transform)

        # Use num_workers for parallel data loading
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        features_list = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                # Forward pass
                # Output shape: (Batch Size, 512)
                outputs = self.model(batch)

                # Move to CPU and convert to numpy
                features_list.append(outputs.cpu().numpy())

        if features_list:
            features = np.vstack(features_list)
        else:
            # Handle empty input case
            features = np.empty((0, 512))

        # 3. Save to cache
        if cache_path:
            print(f"Saving deep features to {cache_path}")
            np.save(cache_path, features)

        return features
