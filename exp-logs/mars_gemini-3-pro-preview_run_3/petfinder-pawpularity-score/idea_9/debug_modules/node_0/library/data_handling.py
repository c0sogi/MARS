import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import Config
from library.utils import seed_everything


class DualViewTransform:
    """
    Callable class that generates two views for every image:
    1. Global View: The full image resized to (IMAGE_SIZE, IMAGE_SIZE).
    2. Zoomed View: A center crop covering ZOOM_CROP_RATIO of the area,
       then resized to (IMAGE_SIZE, IMAGE_SIZE).

    Both views are normalized using ImageNet mean and std.
    """

    def __init__(self, image_size=224, zoom_crop_area_ratio=0.6):
        self.image_size = image_size
        # Calculate linear zoom factor from area ratio (sqrt(0.6) approx 0.775)
        self.zoom_factor = np.sqrt(zoom_crop_area_ratio)

        # Standard ImageNet normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Resize transform
        self.resize_target = transforms.Resize((self.image_size, self.image_size))

    def __call__(self, img: Image.Image):
        """
        Args:
            img (PIL.Image): Input image.

        Returns:
            tuple: (global_view_tensor, zoomed_view_tensor)
        """
        # --- 1. Global View ---
        # Resize full image to target size
        global_img = self.resize_target(img)
        global_tensor = self.normalize(global_img)

        # --- 2. Zoomed View ---
        w, h = img.size
        crop_h = int(h * self.zoom_factor)
        crop_w = int(w * self.zoom_factor)

        # Perform Center Crop
        # Note: Since crop dimensions are derived from image size with factor < 1,
        # crop is always valid.
        zoomed_img = transforms.CenterCrop((crop_h, crop_w))(img)

        # Resize cropped view to target size
        zoomed_img = self.resize_target(zoomed_img)
        zoomed_tensor = self.normalize(zoomed_img)

        return global_tensor, zoomed_tensor


class PetDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity task.
    Loads images, generates dual views, and retrieves metadata/targets.
    """

    def __init__(self, meta_csv_path, mode="train", transform=None):
        """
        Args:
            meta_csv_path (str): Path to the metadata CSV file (generated in metadata/).
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Custom transform. If None, uses DualViewTransform.
        """
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Load Metadata
        if not os.path.exists(meta_csv_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_csv_path}")

        self.df = pd.read_csv(meta_csv_path)

        # Setup Transform
        if transform is None:
            self.transform = DualViewTransform(
                image_size=Config.IMAGE_SIZE,
                zoom_crop_area_ratio=Config.ZOOM_CROP_RATIO,
            )
        else:
            self.transform = transform

        # Define Metadata Feature Columns
        self.meta_features_cols = [
            "Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

        # Pre-process Metadata:
        # Extract binary features and scale them as per Config (e.g., * 10.0)
        # This helps the model weigh these features appropriately against dense image embeddings.
        self.meta_data = (
            self.df[self.meta_features_cols].values.astype(np.float32)
            * Config.METADATA_SCALE
        )

        # Pre-process Targets:
        if self.mode in ["train", "val"]:
            self.targets = self.df["Pawpularity"].values.astype(np.float32)
        else:
            # Dummy targets for test set
            self.targets = np.zeros(len(self.df), dtype=np.float32)

        # Pre-process File Paths:
        # Metadata contains relative paths (e.g., "train/id.jpg")
        self.file_paths = self.df["file_path"].values
        self.ids = self.df["Id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load Image
        # We use PIL for seamless integration with torchvision transforms
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            # Fallback for robustness (should not happen given metadata verification)
            print(f"Error loading image {full_path}: {e}. Using black image.")
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))

        # Apply Dual View Transform
        # Returns two tensors: global view and zoomed view
        global_view, zoomed_view = self.transform(image)

        # Retrieve Metadata and Target
        meta = torch.tensor(self.meta_data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        img_id = self.ids[idx]

        return {
            "global_view": global_view,
            "zoomed_view": zoomed_view,
            "metadata": meta,
            "target": target,
            "id": img_id,
        }


def get_pet_dataloader(
    meta_csv_path,
    mode="train",
    batch_size=32,
    num_workers=Config.NUM_WORKERS,
    shuffle=None,
):
    """
    Factory function to create a configured DataLoader.

    Args:
        meta_csv_path (str): Path to metadata CSV.
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of CPU workers.
        shuffle (bool, optional): Whether to shuffle. Defaults to True for train, False otherwise.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Determine shuffle logic if not explicitly provided
    if shuffle is None:
        shuffle = mode == "train"

    dataset = PetDataset(meta_csv_path, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train"),  # Drop incomplete batch only during training
    )

    return loader
