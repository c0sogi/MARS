import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


def load_metadata_splits():
    """
    Loads the training, validation, and test metadata DataFrames from the
    paths defined in Config. Respects the DEBUG flag to subsample data.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Load DataFrames
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"[DEBUG] Subsampling datasets to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    return train_df, val_df, test_df


def get_transforms(image_size, mean=None, std=None):
    """
    Creates the transformation pipeline for the images.

    Args:
        image_size (int): The target height and width for resizing.
        mean (tuple, optional): Normalization mean. Defaults to ImageNet stats.
        std (tuple, optional): Normalization std. Defaults to ImageNet stats.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity Prediction task.
    Handles image loading, metadata extraction, and Test-Time Augmentation (TTA).
    """

    def __init__(self, df, root_dir, transform=None, use_tta=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            root_dir (str): Root directory where images are stored (e.g., ./input).
            transform (callable, optional): Transform to be applied on a sample.
            use_tta (bool): If True, returns a stack of [original, flipped] images.
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.use_tta = use_tta

        # Pre-check for target column existence
        self.has_target = "Pawpularity" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative (e.g., "train/xxx.jpg")
        # root_dir is "./input"
        img_path = os.path.join(self.root_dir, row["file_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError) as e:
            # Fallback for missing images (should not happen based on metadata check)
            # Create a blank image to prevent crash
            print(f"Warning: Could not load image {img_path}. Using blank image.")
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # 2. Apply Transforms and TTA
        if self.use_tta:
            # Generate original and horizontally flipped versions
            # Note: TTA logic here assumes the transform includes ToTensor/Normalize
            # We apply the base transform to both.

            # Create flipped copy
            image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

            if self.transform:
                img_tensor_orig = self.transform(image)
                img_tensor_flip = self.transform(image_flip)
            else:
                to_tensor = transforms.ToTensor()
                img_tensor_orig = to_tensor(image)
                img_tensor_flip = to_tensor(image_flip)

            # Stack: (2, C, H, W)
            image_tensor = torch.stack([img_tensor_orig, img_tensor_flip])

        else:
            # Standard single image processing
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)

        # 3. Extract Metadata Features
        # Extract the binary features defined in Config
        meta_features = row[Config.METADATA_COLS].values.astype(np.float32)
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32)

        # 4. Extract Target
        if self.has_target:
            target = row["Pawpularity"]
            target_tensor = torch.tensor(target, dtype=torch.float32)
        else:
            # Return dummy target for test set
            target_tensor = torch.tensor(0.0, dtype=torch.float32)

        # 5. Extract ID
        sample_id = str(row["Id"])

        return image_tensor, meta_tensor, target_tensor, sample_id
