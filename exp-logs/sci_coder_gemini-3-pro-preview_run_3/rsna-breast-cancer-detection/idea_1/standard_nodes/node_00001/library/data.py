import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from library.config import Config
from library.utils import load_dicom_image


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for Breast Cancer Detection.
    Handles loading DICOM images, converting to 3-channel tensors,
    and applying augmentations.
    """

    def __init__(self, metadata_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Determines return values.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Pre-construct full file paths to avoid overhead in __getitem__
        # The metadata contains relative paths in 'file_path'
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, rel_path)
            for rel_path in self.df["file_path"].values
        ]

        # Extract targets or IDs depending on mode
        if self.mode in ["train", "val"]:
            self.labels = self.df["cancer"].values.astype(np.float32)
        elif self.mode == "test":
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image using utility function
        # Returns float32 numpy array in [0, 1], shape (H, W)
        img_path = self.file_paths[idx]
        try:
            img = load_dicom_image(img_path, img_size=Config.IMG_SIZE)
        except Exception as e:
            # Fallback for corrupt images (though validation script checked existence)
            # Return a black image to prevent crashing
            print(f"Warning: Failed to load {img_path}. Error: {e}")
            img = np.zeros(Config.IMG_SIZE, dtype=np.float32)

        # Convert to Tensor (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Replicate to 3 channels for ResNet (3, H, W)
        img_tensor = img_tensor.repeat(3, 1, 1)

        # Apply transforms (Augmentations + Normalization)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.mode in ["train", "val"]:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label
        else:
            pred_id = self.prediction_ids[idx]
            return img_tensor, pred_id


def get_dataloaders():
    """
    Factory function to create DataLoaders for train, val, and test.
    Implements Balanced Batch Sampling for the training set.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # --------------------------------------------------------------------------
    # 1. Define Transforms
    # --------------------------------------------------------------------------
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Training transforms: Geometric augmentations + Normalization
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            normalize,
        ]
    )

    # Validation/Test transforms: Normalization only
    eval_transform = transforms.Compose([normalize])

    # --------------------------------------------------------------------------
    # 2. Create Datasets
    # --------------------------------------------------------------------------
    train_dataset = BreastCancerDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        mode="train",
        transform=train_transform,
    )

    val_dataset = BreastCancerDataset(
        metadata_path=Config.VAL_METADATA_PATH, mode="val", transform=eval_transform
    )

    test_dataset = BreastCancerDataset(
        metadata_path=Config.TEST_METADATA_PATH, mode="test", transform=eval_transform
    )

    # --------------------------------------------------------------------------
    # 3. Configure Balanced Batch Sampling for Training
    # --------------------------------------------------------------------------
    # Extract labels to compute weights
    train_labels = train_dataset.labels

    # Count classes
    num_pos = int(np.sum(train_labels))
    num_neg = len(train_labels) - num_pos

    # Avoid division by zero
    if num_pos == 0:
        print(
            "Warning: No positive samples in training set. Disabling weighted sampling."
        )
        sampler = None
    else:
        # Calculate weights based on Config.POSITIVE_SAMPLING_RATIO
        # We want the probability of picking a positive sample to be POSITIVE_SAMPLING_RATIO
        # Weight_pos * num_pos / Total_Weight = POSITIVE_SAMPLING_RATIO
        # A simple way is to assign weights inversely proportional to counts,
        # adjusted by the desired ratio.

        # Target probability per class
        prob_pos = Config.POSITIVE_SAMPLING_RATIO
        prob_neg = 1.0 - prob_pos

        # Weight per sample
        weight_pos = prob_pos / num_pos
        weight_neg = prob_neg / num_neg

        # Create weight array for all samples
        weights = np.where(train_labels == 1, weight_pos, weight_neg)
        weights = torch.tensor(weights, dtype=torch.double)

        # Create sampler
        # num_samples=len(train_dataset) ensures the epoch size remains the same
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_dataset), replacement=True
        )

    # --------------------------------------------------------------------------
    # 4. Create DataLoaders
    # --------------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),  # Shuffle only if sampler is not used
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
