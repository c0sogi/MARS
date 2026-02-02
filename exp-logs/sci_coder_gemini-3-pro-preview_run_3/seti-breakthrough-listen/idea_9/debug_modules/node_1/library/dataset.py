import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(data: str = "train"):
    """
    Returns the Albumentations transformations for the specified mode.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),  # Time reversal
                A.VerticalFlip(p=0.5),  # Frequency inversion
                ToTensorV2(),  # Convert to Tensor and HWC -> CHW
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class SETIDataset(Dataset):
    """
    Custom Dataset for SETI Technosignature Detection.
    Loads .npy spectrogram files, applies padding and augmentation,
    and organizes channels for Siamese network input.
    """

    def __init__(self, df, transform=None, input_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, target, file_path).
            transform (A.Compose): Albumentations transforms.
            input_dir (str): Root directory for input files.
        """
        self.df = df
        self.transform = transform
        self.input_dir = input_dir

        # Pre-calculate padding amount
        # Pad frequency dimension (Height) from 273 to 288
        self.pad_h = Config.IMG_HEIGHT - Config.ORIG_HEIGHT

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative, e.g., "train/0/xxxx.npy"
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load spectrogram
        # Original Shape: (6, 273, 256) -> (Cadence, Freq, Time)
        try:
            image = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for robustness (should not happen with verified metadata)
            image = np.zeros(
                (6, Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.float32
            )

        # Prepare for Albumentations (H, W, C)
        # Transpose: (6, 273, 256) -> (273, 256, 6)
        image = np.transpose(image, (1, 2, 0))

        # Pad Frequency dimension (Height)
        # We pad at the end of the frequency axis (axis 0 in HWC)
        if self.pad_h > 0:
            # ((top, bottom), (left, right), (channels_start, channels_end))
            image = np.pad(
                image,
                ((0, self.pad_h), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Apply Transformations
        # This handles augmentation and conversion to Tensor (CHW)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Shape: (6, 288, 256)
        else:
            # Fallback if no transform provided
            # Convert to tensor and transpose back to CHW
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Reorder Channels for Siamese Input
        # Current Order: 0(A), 1(B), 2(A), 3(C), 4(A), 5(D)
        # Desired Order: [On-Target (Signal), Off-Target (Reference)]
        # On-Target: 0, 2, 4
        # Off-Target: 1, 3, 5
        # Resulting Tensor: [A, A, A, B, C, D]

        # We return a single 6-channel tensor to allow mixup_data to work correctly.
        # The model is responsible for splitting this into two 3-channel streams.
        indices = torch.tensor([0, 2, 4, 1, 3, 5], dtype=torch.long)
        image = torch.index_select(image, 0, indices)

        # Get Target
        target = torch.tensor(row["target"], dtype=torch.float32)

        return image, target
