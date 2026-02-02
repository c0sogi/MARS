import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train", img_size=(224, 224)):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (tuple): Target (height, width).

    Returns:
        A.Compose: The composition of transforms.
    """
    # Standard ImageNet normalization statistics
    # These are applied after the image is scaled to [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms_list = [
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
    ]

    if mode == "train":
        # Horizontal Flip (Frequency Reversal) makes the model robust to
        # positive/negative Doppler drift slopes.
        transforms_list.append(A.HorizontalFlip(p=0.5))

    transforms_list.extend(
        [
            # Normalize expects input in [0, 1] if max_pixel_value=1.0
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )

    return A.Compose(transforms_list)


class CadenceDataset(Dataset):
    """
    PyTorch Dataset for loading and processing Cadence Snippets.

    Splits the 6-channel input into two 3-channel streams:
    - On-Target (A observations): Indices 0, 2, 4
    - Off-Target (B, C, D observations): Indices 1, 3, 5
    """

    def __init__(self, metadata_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
                                             If None, default transforms are generated.
        """
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)

        # Use default transforms if none provided
        if transform is None:
            self.transform = get_transforms(mode=mode, img_size=Config.IMG_SIZE)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_id = str(row["id"])
        target = float(row["target"])

        # Construct full file path
        # metadata 'file_path' is relative to input dir (e.g., "train/0/000....npy")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load data
        # Shape: (6, 273, 256) -> (Cadence Position, Frequency, Time)
        try:
            img = np.load(full_path).astype(np.float32)
        except FileNotFoundError:
            # Fallback for robustness (should not happen given verification)
            # Create a zero array of expected shape
            img = np.zeros((6, 273, 256), dtype=np.float32)

        # --- Preprocessing ---

        # 1. Min-Max Scale to [0, 1] per snippet
        # This preserves the relative intensity of the signal vs noise within the snippet
        # while mapping it to the range expected by ImageNet normalization.
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # 2. Split into On-Target and Off-Target streams
        # On-Target: A, A, A (indices 0, 2, 4)
        # Off-Target: B, C, D (indices 1, 3, 5)
        on_target = img[[0, 2, 4], :, :]  # Shape: (3, 273, 256)
        off_target = img[[1, 3, 5], :, :]  # Shape: (3, 273, 256)

        # 3. Transpose to (Height, Width, Channels) for Albumentations
        # Resulting Shape: (273, 256, 3)
        on_target = np.transpose(on_target, (1, 2, 0))
        off_target = np.transpose(off_target, (1, 2, 0))

        # 4. Apply Transforms (Resize, Normalize, ToTensor)
        # We must apply the SAME geometric transforms to both streams if random
        # (e.g., flip) to maintain correspondence.
        # Albumentations 'replay' or 'additional_targets' can handle this,
        # but here we can just pass them as a dictionary if using additional_targets
        # or rely on the seed if we were strict.
        # However, simpler approach for dual-stream with Albumentations:
        # Use the dictionary interface.

        data = {"image": on_target, "image_off": off_target}
        # We need to re-create the transform with additional targets logic
        # or apply deterministically.
        # A robust way without complex A.Compose setup is to use the seed or
        # simply apply non-random transforms separately and random ones carefully.
        # Given we only use HorizontalFlip, let's use the dictionary target approach.

        # We need to extend the transform to accept 'image_off'
        # But standard A.Compose expects 'image'.
        # Let's do it manually to ensure sync:

        if self.mode == "train":
            # Apply random horizontal flip synchronously
            if np.random.rand() < 0.5:
                on_target = np.fliplr(on_target)
                off_target = np.fliplr(off_target)
                # Note: fliplr flips axis 1 (Width/Time), which is what A.HorizontalFlip does.
                # Arrays are (273, 256, 3), axis 1 is 256.
                # Since numpy array is contiguous, this is fine.
                on_target = np.ascontiguousarray(on_target)
                off_target = np.ascontiguousarray(off_target)

        # Apply deterministic parts (Resize, Normalize, ToTensor)
        # We can reuse the transform pipeline but we must ensure it doesn't apply
        # random flips again if we handled it manually, OR we configure the transform
        # to handle dual inputs.
        # To keep it simple and safe with the provided `get_transforms` which includes Flip:
        # We will use the `additional_targets` feature of Albumentations if we modify `get_transforms`,
        # OR we just apply the transform to `on_target` and `off_target` separately
        # IF the transform was deterministic.
        # BUT `get_transforms` has random flip.

        # REVISION: To ensure perfect sync of random transforms without modifying `get_transforms` signature
        # significantly, we use the ReplayCompose or simply apply the random flip manually (as done above)
        # and use a deterministic transform for the rest.

        # Let's use a deterministic transform for the final steps
        # We create a temporary deterministic transform for the actual resizing/norm
        det_transform = A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )

        res_on = det_transform(image=on_target)["image"]
        res_off = det_transform(image=off_target)["image"]

        return {
            "on_input": res_on,
            "off_input": res_off,
            "target": torch.tensor(target, dtype=torch.float32),
            "id": file_id,
        }
