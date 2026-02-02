import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_dicom_stack, create_25d_stack, apply_windowing


class FractureDataset(Dataset):
    """
    Dataset for Cervical Spine Fracture Detection.
    Loads DICOM volumes, performs uniform subsampling, creates 2.5D stacks,
    and applies preprocessing (windowing, resizing).
    """

    def __init__(self, df, transforms=None, mode="train", load_cached_data=False):
        """
        Args:
            df (pd.DataFrame): Metadata DataFrame containing StudyInstanceUID and labels.
            transforms (callable, optional): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files for volumes.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.load_cached_data = load_cached_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path to the image directory
        # Metadata contains relative paths (e.g., "train_images/UID")
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Load the 3D volume (Depth, Height, Width)
        # This function handles caching internally as per requirements
        volume = load_dicom_stack(
            path=image_dir,
            load_cached_data=self.load_cached_data,
            cache_dir=Config.CACHE_DIR,
        )

        processed_slices = []

        # Handle case where volume loading fails or is empty
        if volume.shape[0] > 0:
            depth = volume.shape[0]

            # Uniformly subsample indices across the volume depth
            # If depth < NUM_SLICES, indices will be repeated or closely spaced
            indices = np.linspace(0, depth - 1, Config.NUM_SLICES).astype(int)

            for i in indices:
                # 1. Create 2.5D Stack: (3, H, W)
                # Stacks z-1, z, z+1
                stack = create_25d_stack(volume, i)

                # 2. Apply Bone Windowing: (3, H, W) -> [0, 1] float32
                stack = apply_windowing(
                    stack, Config.WINDOW_CENTER, Config.WINDOW_WIDTH
                )

                # 3. Resize to target size
                # cv2.resize expects (H, W, C), so we transpose
                stack = np.moveaxis(stack, 0, -1)  # (H, W, 3)
                stack = cv2.resize(stack, (Config.IMG_SIZE, Config.IMG_SIZE))

                # 4. Apply Augmentations (if any)
                if self.transforms:
                    # Albumentations expects (H, W, C)
                    augmented = self.transforms(image=stack)["image"]
                    stack = augmented

                # Transpose back to (C, H, W) for PyTorch
                stack = np.moveaxis(stack, -1, 0)

                processed_slices.append(stack)

            # Stack all slices into a single tensor: (NUM_SLICES, 3, H, W)
            images = np.stack(processed_slices)
        else:
            # Return zero tensor if data is missing
            images = np.zeros(
                (Config.NUM_SLICES, 3, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=np.float32,
            )

        # Convert to PyTorch tensor
        images = torch.from_numpy(images).float()

        # Prepare Targets
        if self.mode in ["train", "val"]:
            # Extract C1-C7 labels and patient_overall
            target_cols = Config.TARGET_COLS + [Config.OVERALL_COL]
            labels = row[target_cols].values.astype(np.float32)
            labels = torch.tensor(labels)
            return images, labels
        else:
            # Test mode: Return dummy targets
            return images, torch.zeros(len(Config.TARGET_COLS))
