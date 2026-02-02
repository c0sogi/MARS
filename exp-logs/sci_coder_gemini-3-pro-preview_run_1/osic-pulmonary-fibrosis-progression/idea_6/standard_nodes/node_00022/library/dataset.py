import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class LungDataset(Dataset):
    """
    PyTorch Dataset for the TQ-SAN model.

    This dataset handles the loading of dual-view (Axial and Coronal) Tri-Slab CT images
    stored in cached .npy files. It applies spatial augmentations during training and
    prepares the tabular query vector and regression targets.
    """

    def __init__(self, data, mode="train"):
        """
        Args:
            data (dict): Dictionary returned by DataUtils.prepare_dataset containing:
                         - img_paths: List of paths to cached .npy files
                         - meta: Tensor of tabular features (Age, Sex, Smoke, Percent)
                         - targets: Tensor of FVC values (Target)
                         - time_delta: Tensor of week offsets (Time)
                         - base_fvc: Tensor of baseline FVC (Prior)
            mode (str): 'train', 'val', or 'test'. Controls augmentation behavior.
        """
        self.img_paths = data["img_paths"]
        self.meta = data["meta"]
        self.targets = data["targets"]
        self.time_delta = data["time_delta"]
        self.base_fvc = data["base_fvc"]
        self.mode = mode

        # ImageNet Normalization Constants
        # Inputs are float32 in range [0, 1], so we use these stats directly.
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # Define Augmentation Pipeline
        if mode == "train":
            self.transform = A.Compose(
                [
                    # Spatial Augmentations: Flips, Shifts, Rotations
                    # Applied to regularize the small dataset without altering density (HU) info
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=0,  # Constant padding (black)
                        value=0,
                    ),
                    # Normalization & Conversion to Tensor (C, H, W)
                    A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                    ToTensorV2(),
                ]
            )
        else:
            # Validation/Test: Normalize only
            self.transform = A.Compose(
                [
                    A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Image Data
        # The .npy file contains a dict: {'axial': np.array, 'coronal': np.array}
        # Arrays are (224, 224, 3) float32 in [0, 1]
        try:
            img_data = np.load(self.img_paths[idx], allow_pickle=True).item()
            img_axial = img_data["axial"]
            img_coronal = img_data["coronal"]
        except Exception as e:
            # Fallback for potentially corrupt files (safety net)
            # Returns black images to prevent crashing
            img_axial = np.zeros(
                (Config.img_size, Config.img_size, 3), dtype=np.float32
            )
            img_coronal = np.zeros(
                (Config.img_size, Config.img_size, 3), dtype=np.float32
            )

        # 2. Apply Transforms
        # We augment views independently. While they represent the same patient,
        # they are distinct spatial projections, so independent noise/shift is valid regularization.
        augmented_axial = self.transform(image=img_axial)["image"]
        augmented_coronal = self.transform(image=img_coronal)["image"]

        # 3. Retrieve Tabular & Target Data
        meta_features = self.meta[idx]
        target = self.targets[idx]
        dt = self.time_delta[idx]
        base = self.base_fvc[idx]

        # 4. Return Dictionary
        return {
            "axial": augmented_axial,  # (3, 224, 224)
            "coronal": augmented_coronal,  # (3, 224, 224)
            "meta": meta_features,  # (4,)
            "target": target,  # (1,)
            "dt": dt,  # (1,)
            "base_fvc": base,  # (1,)
        }
