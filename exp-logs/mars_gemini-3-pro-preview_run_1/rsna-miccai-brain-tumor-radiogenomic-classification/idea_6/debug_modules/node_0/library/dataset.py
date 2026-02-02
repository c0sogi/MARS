import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.image_processing import ImageProcessor


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the Multi-Planar 2.5D Holographic Network.
    Wraps the pre-processed numpy arrays of orthogonal views.
    """

    def __init__(self, data_dict, transform=None):
        """
        Args:
            data_dict (dict): Dictionary returned by ImageProcessor containing:
                              'ids', 'axial', 'coronal', 'sagittal', 'targets'.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.ids = data_dict["ids"]
        self.axial = data_dict["axial"]
        self.coronal = data_dict["coronal"]
        self.sagittal = data_dict["sagittal"]
        self.targets = data_dict.get("targets")
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve the 3 orthogonal views for the subject
        # Shape: (H, W, 3) - Float32 [0, 1]
        ax_img = self.axial[idx]
        cor_img = self.coronal[idx]
        sag_img = self.sagittal[idx]

        # Apply transformations
        # Note: Augmentations are applied independently to each view to prevent overfitting
        # and force the model to learn robust features from each perspective.
        if self.transform:
            ax_t = self.transform(image=ax_img)["image"]
            cor_t = self.transform(image=cor_img)["image"]
            sag_t = self.transform(image=sag_img)["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in standard pipeline)
            # Manually convert HWC -> CHW and to tensor
            ax_t = torch.from_numpy(ax_img.transpose(2, 0, 1))
            cor_t = torch.from_numpy(cor_img.transpose(2, 0, 1))
            sag_t = torch.from_numpy(sag_img.transpose(2, 0, 1))

        sample = {
            "axial": ax_t,
            "coronal": cor_t,
            "sagittal": sag_t,
            "BraTS21ID": self.ids[idx],
        }

        # Add target if available
        if self.targets is not None:
            # Binary classification target
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["label"] = label

        return sample


def get_transforms(split_name):
    """
    Returns the Albumentations transform pipeline for the given split.
    """
    if split_name == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # Non-rigid transformations as requested
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.5, border_mode=0
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5, border_mode=0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Just convert to Tensor (CHW)
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


def get_dataloader(
    df,
    split_name,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Creates a DataLoader for the specified dataset split.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        split_name (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, limits dataset size for debugging.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Initialize ImageProcessor
    # This handles the heavy lifting: DICOM reading, 3D ROI cropping, and Caching.
    processor = ImageProcessor(debug=debug)

    # Load or Process Data
    # This returns a dict with numpy arrays for the 3 views
    data_dict = processor.process_dataset(
        df, split_name=split_name, load_cached_data=load_cached_data
    )

    # Get Transforms
    transform = get_transforms(split_name)

    # Create Dataset
    dataset = BraTSDataset(data_dict, transform=transform)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
