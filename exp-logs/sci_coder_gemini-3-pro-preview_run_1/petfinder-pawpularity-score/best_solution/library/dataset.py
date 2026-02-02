import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pawpularity Contest.

    Handles loading of images, processing with Hugging Face processors,
    and extraction of binary metadata features and targets.
    Supports returning flipped images for Feature-Space Augmentation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor,
        root_dir: str = Config.INPUT_DIR,
        is_train: bool = True,
        return_flipped: bool = False,
    ):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing metadata (Id, features, target, file_path).
            processor: Hugging Face AutoImageProcessor or similar callable.
            root_dir (str): Root directory for image files.
            is_train (bool): If True, returns the target variable.
            return_flipped (bool): If True, returns a flipped version of the image for TTA/Feature Averaging.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.root_dir = root_dir
        self.is_train = is_train
        self.return_flipped = return_flipped

        # Binary metadata feature columns as defined in the dataset description
        self.meta_cols = [
            "Subject Focus",
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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # 1. Load Image
        # The 'file_path' column contains relative paths like "train/{id}.jpg"
        img_path = os.path.join(self.root_dir, row["file_path"])

        try:
            # Load and convert to RGB (standard for HF models)
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError) as e:
            # Fallback for robustness (though analysis showed no missing files)
            # Create a blank black image
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 2. Process Original Image
        # The processor returns a dict with 'pixel_values'
        # return_tensors="pt" gives PyTorch tensors
        inputs = self.processor(images=image, return_tensors="pt")
        # Squeeze the batch dimension (1, C, H, W) -> (C, H, W)
        pixel_values = inputs["pixel_values"].squeeze(0)

        item = {
            "pixel_values": pixel_values,
            "Id": row["Id"],
            # Extract binary metadata as a float tensor
            "features": torch.tensor(
                row[self.meta_cols].values.astype("float32"), dtype=torch.float32
            ),
        }

        # 3. Process Flipped Image (Optional)
        # Used for Feature-Space Augmentation Averaging
        if self.return_flipped:
            image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            inputs_flipped = self.processor(images=image_flipped, return_tensors="pt")
            item["pixel_values_flipped"] = inputs_flipped["pixel_values"].squeeze(0)

        # 4. Extract Target (if training)
        if self.is_train:
            # Target is a float value (1-100)
            item["label"] = torch.tensor(row["Pawpularity"], dtype=torch.float32)

        return item


def get_dataloader(
    dataframe: pd.DataFrame,
    processor,
    batch_size: int,
    is_train: bool = True,
    return_flipped: bool = False,
    num_workers: int = Config.NUM_WORKERS,
    shuffle: bool = False,
):
    """
    Helper function to create a DataLoader for the PawpularityDataset.

    Args:
        dataframe (pd.DataFrame): Data.
        processor: Image processor.
        batch_size (int): Batch size.
        is_train (bool): Whether to include targets.
        return_flipped (bool): Whether to include flipped images.
        num_workers (int): Number of worker threads.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = PawpularityDataset(
        dataframe=dataframe,
        processor=processor,
        is_train=is_train,
        return_flipped=return_flipped,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    return dataloader
