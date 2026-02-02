import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import library.config as config


def get_class_mapping():
    """
    Generates a dictionary mapping breed names to integer indices
    based on the training metadata.

    Returns:
        dict: A dictionary where keys are breed names (str) and values are indices (int).
    """
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {config.TRAIN_METADATA_PATH}"
        )

    df = pd.read_csv(config.TRAIN_METADATA_PATH)
    # Sort breeds alphabetically to ensure deterministic mapping across runs
    unique_breeds = sorted(df["breed"].unique().tolist())
    class_to_idx = {breed: i for i, breed in enumerate(unique_breeds)}
    return class_to_idx


class MultiViewDataset(Dataset):
    """
    Dataset class that loads images and applies three distinct geometric transforms
    (Global, Standard, Local) for the multi-view pipeline.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transforms_dict: dict,
        class_to_idx: dict = None,
        is_test: bool = False,
    ):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'id' and 'file_path' columns.
                                      Must contain 'breed' if is_test is False.
            transforms_dict (dict): Dictionary containing 'global', 'standard', and 'local' transforms.
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required if is_test is False.
            is_test (bool): Flag indicating if this is the test set (no labels).
        """
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms_dict
        self.class_to_idx = class_to_idx
        self.is_test = is_test
        self.input_dir = config.INPUT_DIR

        # Validation
        if not self.is_test:
            if self.class_to_idx is None:
                raise ValueError(
                    "class_to_idx must be provided for training/validation sets."
                )
            if "breed" not in self.df.columns:
                raise ValueError(
                    "DataFrame must contain 'breed' column for training/validation sets."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        rel_path = row["file_path"]

        # Construct full image path
        img_path = os.path.join(self.input_dir, rel_path)

        # Load image (RGB)
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError) as e:
            raise FileNotFoundError(f"Error loading image {img_path}: {e}")

        # Apply the three specific views
        # transforms_dict is expected to have keys: 'global', 'standard', 'local'
        view_global = self.transforms["global"](image)
        view_standard = self.transforms["standard"](image)
        view_local = self.transforms["local"](image)

        sample = {
            "id": img_id,
            "view_global": view_global,
            "view_standard": view_standard,
            "view_local": view_local,
        }

        # Handle Label
        if not self.is_test:
            breed = row["breed"]
            # Map string label to integer index
            label_idx = self.class_to_idx[breed]
            sample["label"] = torch.tensor(label_idx, dtype=torch.long)
        else:
            # Return -1 for test samples as a placeholder
            sample["label"] = torch.tensor(-1, dtype=torch.long)

        return sample
