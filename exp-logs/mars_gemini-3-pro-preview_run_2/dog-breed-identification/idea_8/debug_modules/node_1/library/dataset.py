import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import library.config as config
import library.transforms as lib_transforms


class DogDataset(Dataset):
    """
    A PyTorch Dataset for loading dog images with specific multi-view transformations.

    Args:
        metadata_path (str): Path to the metadata CSV file (train, val, or test).
        transform_type (str): The type of transform to apply ('global', 'standard', 'local').
        class_to_idx (dict, optional): Dictionary mapping breed names to integers.
                                       Required for validation set to ensure consistency with training.
        is_test (bool): If True, returns (image, id) and ignores labels.
        debug (bool): If True, limits the dataset to a small subset for debugging.
    """

    def __init__(
        self,
        metadata_path,
        transform_type="standard",
        class_to_idx=None,
        is_test=False,
        debug=False,
    ):
        self.metadata_path = metadata_path
        self.transform_type = transform_type
        self.is_test = is_test
        self.debug = debug

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Handle Debugging
        if self.debug:
            self.df = self.df.iloc[: config.DEBUG_SAMPLE_SIZE].copy()

        # Initialize Transforms
        if self.transform_type == "global":
            self.transform = lib_transforms.get_global_transform()
        elif self.transform_type == "standard":
            self.transform = lib_transforms.get_standard_transform()
        elif self.transform_type == "local":
            self.transform = lib_transforms.get_local_transform()
        else:
            raise ValueError(
                f"Invalid transform_type '{transform_type}'. Must be 'global', 'standard', or 'local'."
            )

        # Initialize Label Encoding
        if not self.is_test:
            if class_to_idx is not None:
                self.class_to_idx = class_to_idx
            else:
                # Create mapping from the data (sorted for determinism)
                unique_breeds = sorted(self.df["breed"].unique())
                self.class_to_idx = {
                    breed: idx for idx, breed in enumerate(unique_breeds)
                }
        else:
            self.class_to_idx = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains relative path e.g., 'train/id.jpg'
        # config.INPUT_DIR is './input'
        img_path = os.path.join(config.INPUT_DIR, row["file_path"])

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately
            # For this task, we assume data integrity based on metadata checks
            raise e

        # Apply Transform
        # Returns Tensor (C, H, W) for global/standard
        # Returns Tensor (5, C, H, W) for local
        image_tensor = self.transform(image)

        if self.is_test:
            # Return image and ID for submission generation
            return image_tensor, row["id"]
        else:
            # Return image and label index for training/evaluation
            breed = row["breed"]
            label_idx = self.class_to_idx[breed]
            return image_tensor, label_idx

    def get_class_to_idx(self):
        """Returns the breed-to-index mapping."""
        return self.class_to_idx
