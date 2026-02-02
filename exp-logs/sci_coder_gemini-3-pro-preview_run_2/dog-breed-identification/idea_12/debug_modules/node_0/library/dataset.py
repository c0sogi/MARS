import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from library.config import Config
from library.transforms import get_stream_transforms


class DogDataset(Dataset):
    """
    A PyTorch Dataset for loading dog images and generating multiple views
    (Global, Standard, Local) for the Dual-Stream Heterogeneous Ensemble.
    """

    def __init__(
        self, metadata, transforms, input_dir=Config.INPUT_DIR, return_label=True
    ):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing 'id', 'file_path', and optionally 'breed'.
            transforms (dict): Dictionary of transform pipelines (e.g., {'global': ..., 'standard': ...}).
            input_dir (str): Root directory containing the images (e.g., './input').
            return_label (bool): Whether to return the label in the sample dictionary.
        """
        self.metadata = metadata
        self.transforms = transforms
        self.input_dir = input_dir
        self.return_label = return_label

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["id"]

        # Construct the full file path
        # metadata 'file_path' is relative (e.g., 'train/xxx.jpg')
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image and convert to RGB (standard for torchvision models)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # In a strict pipeline, we might want to fail hard or log this.
            # Assuming data integrity is verified by metadata generation.
            raise IOError(f"Failed to load image at {img_path}: {e}")

        # Apply transforms to generate multiple views
        views = {}
        if self.transforms:
            for view_name, transform_pipeline in self.transforms.items():
                views[view_name] = transform_pipeline(image)

        sample = {"id": img_id, "views": views}

        # Handle labels
        if self.return_label:
            # Use .get() to handle cases where 'breed' column might be missing (e.g. if misconfigured)
            # or if it's a test set loaded with return_label=True accidentally.
            label = row.get("breed", None)
            sample["label"] = label

        return sample


def get_dataset(split, stream_config):
    """
    Factory function to create a DogDataset for a specific split and stream configuration.

    Args:
        split (str): One of 'train', 'val', 'test'.
        stream_config (dict): Configuration dictionary for the specific stream (A or B).

    Returns:
        DogDataset: The configured dataset instance.
    """
    # Determine configuration based on split
    if split == "train":
        csv_path = Config.TRAIN_METADATA
        return_label = True
    elif split == "val":
        csv_path = Config.VAL_METADATA
        return_label = True
    elif split == "test":
        csv_path = Config.TEST_METADATA
        return_label = False
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    # Verify metadata exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")

    # Load Metadata
    df = pd.read_csv(csv_path)

    # Generate Transforms
    # We use is_train=False to ensure deterministic transforms (Resize/CenterCrop)
    # are used. TTA (flipping) is handled externally in the feature extraction loop.
    transforms = get_stream_transforms(stream_config, is_train=False)

    # Instantiate Dataset
    dataset = DogDataset(
        metadata=df,
        transforms=transforms,
        input_dir=Config.INPUT_DIR,
        return_label=return_label,
    )

    return dataset
