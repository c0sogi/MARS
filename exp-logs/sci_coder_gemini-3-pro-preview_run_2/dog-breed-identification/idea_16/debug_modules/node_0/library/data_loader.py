import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import library.config as config

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_view_transforms():
    """
    Creates a dictionary of transformations for the defined views (global, standard, local).
    Enforces Bicubic interpolation and standard ImageNet normalization.
    """
    view_transforms = {}

    for view_name, settings in config.VIEWS.items():
        transform_list = []

        # 1. Resize
        # If resize is a tuple, it forces exact dimensions (squish)
        # If resize is an int, it resizes the shortest edge (aspect ratio preserved)
        transform_list.append(
            transforms.Resize(settings["resize"], interpolation=config.INTERPOLATION)
        )

        # 2. Crop (if specified)
        if settings["crop"] is not None:
            transform_list.append(transforms.CenterCrop(settings["crop"]))

        # 3. ToTensor and Normalize
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        view_transforms[view_name] = transforms.Compose(transform_list)

    return view_transforms


class DogDataset(Dataset):
    """
    Dataset class for loading dog images.
    Returns:
        image: Tensor
        target: str (breed name) or -1 if not available
        image_id: str
    """

    def __init__(self, metadata_df, transform=None, input_dir=config.INPUT_DIR):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id', 'file_path', and optionally 'breed'.
            transform (callable, optional): Optional transform to be applied on a sample.
            input_dir (str): Root directory where images are stored.
        """
        self.metadata = metadata_df
        self.transform = transform
        self.input_dir = input_dir

        # Check if labels exist
        self.has_labels = "breed" in metadata_df.columns

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Construct full file path
        # metadata 'file_path' is relative to input_dir (e.g., "train/id.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image and convert to RGB (handles grayscale or RGBA)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get ID
        image_id = row["id"]

        # Get Label
        if self.has_labels:
            target = row["breed"]
        else:
            target = -1  # Placeholder for test set

        return image, target, image_id
