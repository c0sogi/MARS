import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config


def get_class_names():
    """
    Reads the training metadata to retrieve the list of unique class names (breeds).

    Returns:
        list: Sorted list of unique breed names.
    """
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    return sorted(df["breed"].unique().tolist())


def get_stream_transforms(stream_name):
    """
    Generates the dictionary of transforms for the specified stream.

    Args:
        stream_name (str): 'stream_a' (ConvNeXt) or 'stream_b' (DINOv2).

    Returns:
        dict: A dictionary where keys are view names ('global', 'standard', 'local')
              and values are torchvision.transforms.Compose objects.
    """
    # ImageNet Statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Determine Interpolation mode
    if stream_name == "stream_a":
        # ConvNeXt / Supervised: Standard Bilinear
        interpolation = transforms.InterpolationMode.BILINEAR
    elif stream_name == "stream_b":
        # DINOv2 / SSL: Bicubic is preferred for ViTs/DINO
        interpolation = transforms.InterpolationMode.BICUBIC
    else:
        raise ValueError(
            f"Unknown stream_name: {stream_name}. Must be 'stream_a' or 'stream_b'."
        )

    transform_dict = {}

    for view_cfg in Config.VIEWS:
        view_name = view_cfg["name"]
        resize_param = view_cfg["resize"]
        crop_param = view_cfg["crop"]

        t_list = []

        # 1. Resize
        # If resize_param is a tuple (h, w), it squishes.
        # If int, it resizes the shorter edge to int.
        t_list.append(transforms.Resize(resize_param, interpolation=interpolation))

        # 2. Center Crop (if defined)
        if crop_param is not None:
            t_list.append(transforms.CenterCrop(crop_param))

        # 3. ToTensor
        t_list.append(transforms.ToTensor())

        # 4. Normalize
        t_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

        transform_dict[view_name] = transforms.Compose(t_list)

    return transform_dict


class DogDataset(Dataset):
    """
    Dataset class for loading dog images and generating multiple views.
    """

    def __init__(
        self,
        metadata_df,
        transforms_dict,
        class_to_idx=None,
        input_dir=Config.INPUT_DIR,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id', 'file_path', and optionally 'breed'.
            transforms_dict (dict): Dictionary of transforms for each view.
            class_to_idx (dict, optional): Mapping from breed name to integer index.
            input_dir (str): Root directory for images.
        """
        self.df = metadata_df
        self.transforms_dict = transforms_dict
        self.class_to_idx = class_to_idx
        self.input_dir = input_dir

        # Pre-extract columns to arrays for faster access
        self.paths = self.df["file_path"].values
        self.ids = self.df["id"].values

        # Handle labels if they exist
        if "breed" in self.df.columns:
            self.labels = self.df["breed"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        rel_path = self.paths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load Image
        try:
            # Open image and convert to RGB (handles Grayscale/RGBA)
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            # In case of corruption, create a blank image (should not happen given metadata checks)
            print(f"Warning: Could not load image {full_path}. Error: {e}")
            image = Image.new("RGB", (224, 224))

        output = {"id": self.ids[idx]}

        # Apply transforms for each view
        # This generates 'global', 'standard', 'local' tensors
        for view_name, transform in self.transforms_dict.items():
            output[view_name] = transform(image)

        # Handle Label
        if self.labels is not None and self.class_to_idx is not None:
            label_str = self.labels[idx]
            label_idx = self.class_to_idx.get(label_str, -1)
            output["label"] = torch.tensor(label_idx, dtype=torch.long)
        else:
            # Return -1 for test set or if no mapping provided
            output["label"] = torch.tensor(-1, dtype=torch.long)

        return output


def get_data_loaders(
    stream_name, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates DataLoaders for Train, Validation, and Test sets for a specific stream.

    Args:
        stream_name (str): 'stream_a' or 'stream_b'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_to_idx)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Setup Class Mapping
    # Ensure consistent mapping based on sorted unique breeds in training set
    classes = sorted(train_df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # 3. Get Stream-Specific Transforms
    transforms_dict = get_stream_transforms(stream_name)

    # 4. Create Datasets
    # Note: We use the same transforms for Train/Val/Test here.
    # TTA (Horizontal Flip) is handled in the feature extraction loop as per strategy.

    train_dataset = DogDataset(
        metadata_df=train_df, transforms_dict=transforms_dict, class_to_idx=class_to_idx
    )

    val_dataset = DogDataset(
        metadata_df=val_df, transforms_dict=transforms_dict, class_to_idx=class_to_idx
    )

    test_dataset = DogDataset(
        metadata_df=test_df, transforms_dict=transforms_dict, class_to_idx=class_to_idx
    )

    # 5. Create DataLoaders
    # Shuffle only for training if we were training end-to-end,
    # but for feature extraction, order doesn't strictly matter unless we want to map back easily.
    # However, standard practice is shuffle=True for train.
    # For Val/Test, shuffle=False is critical to match IDs.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_to_idx
