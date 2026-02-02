import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import get_weight
import library.config as config


def get_class_mapping():
    """
    Generates a consistent label-to-integer mapping based on the master labels file.
    """
    labels_path = os.path.join(config.INPUT_DIR, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    df = pd.read_csv(labels_path)
    unique_breeds = sorted(df["breed"].unique())
    label_map = {breed: idx for idx, breed in enumerate(unique_breeds)}
    return label_map


def get_model_transforms(weights_name):
    """
    Generates the 3-view transform pipeline specific to the model weights.
    Extracts mean, std, and interpolation from the pretrained weights metadata.
    """
    try:
        weights = get_weight(weights_name)
        auto_transforms = weights.transforms()

        # Extract specific preprocessing parameters
        mean = auto_transforms.mean
        std = auto_transforms.std
        interpolation = auto_transforms.interpolation
    except Exception as e:
        print(f"Error loading weights {weights_name}: {e}")
        # Fallback to standard ImageNet stats if specific weights fail
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        interpolation = transforms.InterpolationMode.BICUBIC

    normalize = transforms.Normalize(mean=mean, std=std)

    # Determine input resolution
    # Cite debug_lesson_3: Enforce Global Output Schemas (Input Schema consistency)
    if "SWAG" in weights_name:
        # SWAG weights require 512x512 resolution
        s_resize, s_crop = 512, 512
        g_size = 512
        l_resize, l_crop = 512, 512
    else:
        s_resize, s_crop = config.VIEW_STANDARD_RESIZE, config.VIEW_STANDARD_CROP
        g_size = config.VIEW_GLOBAL_SIZE
        l_resize, l_crop = config.VIEW_LOCAL_RESIZE, config.VIEW_LOCAL_CROP

    # View 1: Standard (Resize -> CenterCrop)
    # Captures the main object with some context
    view_standard = transforms.Compose(
        [
            transforms.Resize(s_resize, interpolation=interpolation),
            transforms.CenterCrop(s_crop),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # View 2: Global (Squish)
    # Captures the entire image structure, distorting aspect ratio if necessary
    view_global = transforms.Compose(
        [
            transforms.Resize(
                (g_size, g_size),
                interpolation=interpolation,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # View 3: Local (Resize Large -> CenterCrop)
    # Captures fine-grained texture details by zooming in
    view_local = transforms.Compose(
        [
            transforms.Resize(l_resize, interpolation=interpolation),
            transforms.CenterCrop(l_crop),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {"standard": view_standard, "global": view_global, "local": view_local}


class MultiViewDataset(Dataset):
    """
    Dataset that returns three geometric views of each image.
    """

    def __init__(self, csv_path, transform_dict, label_map=None, debug=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV (train/val/test).
            transform_dict (dict): Dictionary of transforms for 'standard', 'global', 'local' views.
            label_map (dict, optional): Mapping from breed string to integer index.
            debug (bool): If True, limits dataset size for debugging.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        if debug:
            self.df = self.df.iloc[: config.DEBUG_DATASET_SIZE]

        self.transform_dict = transform_dict
        self.label_map = label_map
        self.input_dir = config.INPUT_DIR

        # Determine if this is a labeled dataset
        self.has_labels = "breed" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains relative path (e.g., "train/id.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        try:
            # Load image and convert to RGB (standard for torchvision)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # In a strict setting, we might raise, but here we'll let it fail loudly
            raise e

        # Apply transforms for each view
        views = {}
        for view_name, transform in self.transform_dict.items():
            views[view_name] = transform(image)

        # Process label
        label = -1
        if self.has_labels:
            breed = row["breed"]
            if self.label_map:
                label = self.label_map.get(breed, -1)
            else:
                # If no map provided but labels exist, we can't return a valid int
                label = -1

        return views, label, row["id"]


def get_dataloader(
    csv_path, model_weights, batch_size=None, shuffle=False, debug=False
):
    """
    Factory function to create a DataLoader with the correct transforms.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # 1. Get Class Mapping
    label_map = get_class_mapping()

    # 2. Get Model-Specific Transforms
    transform_dict = get_model_transforms(model_weights)

    # 3. Create Dataset
    dataset = MultiViewDataset(
        csv_path=csv_path,
        transform_dict=transform_dict,
        label_map=label_map,
        debug=debug,
    )

    # 4. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return loader
