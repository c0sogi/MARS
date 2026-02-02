import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library import config

# Standard ImageNet Normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class MultiViewDataset(Dataset):
    """
    Dataset that returns three distinct geometric views of the same image:
    1. Global (Squish)
    2. Standard (Resize + Crop)
    3. Local (Zoom + Crop)
    """

    def __init__(
        self,
        metadata_path,
        class_to_idx=None,
        is_test=False,
        transform_global=None,
        transform_standard=None,
        transform_local=None,
        debug=False,
    ):
        self.df = pd.read_csv(metadata_path)
        self.root_dir = config.INPUT_DIR
        self.is_test = is_test
        self.class_to_idx = class_to_idx

        # Debug: Subsample dataset
        if debug:
            # Cite debug_lesson_8: Ensure Synthetic Data Cardinality Satisfies Cross-Validation Constraints
            if not self.is_test and "breed" in self.df.columns:
                top_breeds = self.df["breed"].value_counts().head(5).index
                self.df = self.df[self.df["breed"].isin(top_breeds)]
            self.df = self.df.iloc[:100].copy()

        # Assign transforms
        self.transform_global = transform_global
        self.transform_standard = transform_standard
        self.transform_local = transform_local

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image and convert to RGB (handles grayscale/RGBA)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately
            # For this task, we assume data integrity based on metadata checks
            image = Image.new("RGB", (224, 224))

        # Apply the three specific views
        # Note: Transforms include ToTensor and Normalize
        view_global = self.transform_global(image)
        view_standard = self.transform_standard(image)
        view_local = self.transform_local(image)

        sample = {
            "id": row["id"],
            "global": view_global,
            "standard": view_standard,
            "local": view_local,
        }

        # Add label for training/validation
        if not self.is_test:
            breed = row["breed"]
            if self.class_to_idx is not None:
                label = self.class_to_idx[breed]
                sample["label"] = torch.tensor(label, dtype=torch.long)

        return sample


def get_transforms():
    """
    Creates the three transformation pipelines using Bicubic interpolation.
    """
    # Common normalization
    norm = transforms.Normalize(mean=MEAN, std=STD)

    # 1. Global View: Squish to 224x224
    t_global = transforms.Compose(
        [
            transforms.Resize(
                config.VIEW_GLOBAL_SIZE, interpolation=config.INTERPOLATION
            ),
            transforms.ToTensor(),
            norm,
        ]
    )

    # 2. Standard View: Resize 232 -> CenterCrop 224
    t_standard = transforms.Compose(
        [
            transforms.Resize(
                config.VIEW_STANDARD_RESIZE, interpolation=config.INTERPOLATION
            ),
            transforms.CenterCrop(config.VIEW_STANDARD_CROP),
            transforms.ToTensor(),
            norm,
        ]
    )

    # 3. Local View: Resize 288 -> CenterCrop 224
    t_local = transforms.Compose(
        [
            transforms.Resize(
                config.VIEW_LOCAL_RESIZE, interpolation=config.INTERPOLATION
            ),
            transforms.CenterCrop(config.VIEW_LOCAL_CROP),
            transforms.ToTensor(),
            norm,
        ]
    )

    return t_global, t_standard, t_local


def get_dataloaders(debug=False):
    """
    Initializes Datasets and DataLoaders for Train, Val, and Test.
    Returns: train_loader, val_loader, test_loader, class_to_idx
    """
    # 1. Generate Class Mapping from Training Data
    # Ensure deterministic mapping by sorting unique breeds
    train_meta_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    classes = sorted(train_meta_df["breed"].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # 2. Get Transformation Pipelines
    t_global, t_standard, t_local = get_transforms()

    # 3. Create Datasets
    train_dataset = MultiViewDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        class_to_idx=class_to_idx,
        is_test=False,
        transform_global=t_global,
        transform_standard=t_standard,
        transform_local=t_local,
        debug=debug,
    )

    val_dataset = MultiViewDataset(
        metadata_path=config.VAL_METADATA_PATH,
        class_to_idx=class_to_idx,
        is_test=False,
        transform_global=t_global,
        transform_standard=t_standard,
        transform_local=t_local,
        debug=debug,
    )

    test_dataset = MultiViewDataset(
        metadata_path=config.TEST_METADATA_PATH,
        class_to_idx=None,
        is_test=True,
        transform_global=t_global,
        transform_standard=t_standard,
        transform_local=t_local,
        debug=debug,
    )

    # 4. Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_to_idx
