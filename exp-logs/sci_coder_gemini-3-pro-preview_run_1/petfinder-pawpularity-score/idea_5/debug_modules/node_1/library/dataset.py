import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config

# =========================================================================
# Normalization Constants
# =========================================================================
# CLIP uses specific mean/std from its pre-training
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ImageNet mean/std for DINOv2 and ConvNeXt
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(model_name: str, mode: str = "train"):
    """
    Generates the transformation pipeline for a specific model and mode.

    Args:
        model_name (str): Key from Config.MODELS (e.g., 'clip', 'dinov2', 'convnext').
        mode (str): 'train' for augmentation, 'val'/'test' for deterministic processing.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    if model_name not in Config.MODELS:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available: {list(Config.MODELS.keys())}"
        )

    cfg = Config.MODELS[model_name]
    target_size = cfg["target_size"]

    # Select normalization stats
    if "clip" in model_name.lower():
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    t_list = []

    if mode == "train":
        # Training: Augmentation to improve generalization
        # Resize slightly larger then random crop to introduce scale/position invariance
        resize_dim = int(target_size * 1.14)  # Approx 256 for 224 input
        t_list.append(transforms.Resize((resize_dim, resize_dim)))
        t_list.append(transforms.RandomCrop((target_size, target_size)))
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        # Validation/Test: Deterministic preprocessing
        # Resize shortest edge to target_size (or slightly larger) then CenterCrop
        # This preserves aspect ratio better than simple resizing
        resize_dim = int(target_size * 1.14)
        t_list.append(transforms.Resize((resize_dim, resize_dim)))
        t_list.append(transforms.CenterCrop((target_size, target_size)))

    # Common steps
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(t_list)


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pawpularity Contest.
    Loads images, binary metadata features, and targets.
    """

    def __init__(
        self, metadata_path: str, model_name: str, mode: str = "train", transform=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train/val/test).
            model_name (str): Name of the model to determine default transforms.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Custom transform pipeline. If None, generated based on model_name.
        """
        self.metadata_path = metadata_path
        self.mode = mode
        self.root_dir = Config.INPUT_DIR

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.df = pd.read_csv(metadata_path)

        # Debugging: Subset data if configured
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Setup Transform
        if transform is None:
            self.transform = get_transforms(model_name, mode)
        else:
            self.transform = transform

        # Column definitions
        self.path_col = Config.PATH_COL
        self.target_col = Config.TARGET_COL
        self.id_col = Config.ID_COL
        self.meta_cols = Config.META_FEATURES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in CSV is relative (e.g., "train/id.jpg")
        img_rel_path = row[self.path_col]
        img_full_path = os.path.join(self.root_dir, img_rel_path)

        try:
            image = Image.open(img_full_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for missing images (should be caught by verification script, but for safety)
            # Create a black image
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 2. Apply Transforms
        if self.transform:
            image = self.transform(image)

        # 3. Extract Metadata Features
        # Convert binary flags to float tensor
        meta_features = torch.tensor(
            [row[col] for col in self.meta_cols], dtype=torch.float32
        )

        # 4. Extract Target
        # If target column exists (train/val), return it. Else (test), return 0.0.
        if self.target_col in row:
            target = torch.tensor(row[self.target_col], dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)

        # 5. Get ID
        sample_id = str(row[self.id_col])

        return image, meta_features, target, sample_id
