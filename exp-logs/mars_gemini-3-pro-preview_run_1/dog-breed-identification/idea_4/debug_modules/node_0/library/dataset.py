import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from library.config import Config
from library.utils import log_message

# ImageNet normalization statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(phase: str):
    """
    Returns the data transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    transform_list = []

    # 1. Resize
    # Resize to a slightly larger size before cropping
    transform_list.append(transforms.Resize((Config.RESIZE_SIZE, Config.RESIZE_SIZE)))

    # 2. Phase-specific Augmentation/Cropping
    if phase == "train":
        # Random Horizontal Flip for data augmentation
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # Center Crop to the target model input size
        transform_list.append(transforms.CenterCrop(Config.IMAGE_SIZE))
    else:
        # Deterministic Center Crop for Validation and Test
        transform_list.append(transforms.CenterCrop(Config.IMAGE_SIZE))

    # 3. Convert to Tensor and Normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=MEAN, std=STD))

    return transforms.Compose(transform_list)


def get_class_mapping(
    metadata_path=Config.TRAIN_METADATA_PATH,
    cache_dir=Config.WORKING_DIR,
    load_cached_data=True,
):
    """
    Generates or loads the class-to-index mapping.
    Implements caching to ensure deterministic mapping across runs.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        cache_dir (str): Directory to save/load the mapping.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {breed_name: index}
        list: [breed_name] (where index corresponds to the list index)
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "label_map.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            classes = np.load(cache_path, allow_pickle=True).tolist()
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            # log_message(f"Loaded class mapping from {cache_path}")
            return class_to_idx, classes
        except Exception as e:
            log_message(f"Failed to load cached class mapping: {e}. Recomputing.")

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    # Get unique breeds and sort them alphabetically for consistency
    classes = sorted(df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # 3. Save to cache
    try:
        np.save(cache_path, np.array(classes))
        log_message(f"Saved class mapping to {cache_path}")
    except Exception as e:
        log_message(f"Warning: Failed to save class mapping cache: {e}")

    return class_to_idx, classes


class DogDataset(Dataset):
    """
    Custom Dataset for Dog Breed Classification.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        class_to_idx=None,
        input_dir=Config.INPUT_DIR,
        return_ids=False,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, breed).
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required if return_ids is False.
            input_dir (str): Root directory for images.
            return_ids (bool): If True, returns (image, id) for inference. If False, returns (image, label).
        """
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.input_dir = input_dir
        self.return_ids = return_ids

        # Validation
        if not self.return_ids and self.class_to_idx is None:
            raise ValueError(
                "class_to_idx must be provided when return_ids is False (Training/Validation mode)."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative, e.g., "train/xxx.jpg"
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        # Use PIL for compatibility with torchvision transforms
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # In a real scenario, we might handle this gracefully, but for this task, we fail hard to detect issues.
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")

        # Apply Transforms
        if self.transform:
            image = self.transform(image)

        # Return Data
        if self.return_ids:
            # Inference Mode: Return Image and ID
            return image, row["id"]
        else:
            # Training Mode: Return Image and Label Index
            breed = row["breed"]
            label = self.class_to_idx[breed]
            return image, torch.tensor(label, dtype=torch.long)


def load_datasets(load_cached_data=True):
    """
    Loads metadata, generates class mappings, and prepares DataFrames for training and testing.
    Handles K-Fold preparation by combining train and val metadata.
    Handles DEBUG mode by subsetting data.

    Args:
        load_cached_data (bool): Whether to use cached label mappings.

    Returns:
        tuple: (full_train_df, test_df, class_to_idx, classes)
    """
    # 1. Get Class Mapping
    # We use the training metadata to establish the ground truth classes
    class_to_idx, classes = get_class_mapping(
        metadata_path=Config.TRAIN_METADATA_PATH, load_cached_data=load_cached_data
    )

    # 2. Load Metadata DataFrames
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError("Training or Validation metadata not found.")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Combine Train and Val for K-Fold Cross Validation
    # We will split this combined dataset later using StratifiedKFold
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # 4. Handle Debug Mode
    if Config.DEBUG:
        log_message(
            f"DEBUG mode enabled. Subsetting to {Config.DEBUG_SAMPLE_SIZE} samples."
        )

        # Subset training data
        full_train_df = full_train_df.sample(
            n=min(len(full_train_df), Config.DEBUG_SAMPLE_SIZE),
            random_state=Config.SEED,
        ).reset_index(drop=True)

        # Subset test data
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    return full_train_df, test_df, class_to_idx, classes
