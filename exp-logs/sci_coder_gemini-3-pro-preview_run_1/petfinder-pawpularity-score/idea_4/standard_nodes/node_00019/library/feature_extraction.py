import os
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union

from library.config import Config
from library.dataset import PawpularityDataset, get_transforms
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class FeatureExtractor:
    """
    Wrapper class to load and handle the backbone models for feature extraction.
    Uses 'timm' for model creation as per the provided configuration identifiers.
    """

    def __init__(self, model_key: str, device: torch.device):
        """
        Args:
            model_key (str): Key identifying the model in Config.MODEL_CONFIGS.
            device (torch.device): Device to load the model onto.
        """
        if model_key not in Config.MODEL_CONFIGS:
            raise ValueError(f"Model key '{model_key}' not found in Config.")

        self.model_key = model_key
        self.device = device
        self.model_config = Config.MODEL_CONFIGS[model_key]

        print(
            f"Initializing model: {self.model_key} using {self.model_config['model_name']}"
        )

        # Initialize model using timm
        # num_classes=0 removes the head and returns the pooled features
        self.model = timm.create_model(
            self.model_config["model_name"], pretrained=True, num_classes=0
        )

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            torch.Tensor: Feature embeddings (B, D).
        """
        with torch.no_grad():
            features = self.model(x)
        return features


def extract_and_cache_features(
    model_key: str,
    split: str,
    load_cached_data: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    subset_size: Optional[int] = None,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
) -> Dict[str, Union[np.ndarray, None]]:
    """
    Extracts features using the specified model and split.
    Applies Test Time Augmentation (Horizontal Flip) and averages the embeddings.
    Caches the results to disk in the working directory.

    Args:
        model_key (str): Key from Config.MODEL_CONFIGS (e.g., 'swin_large').
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.
        device (str): 'cuda' or 'cpu'.
        subset_size (int, optional): If provided, limits the dataset size (for debugging).
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for the dataloader.

    Returns:
        Dict: Dictionary containing:
            - 'features': np.ndarray of shape (N, D) - Image embeddings
            - 'meta': np.ndarray of shape (N, M) - Binary metadata features
            - 'targets': np.ndarray of shape (N, 1) or None - Target values
            - 'ids': np.ndarray of shape (N,) - Image IDs
    """

    # Define cache directory and filenames
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Suffix for subset debugging
    suffix = "" if subset_size is None else f"_subset{subset_size}"

    # Paths for cached files
    feat_path = os.path.join(cache_dir, f"{model_key}_{split}_features{suffix}.npy")
    meta_path = os.path.join(cache_dir, f"{split}_meta{suffix}.npy")
    target_path = os.path.join(cache_dir, f"{split}_targets{suffix}.npy")
    ids_path = os.path.join(cache_dir, f"{split}_ids{suffix}.npy")

    # Check if files exist
    # For test split, target file is not expected
    files_exist = (
        os.path.exists(feat_path)
        and os.path.exists(meta_path)
        and os.path.exists(ids_path)
        and (split == "test" or os.path.exists(target_path))
    )

    if load_cached_data and files_exist:
        print(f"Loading cached features for {model_key} ({split}) from {cache_dir}...")
        data = {
            "features": np.load(feat_path),
            "meta": np.load(meta_path),
            "ids": np.load(ids_path),
        }
        if split != "test":
            data["targets"] = np.load(target_path)
        else:
            data["targets"] = None
        return data

    print(f"Starting feature extraction for {model_key} ({split})...")

    # Determine which metadata file to use
    if split == "train":
        csv_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        csv_path = Config.VAL_METADATA_PATH
    elif split == "test":
        csv_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Setup Dataset
    # We use 'valid' transforms (Resize + Normalize) and handle Flip manually for TTA
    transform = get_transforms(model_key, split="valid")

    dataset = PawpularityDataset(
        csv_path=csv_path, transform=transform, return_target=(split != "test")
    )

    # Handle subset
    if subset_size is not None:
        indices = list(range(min(subset_size, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Subset active: Processing {len(dataset)} samples.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Setup Model
    device_obj = torch.device(device)
    extractor = FeatureExtractor(model_key, device_obj)

    # Storage
    all_features = []
    all_meta = []
    all_targets = []
    all_ids = []

    # Inference loop
    for batch in loader:
        images = batch["image"].to(device_obj)
        meta = batch["features"].numpy()
        ids = batch["id"]

        # 1. Original Forward Pass
        emb_orig = extractor(images)  # (B, D)

        # 2. Flipped Forward Pass (TTA)
        # Flip along width (dim 3 for NCHW)
        images_flip = torch.flip(images, dims=[3])
        emb_flip = extractor(images_flip)  # (B, D)

        # 3. Average
        emb_avg = (emb_orig + emb_flip) / 2.0

        # Store
        all_features.append(emb_avg.cpu().numpy())
        all_meta.append(meta)
        all_ids.extend(ids)

        if "target" in batch:
            all_targets.append(batch["target"].numpy())

    # Concatenate results
    features_arr = np.concatenate(all_features, axis=0)
    meta_arr = np.concatenate(all_meta, axis=0)
    ids_arr = np.array(all_ids)

    if all_targets:
        targets_arr = np.concatenate(all_targets, axis=0)
    else:
        targets_arr = None

    # Save to cache
    print(f"Saving extracted features to {cache_dir}...")
    np.save(feat_path, features_arr)
    # Meta/IDs are saved per extraction to ensure alignment, overwrites are safe
    np.save(meta_path, meta_arr)
    np.save(ids_path, ids_arr)

    if targets_arr is not None:
        np.save(target_path, targets_arr)

    data = {
        "features": features_arr,
        "meta": meta_arr,
        "ids": ids_arr,
        "targets": targets_arr,
    }

    return data
