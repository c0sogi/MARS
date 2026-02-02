import os
import torch
import timm
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from library.config import Config
from library.utils import get_config_hash, seed_everything
from library.data_loader import get_dataloaders


class FeatureExtractor:
    """
    Handles loading of pre-trained backbones and extraction of features
    from the dataset with Test-Time Augmentation (TTA) and caching.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.backbone_names = Config.BACKBONES
        self.models = []
        self._load_backbones()

    def _load_backbones(self):
        """
        Loads the models specified in Config.BACKBONES using timm.
        Sets them to evaluation mode and moves them to the appropriate device.
        """
        print(f"Loading backbones on {self.device}...")
        for model_name in self.backbone_names:
            print(f"  - Loading {model_name}...")
            # Create model with num_classes=0 to get feature embeddings (pooling layer output)
            # Cite debug_lesson_7: Conditionally Pass img_size Based on Model Architecture
            kwargs = {
                "pretrained": True,
                "num_classes": 0,
                "in_chans": 3,
            }
            # Only pass img_size to Transformers that need it for positional embedding interpolation
            if "swin" in model_name or "vit" in model_name:
                kwargs["img_size"] = Config.IMAGE_SIZE

            model = timm.create_model(model_name, **kwargs)
            model.to(self.device)
            model.eval()
            self.models.append(model)
        print("All backbones loaded successfully.")

    def _extract_epoch(self, dataloader, desc: str) -> Dict[str, np.ndarray]:
        """
        Extracts features for a single dataloader.
        Applies TTA (Horizontal Flip) and averages the embeddings.

        Args:
            dataloader: PyTorch DataLoader
            desc: Description for logging (e.g., "Extracting Train")

        Returns:
            Dict containing:
                - 'features_{backbone_name}': np.ndarray of shape (N, embed_dim)
                - 'metadata': np.ndarray of shape (N, 12)
                - 'targets': np.ndarray of shape (N,) or None
                - 'ids': np.ndarray of shape (N,)
        """
        # Storage containers
        features_list = {name: [] for name in self.backbone_names}
        metadata_list = []
        targets_list = []
        ids_list = []

        with torch.no_grad():
            # Iterate without progress bar as per requirements, but print status occasionally if needed
            # Using simple enumeration for internal tracking if debug needed
            for batch in dataloader:
                images = batch["image"].to(self.device)
                meta = batch["metadata"].numpy()
                batch_ids = batch["id"]

                # Handle targets if they exist (train/val)
                if "target" in batch:
                    targets = batch["target"].numpy()
                    targets_list.append(targets)

                metadata_list.append(meta)
                ids_list.append(batch_ids)

                # Test-Time Augmentation: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])

                # Extract features from each backbone
                for model, name in zip(self.models, self.backbone_names):
                    # Forward pass original
                    emb_orig = model(images)
                    # Forward pass flipped
                    emb_flip = model(images_flipped)

                    # Average the embeddings
                    emb_avg = (emb_orig + emb_flip) / 2.0

                    # Move to CPU and store
                    features_list[name].append(emb_avg.cpu().numpy())

        # Concatenate all batches
        result = {}
        for name in self.backbone_names:
            result[f"features_{name}"] = np.vstack(features_list[name])

        result["metadata"] = np.vstack(metadata_list)
        result["ids"] = np.concatenate(ids_list)

        if targets_list:
            result["targets"] = np.concatenate(targets_list)
        else:
            result["targets"] = None

        return result

    def extract_and_cache_features(
        self, load_cached_data: bool = True
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Main method to get features for train, val, and test sets.
        Handles caching logic based on configuration hash.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            Tuple of dictionaries (train_data, val_data, test_data).
            Each dictionary contains features per backbone, metadata, targets, and ids.
        """
        # Generate unique hash for current configuration
        config_hash = get_config_hash(Config)

        # Define cache file paths
        cache_files = {
            "train": os.path.join(
                Config.WORKING_DIR, f"train_features_{config_hash}.npz"
            ),
            "val": os.path.join(Config.WORKING_DIR, f"val_features_{config_hash}.npz"),
            "test": os.path.join(
                Config.WORKING_DIR, f"test_features_{config_hash}.npz"
            ),
        }

        # Check if all cache files exist
        all_cached = all(os.path.exists(path) for path in cache_files.values())

        if load_cached_data and all_cached:
            print(
                f"Loading cached features from {Config.WORKING_DIR} (Hash: {config_hash})..."
            )
            data_splits = {}
            for split, path in cache_files.items():
                loaded = np.load(path, allow_pickle=True)
                # Convert NpzFile object to standard dictionary
                data_dict = {key: loaded[key] for key in loaded.files}
                # Handle None for targets in test set (saved as object array or skipped)
                if split == "test" and "targets" not in data_dict:
                    data_dict["targets"] = None
                elif split == "test" and data_dict["targets"].shape == ():
                    data_dict["targets"] = None

                data_splits[split] = data_dict

            return data_splits["train"], data_splits["val"], data_splits["test"]

        # If not cached or reload forced, compute from scratch
        print(f"Extracting features (Hash: {config_hash})...")

        # Get DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders()

        # Extract
        print("Processing Train Set...")
        train_data = self._extract_epoch(train_loader, "Train")

        print("Processing Validation Set...")
        val_data = self._extract_epoch(val_loader, "Val")

        print("Processing Test Set...")
        test_data = self._extract_epoch(test_loader, "Test")

        # Save to cache
        print(f"Saving features to {Config.WORKING_DIR}...")

        # Helper to save dictionary to npz
        def save_npz(path, data_dict):
            # Filter out None values (e.g. test targets) to avoid saving issues
            save_dict = {k: v for k, v in data_dict.items() if v is not None}
            np.savez_compressed(path, **save_dict)

        save_npz(cache_files["train"], train_data)
        save_npz(cache_files["val"], val_data)
        save_npz(cache_files["test"], test_data)

        return train_data, val_data, test_data
