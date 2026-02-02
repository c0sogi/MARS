import os
import numpy as np
import torch
import timm
from library.config import Config
from library.utils import setup_logger
from library.data_loader import get_dataloaders


class FeatureExtractor:
    """
    Handles feature extraction using pre-trained models with caching and TTA support.
    """

    def __init__(self):
        self.logger = setup_logger(name="feature_extractor")
        self.device = torch.device(Config.DEVICE)
        self.models = self._load_models()

    def _load_models(self):
        """
        Loads and initializes the backbone models defined in Config.
        """
        models = []
        self.logger.info(f"Loading backbones: {Config.BACKBONES}")

        for model_name in Config.BACKBONES:
            # Create model with num_classes=0 to get feature vector (pooling applied)
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            model.to(self.device)
            model.eval()

            # Freeze parameters
            for param in model.parameters():
                param.requires_grad = False

            models.append(model)

        return models

    def extract_features(self, dataloader, split_name):
        """
        Runs inference on the dataloader to extract features.

        Args:
            dataloader: PyTorch DataLoader.
            split_name: Name of the split (train/val/test) for logging.

        Returns:
            dict: {'features': np.array, 'targets': np.array, 'ids': np.array}
        """
        self.logger.info(f"Extracting features for {split_name} set...")

        all_features = []
        all_targets = []
        all_ids = []

        # Ensure models are in eval mode
        for model in self.models:
            model.eval()

        with torch.no_grad():
            for batch_idx, (images, meta, targets, ids) in enumerate(dataloader):
                images = images.to(self.device)
                meta = meta.to(self.device)

                batch_features = []

                # Process through each backbone
                for model in self.models:
                    # Forward pass original images
                    out = model(images)

                    # Test-Time Augmentation
                    if Config.USE_TTA:
                        # Horizontal flip (N, C, H, W) -> dim 3 is width
                        images_flipped = torch.flip(images, dims=[3])
                        out_flipped = model(images_flipped)
                        # Average predictions
                        out = (out + out_flipped) / 2.0

                    batch_features.append(out)

                # Concatenate backbone features: [N, Dim1] + [N, Dim2] -> [N, Dim1+Dim2]
                img_features = torch.cat(batch_features, dim=1)

                # Concatenate metadata features: [N, Dim_Img] + [N, Dim_Meta] -> [N, Total_Dim]
                final_features = torch.cat([img_features, meta], dim=1)

                all_features.append(final_features.cpu().numpy())
                all_targets.append(targets.numpy())
                all_ids.extend(ids)

                if (batch_idx + 1) % 50 == 0:
                    self.logger.info(
                        f"Processed {batch_idx + 1} batches for {split_name}"
                    )

        # Concatenate all batches
        features_arr = np.concatenate(all_features, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        ids_arr = np.array(all_ids)

        self.logger.info(f"Finished {split_name}. Shape: {features_arr.shape}")

        return {"features": features_arr, "targets": targets_arr, "ids": ids_arr}

    def extract_and_cache(self, load_cached_data=True):
        """
        Main method to get features. Checks cache first, otherwise computes and saves.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            dict: Dictionary with keys 'train', 'val', 'test', each containing
                  {'features', 'targets', 'ids'}.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Generate hash for versioning
        config_hash = Config.get_config_hash()
        self.logger.info(f"Config Hash: {config_hash}")

        splits = ["train", "val", "test"]
        data = {}

        # Check if all cache files exist
        cache_files = {}
        all_cached = True

        for split in splits:
            feat_path = os.path.join(
                Config.WORKING_DIR, f"{split}_features_{config_hash}.npy"
            )
            target_path = os.path.join(
                Config.WORKING_DIR, f"{split}_targets_{config_hash}.npy"
            )
            id_path = os.path.join(Config.WORKING_DIR, f"{split}_ids_{config_hash}.npy")

            cache_files[split] = {
                "features": feat_path,
                "targets": target_path,
                "ids": id_path,
            }

            if not (
                os.path.exists(feat_path)
                and os.path.exists(target_path)
                and os.path.exists(id_path)
            ):
                all_cached = False

        # Load from cache if requested and available
        if load_cached_data and all_cached:
            self.logger.info("Loading features from cache...")
            try:
                for split in splits:
                    data[split] = {
                        "features": np.load(cache_files[split]["features"]),
                        "targets": np.load(cache_files[split]["targets"]),
                        "ids": np.load(cache_files[split]["ids"]),
                    }
                self.logger.info("Successfully loaded cached features.")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        self.logger.info("Computing features from scratch...")
        loaders = get_dataloaders()

        for split in splits:
            if split not in loaders:
                continue

            split_data = self.extract_features(loaders[split], split)
            data[split] = split_data

            # Save to cache
            np.save(cache_files[split]["features"], split_data["features"])
            np.save(cache_files[split]["targets"], split_data["targets"])
            np.save(cache_files[split]["ids"], split_data["ids"])

        self.logger.info("Feature extraction and caching completed.")
        return data
