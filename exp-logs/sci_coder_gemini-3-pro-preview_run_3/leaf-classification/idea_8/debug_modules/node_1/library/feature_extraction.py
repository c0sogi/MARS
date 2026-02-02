import os
import torch
import numpy as np
import timm
from library.config import Config
from library.data_loader import get_data_loaders
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles feature extraction using pre-trained DINOv2 and ConvNeXt models.
    Implements multi-view averaging and caching.
    """

    def __init__(self):
        """
        Initializes the models and sets the device.
        """
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        # Model names
        # Using specific timm name for DINOv2 as per task description
        self.dino_model_name = "vit_large_patch14_dinov2"
        self.convnext_model_name = Config.CONVNEXT_MODEL_NAME

        print(f"Initializing Feature Extractor on {self.device}...")

        # Initialize DINOv2 (Global Geometry Stream)
        print(f"  Loading DINOv2: {self.dino_model_name}")
        self.dino = timm.create_model(
            self.dino_model_name,
            pretrained=True,
            num_classes=0,  # Features only, no classifier head
            img_size=Config.IMAGE_SIZE,
        ).to(self.device)
        self.dino.eval()

        # Initialize ConvNeXt (Local Texture Stream)
        print(f"  Loading ConvNeXt: {self.convnext_model_name}")
        self.convnext = timm.create_model(
            self.convnext_model_name,
            pretrained=True,
            num_classes=0,  # Features only
        ).to(self.device)
        self.convnext.eval()

    def _get_loader(self, split):
        """
        Helper to get the correct loader for the split.
        """
        train_loader, val_loader, test_loader = get_data_loaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            load_cached_data=True,
        )

        if split == "train":
            return train_loader
        elif split == "val":
            return val_loader
        elif split == "test":
            return test_loader
        else:
            raise ValueError(f"Unknown split: {split}")

    def extract_features(self, split, load_cached_data=True):
        """
        Extracts features for a specific data split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            tuple: (dino_features, conv_features, ids, labels)
                   labels will be None for 'test' split.
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Define cache paths
        path_dino = os.path.join(Config.CACHE_DIR, f"{split}_dino_features.npy")
        path_conv = os.path.join(Config.CACHE_DIR, f"{split}_conv_features.npy")
        path_ids = os.path.join(Config.CACHE_DIR, f"{split}_ids.npy")
        path_labels = os.path.join(Config.CACHE_DIR, f"{split}_labels.npy")

        # 1. Try Loading from Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(path_dino)
                and os.path.exists(path_conv)
                and os.path.exists(path_ids)
            )

            # Check labels file existence only if not test split
            if split != "test":
                files_exist = files_exist and os.path.exists(path_labels)

            if files_exist:
                print(
                    f"Loading cached features for '{split}' from {Config.CACHE_DIR}..."
                )
                dino_feats = np.load(path_dino)
                conv_feats = np.load(path_conv)
                ids = np.load(path_ids)
                labels = np.load(path_labels) if split != "test" else None
                return dino_feats, conv_feats, ids, labels

        # 2. Compute Features
        print(f"Extracting features for '{split}' split...")
        loader = self._get_loader(split)

        dino_feats_list = []
        conv_feats_list = []
        ids_list = []
        labels_list = []

        with torch.no_grad():
            for i, (images, batch_labels, batch_ids) in enumerate(loader):
                # Input images shape: (Batch, 4, Channels, Height, Width)
                B, V, C, H, W = images.shape

                # Flatten views into batch dimension for parallel inference
                # Shape: (Batch * 4, C, H, W)
                flat_images = images.view(B * V, C, H, W).to(self.device)

                # --- DINOv2 Inference ---
                # Output: (Batch * 4, Embed_Dim_DINO)
                dino_out = self.dino(flat_images)

                # --- ConvNeXt Inference ---
                # Output: (Batch * 4, Embed_Dim_Conv)
                conv_out = self.convnext(flat_images)

                # --- Multi-View Averaging ---
                # Reshape back to (Batch, 4, Embed_Dim) and mean over dim 1
                dino_out = dino_out.view(B, V, -1).mean(dim=1)
                conv_out = conv_out.view(B, V, -1).mean(dim=1)

                # Store results
                dino_feats_list.append(dino_out.cpu().numpy())
                conv_feats_list.append(conv_out.cpu().numpy())
                ids_list.append(batch_ids.numpy())

                if split != "test":
                    labels_list.append(batch_labels.numpy())

                if (i + 1) % 10 == 0:
                    print(f"  Processed batch {i + 1}/{len(loader)}")

        # Concatenate all batches
        dino_feats = np.concatenate(dino_feats_list, axis=0)
        conv_feats = np.concatenate(conv_feats_list, axis=0)
        ids = np.concatenate(ids_list, axis=0)
        labels = np.concatenate(labels_list, axis=0) if split != "test" else None

        # 3. Save to Cache
        print(f"Saving features to {Config.CACHE_DIR}...")
        np.save(path_dino, dino_feats)
        np.save(path_conv, conv_feats)
        np.save(path_ids, ids)
        if labels is not None:
            np.save(path_labels, labels)

        return dino_feats, conv_feats, ids, labels
