import os
import numpy as np
import torch
import timm
from library.utils import get_device, IDEA_DIR
from library.data import create_dataloaders


class DualBackboneExtractor:
    """
    Extracts features using an ensemble of Swin Transformer Large and ConvNeXt Large.
    Implements Feature-Space Augmentation (TTA) and caching.
    """

    def __init__(self):
        self.device = get_device()
        print(f"Initializing DualBackboneExtractor on {self.device}...")

        # Initialize Swin Transformer Large (ImageNet-22k pretrained)
        # Cite Lesson 14: Backbone Capacity Dominates
        # num_classes=0 returns the pooled feature vector
        self.swin = timm.create_model(
            "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
        ).to(self.device)
        self.swin.eval()

        # Initialize ConvNeXt Large (ImageNet-22k pretrained)
        # Cite Lesson 15: Cross-Architecture Feature Fusion
        self.convnext = timm.create_model(
            "convnext_large.fb_in22k_ft_in1k", pretrained=True, num_classes=0
        ).to(self.device)
        self.convnext.eval()

    def _extract_from_loader(self, loader, is_test=False):
        """
        Internal method to iterate over a dataloader and extract features.

        Args:
            loader (DataLoader): The dataloader to iterate over.
            is_test (bool): Whether the loader is for the test set (returns IDs instead of targets).

        Returns:
            tuple: (features_arr, meta_arr, targets_arr)
        """
        all_features = []
        all_meta = []
        all_targets = []  # Holds float targets for train/val, string IDs for test

        with torch.no_grad():
            for batch in loader:
                if is_test:
                    images, meta, ids = batch
                    # For test set, we store IDs to align predictions later
                    targets_batch = np.array(ids)
                else:
                    images, meta, targets = batch
                    targets_batch = targets.numpy()

                images = images.to(self.device)

                # Feature-Space Augmentation: Horizontal Flip
                # Images are (N, C, H, W). Flip on Width (dim 3).
                images_flipped = torch.flip(images, dims=[3])

                # --- Swin Transformer Extraction ---
                swin_orig = self.swin(images)
                swin_flip = self.swin(images_flipped)
                # Average embeddings
                swin_emb = (swin_orig + swin_flip) / 2.0

                # --- ConvNeXt Extraction ---
                conv_orig = self.convnext(images)
                conv_flip = self.convnext(images_flipped)
                # Average embeddings
                conv_emb = (conv_orig + conv_flip) / 2.0

                # --- Fusion ---
                # Concatenate the two backbone embeddings
                combined_emb = torch.cat([swin_emb, conv_emb], dim=1)

                # Store results
                all_features.append(combined_emb.cpu().numpy())
                all_meta.append(meta.numpy())
                all_targets.append(targets_batch)

        # Concatenate all batches into single arrays
        features_arr = np.concatenate(all_features, axis=0)
        meta_arr = np.concatenate(all_meta, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)

        return features_arr, meta_arr, targets_arr

    def get_features(self, mode, load_cached_data=True):
        """
        Retrieves features for a specific mode (train/val/test).
        Handles caching logic: loads from disk if available and requested,
        otherwise computes and saves.

        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            tuple: (features, meta, targets)
        """
        # Ensure cache directory exists
        os.makedirs(IDEA_DIR, exist_ok=True)

        # Define cache file paths
        feat_path = os.path.join(IDEA_DIR, f"{mode}_features.npy")
        meta_path = os.path.join(IDEA_DIR, f"{mode}_meta.npy")
        target_path = os.path.join(IDEA_DIR, f"{mode}_targets.npy")

        # Check if cache files exist
        cache_exists = (
            os.path.exists(feat_path)
            and os.path.exists(meta_path)
            and os.path.exists(target_path)
        )

        # Load from cache if requested and available
        if load_cached_data and cache_exists:
            print(f"Loading cached features for '{mode}' from {IDEA_DIR}...")
            features = np.load(feat_path)
            meta = np.load(meta_path)
            targets = np.load(target_path)
            return features, meta, targets

        # Otherwise, compute features
        print(f"Computing features for '{mode}'...")

        # Initialize DataLoaders
        train_loader, val_loader, test_loader = create_dataloaders()

        # Select appropriate loader
        if mode == "train":
            loader = train_loader
            is_test = False
        elif mode == "val":
            loader = val_loader
            is_test = False
        elif mode == "test":
            loader = test_loader
            is_test = True
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Extract features
        features, meta, targets = self._extract_from_loader(loader, is_test=is_test)

        # Save to cache
        print(f"Saving features for '{mode}' to {IDEA_DIR}...")
        np.save(feat_path, features)
        np.save(meta_path, meta)
        np.save(target_path, targets)

        return features, meta, targets
