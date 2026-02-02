import os
import gc
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, save_numpy, load_numpy
from library.data_loader import LeafDataset


class DualStreamExtractor:
    """
    Handles feature extraction using two distinct backbones:
    1. Shape Stream: DINOv2 (ViT-Large) - Self-supervised, geometric features.
    2. Texture Stream: ConvNeXt Large - Supervised, texture/margin features.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device

    def _get_dataloader(self, metadata_path, img_size, debug):
        """
        Creates a DataLoader for the LeafDataset.
        Shuffle is strictly False to ensure alignment between the two streams.
        """
        dataset = LeafDataset(metadata_path, img_size, debug=debug)
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    def extract_stream(self, model_name, img_size, metadata_path, debug=False):
        """
        Runs inference for a specific model stream.
        Performs Multi-View Averaging (4 rotations) for rotation invariance.
        """
        seed_everything(Config.SEED)

        print(f"Initializing model: {model_name}...")
        # num_classes=0 returns the pooled feature vector or CLS token
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.to(self.device)
        model.eval()

        dataloader = self._get_dataloader(metadata_path, img_size, debug)

        all_features = []
        all_labels = []
        all_ids = []

        print(f"Starting inference on {metadata_path}...")
        with torch.no_grad():
            for images, labels, ids in dataloader:
                # images shape: (Batch, 4_Views, 3_Channels, Height, Width)
                B, V, C, H, W = images.shape

                # Flatten batch and views to treat them as independent samples
                # Shape: (Batch * 4, 3, H, W)
                images_flat = images.view(-1, C, H, W).to(self.device)

                # Forward pass
                # Shape: (Batch * 4, Embedding_Dim)
                features_flat = model(images_flat)

                # Reshape back to group views by image
                # Shape: (Batch, 4, Embedding_Dim)
                features_grouped = features_flat.view(B, V, -1)

                # Compute Element-wise Average across the 4 views
                # Shape: (Batch, Embedding_Dim)
                features_avg = features_grouped.mean(dim=1)

                # Collect results
                all_features.append(features_avg.cpu().numpy())
                all_labels.extend(labels)
                all_ids.extend(ids.numpy())

        # Clean up to free GPU memory for the next stream
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return (
            np.concatenate(all_features, axis=0),
            np.array(all_labels),
            np.array(all_ids),
        )


def extract_features(split_name, load_cached_data=True, debug=False):
    """
    Main function to extract or load features for a specific data split.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.
        debug (bool): If True, runs on a small subset of data.

    Returns:
        tuple: (features_shape, features_texture, aux_data)
               aux_data is labels (for train/val) or ids (for test).
    """
    # 1. Determine paths and settings based on split
    if split_name == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        path_shape = Config.CACHE_TRAIN_FEATS_SHAPE
        path_texture = Config.CACHE_TRAIN_FEATS_TEXTURE
        path_labels = Config.CACHE_TRAIN_LABELS
        path_ids = None
    elif split_name == "val":
        metadata_path = Config.VAL_METADATA_PATH
        path_shape = Config.CACHE_VAL_FEATS_SHAPE
        path_texture = Config.CACHE_VAL_FEATS_TEXTURE
        path_labels = Config.CACHE_VAL_LABELS
        path_ids = None
    elif split_name == "test":
        metadata_path = Config.TEST_METADATA_PATH
        path_shape = Config.CACHE_TEST_FEATS_SHAPE
        path_texture = Config.CACHE_TEST_FEATS_TEXTURE
        path_labels = None
        path_ids = Config.CACHE_TEST_IDS
    else:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'."
        )

    # 2. Try loading from cache
    if load_cached_data and not debug:
        # We assume if one exists, all exist for that split to ensure consistency
        feats_shape = load_numpy(path_shape)
        feats_texture = load_numpy(path_texture)

        aux_data = None
        if path_labels:
            aux_data = load_numpy(path_labels)
        elif path_ids:
            aux_data = load_numpy(path_ids)

        if (
            feats_shape is not None
            and feats_texture is not None
            and aux_data is not None
        ):
            print(f"Successfully loaded cached features for '{split_name}'.")
            return feats_shape, feats_texture, aux_data

    # 3. Compute from scratch
    print(
        f"Cache miss or forced re-computation. Extracting features for '{split_name}'..."
    )

    extractor = DualStreamExtractor()

    # Stream 1: Shape (DINOv2)
    print("--- Stream 1: Shape (DINOv2) ---")
    feats_shape, labels_shape, ids_shape = extractor.extract_stream(
        Config.MODEL_SHAPE_NAME, Config.IMG_SIZE_SHAPE, metadata_path, debug
    )

    # Stream 2: Texture (ConvNeXt)
    print("--- Stream 2: Texture (ConvNeXt) ---")
    feats_texture, labels_texture, ids_texture = extractor.extract_stream(
        Config.MODEL_TEXTURE_NAME, Config.IMG_SIZE_TEXTURE, metadata_path, debug
    )

    # Verification: Ensure alignment
    if not np.array_equal(ids_shape, ids_texture):
        raise RuntimeError(
            "Critical Error: Sample ID mismatch between Shape and Texture streams."
        )

    # 4. Save to cache (only if not debugging)
    if not debug:
        print(f"Saving features to cache in {Config.WORKING_DIR}...")
        save_numpy(feats_shape, path_shape)
        save_numpy(feats_texture, path_texture)

        aux_data = None
        if split_name in ["train", "val"]:
            save_numpy(labels_shape, path_labels)
            aux_data = labels_shape
        elif split_name == "test":
            save_numpy(ids_shape, path_ids)
            aux_data = ids_shape

        return feats_shape, feats_texture, aux_data
    else:
        # In debug mode, return the computed data but do not overwrite cache
        if split_name in ["train", "val"]:
            return feats_shape, feats_texture, labels_shape
        else:
            return feats_shape, feats_texture, ids_shape
