import os
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from library.config import Config
from library.image_loader import LeafDataset
from library.utils import seed_everything


class DualStreamExtractor:
    """
    A dual-stream feature extractor that utilizes DINOv2 (Global Geometry) and
    ConvNeXt (Local Texture) to generate dense visual embeddings for leaf images.

    It supports multi-view extraction by iterating through a defined set of
    rotation angles.
    """

    def __init__(self, device=None):
        """
        Initialize the extractor by loading pre-trained models.

        Args:
            device (str, optional): Computation device ('cuda' or 'cpu').
                                    Defaults to auto-detection.
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Initializing DualStreamExtractor on {self.device}...")

        # Load DINOv2 (ViT) - Global Geometry Stream
        # num_classes=0 returns the pooled feature vector (CLS token or Avg Pool)
        self.dino_model = timm.create_model(
            Config.MODEL_DINO, pretrained=True, num_classes=0
        )
        self.dino_model.to(self.device).eval()

        # Load ConvNeXt - Local Texture Stream
        self.convnext_model = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        )
        self.convnext_model.to(self.device).eval()

    def extract_features(self, file_paths):
        """
        Extracts visual features for the provided images across all configured rotation angles.

        Args:
            file_paths (list): List of absolute file paths to the images.

        Returns:
            tuple: (dino_features, convnext_features)
                - dino_features: Numpy array of shape (N_samples, N_views, D_dino)
                - convnext_features: Numpy array of shape (N_samples, N_views, D_convnext)
        """
        n_samples = len(file_paths)
        n_views = Config.NUM_VIEWS

        # Determine embedding dimensions dynamically using a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(self.device)
            dino_dim = self.dino_model(dummy).shape[1]
            conv_dim = self.convnext_model(dummy).shape[1]

        # Pre-allocate memory for the dense feature tensors
        # Shape: (Samples, Views, Embedding_Dim)
        dino_all = np.zeros((n_samples, n_views, dino_dim), dtype=np.float32)
        conv_all = np.zeros((n_samples, n_views, conv_dim), dtype=np.float32)

        print(f"Starting extraction: {n_samples} images x {n_views} views")

        # Iterate through each rotation angle defined in the topology
        for view_idx, angle in enumerate(Config.ROTATION_ANGLES):
            # Create dataset for the specific view
            dataset = LeafDataset(file_paths, rotation_angle=angle)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            view_dino_list = []
            view_conv_list = []

            # Inference loop
            with torch.no_grad():
                for imgs in loader:
                    imgs = imgs.to(self.device)

                    # Extract Global Geometry (DINOv2)
                    d_feats = self.dino_model(imgs)
                    view_dino_list.append(d_feats.cpu().numpy())

                    # Extract Local Texture (ConvNeXt)
                    c_feats = self.convnext_model(imgs)
                    view_conv_list.append(c_feats.cpu().numpy())

            # Aggregate batches for this view
            dino_all[:, view_idx, :] = np.concatenate(view_dino_list, axis=0)
            conv_all[:, view_idx, :] = np.concatenate(view_conv_list, axis=0)

        return dino_all, conv_all


def process_split(split_name, load_cached_data=True, limit=None):
    """
    Orchestrates the feature extraction pipeline for a specific dataset split.
    Handles metadata loading, caching logic, and calls the DualStreamExtractor.

    Args:
        split_name (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk before computing.
        limit (int, optional): Limit the number of samples for debugging purposes.

    Returns:
        dict: A dictionary containing:
            - 'dino': Visual embeddings from DINOv2
            - 'convnext': Visual embeddings from ConvNeXt
            - 'tabular': Extracted tabular features
            - 'ids': Image IDs
            - 'labels': Target labels (if available in the split)
    """
    seed_everything()

    # 1. Configuration Mapping
    if split_name == "train":
        meta_path = Config.TRAIN_METADATA
        cache_map = {
            "dino": Config.CACHE_TRAIN_DINO,
            "convnext": Config.CACHE_TRAIN_CONVNEXT,
            "tabular": Config.CACHE_TRAIN_TABULAR,
            "ids": Config.CACHE_TRAIN_IDS,
            "labels": Config.CACHE_TRAIN_LABELS,
        }
    elif split_name == "val":
        meta_path = Config.VAL_METADATA
        cache_map = {
            "dino": Config.CACHE_VAL_DINO,
            "convnext": Config.CACHE_VAL_CONVNEXT,
            "tabular": Config.CACHE_VAL_TABULAR,
            "ids": Config.CACHE_VAL_IDS,
            "labels": Config.CACHE_VAL_LABELS,
        }
    elif split_name == "test":
        meta_path = Config.TEST_METADATA
        cache_map = {
            "dino": Config.CACHE_TEST_DINO,
            "convnext": Config.CACHE_TEST_CONVNEXT,
            "tabular": Config.CACHE_TEST_TABULAR,
            "ids": Config.CACHE_TEST_IDS,
        }
    else:
        raise ValueError(f"Invalid split name: {split_name}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Caching Logic
    # We only load from cache if ALL required files exist and limit is None (full dataset)
    cache_files_exist = all(os.path.exists(p) for p in cache_map.values())

    if load_cached_data and cache_files_exist and limit is None:
        print(f"[{split_name}] Loading features from cache...")
        data = {}
        for key, path in cache_map.items():
            data[key] = np.load(path)
        return data

    # 3. Processing (Cache Miss or Forced Reload)
    print(f"[{split_name}] Processing from scratch (Cache miss or limit set)...")

    # Load Metadata
    df = pd.read_csv(meta_path)
    if limit:
        print(f"[{split_name}] Limiting to first {limit} samples.")
        df = df.head(limit)

    # Prepare File Paths
    # Metadata contains relative paths (e.g., "images/1.jpg").
    # Config.INPUT_DIR is "./input". Result: "./input/images/1.jpg"
    file_paths = [
        os.path.join(Config.INPUT_DIR, str(row["file_path"]))
        for _, row in df.iterrows()
    ]

    # Extract Visual Features
    extractor = DualStreamExtractor()
    dino_feats, conv_feats = extractor.extract_features(file_paths)

    # Extract Tabular Features
    # Identify columns: margin_1..64, shape_1..64, texture_1..64
    tab_cols = [
        c
        for c in df.columns
        if c.startswith("margin") or c.startswith("shape") or c.startswith("texture")
    ]
    tabular_feats = df[tab_cols].values.astype(np.float32)

    # Extract IDs
    ids = df["id"].values

    # Prepare Result Dictionary
    result = {
        "dino": dino_feats,
        "convnext": conv_feats,
        "tabular": tabular_feats,
        "ids": ids,
    }

    # Handle Labels (if present)
    if "species" in df.columns:
        labels = df["species"].values
        result["labels"] = labels
    elif "labels" in cache_map:
        # If we expected labels but they aren't in DF (shouldn't happen for train/val)
        print(
            f"Warning: Labels requested for {split_name} but 'species' column not found."
        )

    # 4. Save to Cache
    # Only save if we processed the full dataset (limit is None)
    if limit is None:
        print(f"[{split_name}] Saving features to cache...")
        np.save(cache_map["dino"], dino_feats)
        np.save(cache_map["convnext"], conv_feats)
        np.save(cache_map["tabular"], tabular_feats)
        np.save(cache_map["ids"], ids)

        if "labels" in result and "labels" in cache_map:
            np.save(cache_map["labels"], result["labels"])

    return result
