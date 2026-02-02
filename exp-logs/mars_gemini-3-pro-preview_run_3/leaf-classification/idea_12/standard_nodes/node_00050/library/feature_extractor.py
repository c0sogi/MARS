import os
import torch
import timm
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from library.config import (
    WORKING_DIR,
    IMAGE_SIZE,
    ROTATION_ANGLES,
    BACKBONE_DINO,
    BACKBONE_CONV,
    BATCH_SIZE,
    NUM_WORKERS,
    INPUT_DIR,
    SEED,
)
from library.utils import seed_everything, save_npy, load_npy

# Set seed for reproducibility across operations
seed_everything(SEED)


class LeafDataset(Dataset):
    """
    Dataset class that loads images and generates 4 canonical rotated views.
    """

    def __init__(self, df, input_dir, transform=None):
        self.df = df
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path. Metadata file_path is relative (e.g., "images/1.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Read image (Binary black leaves on white background)
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for safety, though metadata validation ensures files exist
            img = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL for easy rotation and resizing
        img_pil = Image.fromarray(img)

        # Resize to target size
        resize_tf = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        img_resized = resize_tf(img_pil)

        views = []
        for angle in ROTATION_ANGLES:
            # Apply rotation
            if angle == 0:
                img_rot = img_resized
            else:
                img_rot = img_resized.rotate(angle)

            # Apply Normalization and ToTensor
            if self.transform:
                img_tensor = self.transform(img_rot)
            else:
                img_tensor = transforms.ToTensor()(img_rot)

            views.append(img_tensor)

        # Stack views to return tensor of shape (4, C, H, W)
        return torch.stack(views)


class FeatureExtractor:
    """
    Handles loading of models, extraction of image embeddings,
    and processing of tabular data with caching.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_model = None
        self.conv_model = None

    def _init_models(self):
        """
        Initializes the DINOv2 and ConvNeXt backbones.
        """
        if self.dino_model is not None:
            return

        print(f"Initializing backbones on {self.device}...")

        # Initialize DINOv2 (Global Geometry)
        try:
            self.dino_model = timm.create_model(
                BACKBONE_DINO, pretrained=True, num_classes=0, img_size=IMAGE_SIZE
            )
            self.dino_model.to(self.device)
            self.dino_model.eval()
        except Exception as e:
            print(f"Failed to load DINO model {BACKBONE_DINO}: {e}")
            raise e

        # Initialize ConvNeXt (Local Texture)
        try:
            self.conv_model = timm.create_model(
                BACKBONE_CONV, pretrained=True, num_classes=0
            )
        except Exception as e:
            print(f"Warning: Failed to load {BACKBONE_CONV}: {e}")
            # Fallback logic in case 'mlp' suffix is custom/unavailable
            fallback = "convnext_large"
            print(f"Attempting fallback to {fallback}...")
            self.conv_model = timm.create_model(
                fallback, pretrained=True, num_classes=0
            )

        self.conv_model.to(self.device)
        self.conv_model.eval()

    def process_tabular(self, df):
        """
        Extracts and structures the 192 tabular features (margin, shape, texture).

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: Array of shape (N, 192).
        """
        # Identify feature columns
        margin_cols = [c for c in df.columns if c.startswith("margin")]
        shape_cols = [c for c in df.columns if c.startswith("shape")]
        texture_cols = [c for c in df.columns if c.startswith("texture")]

        # Helper to sort columns numerically (e.g., margin_2 vs margin_10)
        def sort_key(x):
            try:
                return int(x.split("_")[-1])
            except:
                return 0

        margin_cols.sort(key=sort_key)
        shape_cols.sort(key=sort_key)
        texture_cols.sort(key=sort_key)

        feature_cols = margin_cols + shape_cols + texture_cols

        # Extract as float32 array
        X_tab = df[feature_cols].values.astype(np.float32)
        return X_tab

    def extract_features(self, df, split_name, load_cached_data=True):
        """
        Main method to extract features for a dataset split.

        Args:
            df (pd.DataFrame): Metadata dataframe containing file paths and tabular data.
            split_name (str): Name of the split ('train', 'val', 'test') for cache naming.
            load_cached_data (bool): If True, attempts to load from disk first.

        Returns:
            tuple: (img_features, tab_features, ids, labels)
                img_features: (N, 4, Embedding_Dim)
                tab_features: (N, 192)
                ids: (N,)
                labels: (N,) or None
        """
        # Construct cache file paths
        cache_dir = WORKING_DIR
        path_img = os.path.join(cache_dir, f"{split_name}_img_features.npy")
        path_tab = os.path.join(cache_dir, f"{split_name}_tab_features.npy")
        path_ids = os.path.join(cache_dir, f"{split_name}_ids.npy")
        path_lbl = os.path.join(cache_dir, f"{split_name}_labels.npy")

        has_labels = "species" in df.columns

        # 1. Try to load from cache
        if load_cached_data:
            files_exist = (
                os.path.exists(path_img)
                and os.path.exists(path_tab)
                and os.path.exists(path_ids)
            )
            labels_ok = (not has_labels) or os.path.exists(path_lbl)

            if files_exist and labels_ok:
                print(f"Loading cached features for '{split_name}' from {cache_dir}...")
                img_features = load_npy(path_img)
                tab_features = load_npy(path_tab)
                ids = load_npy(path_ids)
                labels = load_npy(path_lbl) if has_labels else None
                return img_features, tab_features, ids, labels

        # 2. Compute features if cache miss
        print(f"Computing features for '{split_name}' (Cache miss or force reload)...")

        # Initialize models
        self._init_models()

        # Standard ImageNet normalization
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Setup Dataset and DataLoader
        dataset = LeafDataset(df, INPUT_DIR, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        all_img_features = []

        with torch.no_grad():
            for batch_imgs in loader:
                # batch_imgs shape: (B, 4, 3, H, W)
                B, V, C, H, W = batch_imgs.shape

                # Flatten views into batch dimension for efficient inference
                # (B*4, 3, H, W)
                flat_imgs = batch_imgs.view(B * V, C, H, W).to(self.device)

                # Extract features from both backbones
                feat_dino = self.dino_model(flat_imgs)  # (B*V, D1)
                feat_conv = self.conv_model(flat_imgs)  # (B*V, D2)

                # Concatenate embeddings (Early Fusion)
                feat_concat = torch.cat([feat_dino, feat_conv], dim=1)  # (B*V, D1+D2)

                # Reshape back to (B, V, D) to preserve view structure
                feat_reshaped = feat_concat.view(B, V, -1)

                all_img_features.append(feat_reshaped.cpu().numpy())

        # Concatenate all batches
        img_features = np.concatenate(all_img_features, axis=0)  # (N, 4, D)

        # Process Tabular Data
        tab_features = self.process_tabular(df)

        # Extract IDs and Labels
        ids = df["id"].values
        labels = df["species"].values if has_labels else None

        # 3. Save to cache
        print(f"Saving computed features to {cache_dir}...")
        save_npy(img_features, path_img)
        save_npy(tab_features, path_tab)
        save_npy(ids, path_ids)
        if labels is not None:
            save_npy(labels, path_lbl)

        return img_features, tab_features, ids, labels
