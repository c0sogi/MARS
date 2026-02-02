import os
import numpy as np
import pandas as pd
import torch
import timm
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import load_image, rotate_image


class LeafDataset(Dataset):
    """
    Dataset class that loads leaf images, applies 12 equidistant rotations,
    and prepares them for the dual-stream models.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'file_path' and optionally 'id'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform
        self.rotation_angles = Config.ROTATION_ANGLES
        self.input_dir = Config.INPUT_DIR
        self.image_size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image (grayscale) and resize to target size
        # We resize first to ensure consistent dimensions for rotation center calculation
        try:
            img_gray = load_image(full_path, target_size=self.image_size)
        except Exception as e:
            # Fallback for missing/corrupt images (should not happen based on metadata check)
            # Create a black image
            img_gray = np.zeros(self.image_size, dtype=np.uint8)

        # Generate 12 rotated views
        views = []
        for angle in self.rotation_angles:
            # Rotate
            rotated = rotate_image(img_gray, angle, border_value=255)

            # Convert to RGB (Duplicate channels) as models expect 3 channels
            img_rgb = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)

            # Normalize to [0, 1] float32
            img_tensor = transforms.functional.to_tensor(img_rgb)

            # Apply ImageNet normalization
            # Mean and Std for ImageNet
            img_tensor = transforms.functional.normalize(
                img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            views.append(img_tensor)

        # Stack views: Shape (12, 3, H, W)
        stack = torch.stack(views)

        return stack


class FeatureExtractor:
    """
    Manages the deep learning models and executes the feature extraction pipeline.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = Config.BATCH_SIZE
        self.num_workers = Config.NUM_WORKERS

        print(f"Initializing FeatureExtractor on {self.device}...")

        # Initialize Models
        # DINOv2 (Global Geometry)
        print(f"Loading DINOv2: {Config.MODEL_DINOV2}")
        self.model_dino = timm.create_model(
            Config.MODEL_DINOV2, pretrained=True, num_classes=0
        )
        self.model_dino.to(self.device)
        self.model_dino.eval()

        # ConvNeXt (Local Texture)
        print(f"Loading ConvNeXt: {Config.MODEL_CONVNEXT}")
        self.model_conv = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        )
        self.model_conv.to(self.device)
        self.model_conv.eval()

    def extract_dataset_features(
        self,
        csv_path,
        cache_img_path,
        cache_tab_path,
        cache_ids_path,
        cache_labels_path=None,
        load_cached_data=True,
    ):
        """
        Main pipeline method to extract features for a dataset defined by a CSV file.
        Handles caching of results to disk.

        Args:
            csv_path (str): Path to the metadata CSV (train/val/test).
            cache_img_path (str): Path to save/load image features .npy.
            cache_tab_path (str): Path to save/load tabular features .npy.
            cache_ids_path (str): Path to save/load IDs .npy.
            cache_labels_path (str, optional): Path to save/load labels .npy.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (img_features, tab_features, ids, labels)
                   labels will be None if cache_labels_path is None.
        """
        # 1. Check Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(cache_img_path)
                and os.path.exists(cache_tab_path)
                and os.path.exists(cache_ids_path)
            )
            if cache_labels_path:
                files_exist = files_exist and os.path.exists(cache_labels_path)

            if files_exist:
                print(
                    f"Loading cached features from {os.path.dirname(cache_img_path)}..."
                )
                img_features = np.load(cache_img_path)
                tab_features = np.load(cache_tab_path)
                ids = np.load(cache_ids_path)
                labels = (
                    np.load(cache_labels_path, allow_pickle=True)
                    if cache_labels_path
                    else None
                )
                return img_features, tab_features, ids, labels

        # 2. Process from Scratch
        print(f"Processing dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        # Debug Mode
        if Config.DEBUG:
            print(
                f"DEBUG mode: Limiting dataset to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        # 2a. Extract Tabular Features, IDs, Labels
        print("Extracting tabular features and metadata...")
        margin_cols = [c for c in df.columns if c.startswith("margin")]
        shape_cols = [c for c in df.columns if c.startswith("shape")]
        texture_cols = [c for c in df.columns if c.startswith("texture")]
        feature_cols = margin_cols + shape_cols + texture_cols

        tab_features = df[feature_cols].values.astype(np.float32)
        ids = df["id"].values.astype(np.int32)

        labels = None
        if "species" in df.columns and cache_labels_path:
            labels = df["species"].values  # Keep as strings/objects

        # 2b. Extract Image Features
        print("Starting deep learning inference...")
        dataset = LeafDataset(df)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        all_img_features = []

        with torch.inference_mode():
            for batch_idx, batch_imgs in enumerate(dataloader):
                # batch_imgs shape: (B, 12, 3, H, W)
                B, V, C, H, W = batch_imgs.shape

                # Flatten views into batch dimension: (B*12, 3, H, W)
                flat_imgs = batch_imgs.view(B * V, C, H, W).to(self.device)

                # Forward Pass - DINOv2
                dino_feats = self.model_dino(flat_imgs)  # (B*12, D1)

                # Forward Pass - ConvNeXt
                conv_feats = self.model_conv(flat_imgs)  # (B*12, D2)

                # Concatenate features
                combined_feats = torch.cat(
                    [dino_feats, conv_feats], dim=1
                )  # (B*12, D1+D2)

                # Reshape back to (B, 12, D_total)
                combined_feats = combined_feats.view(B, V, -1)

                # Move to CPU and store
                all_img_features.append(combined_feats.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        img_features = np.concatenate(all_img_features, axis=0)  # (N, 12, D_total)

        # 3. Save to Cache
        print("Saving features to cache...")
        os.makedirs(os.path.dirname(cache_img_path), exist_ok=True)

        np.save(cache_img_path, img_features)
        np.save(cache_tab_path, tab_features)
        np.save(cache_ids_path, ids)
        if labels is not None and cache_labels_path:
            np.save(cache_labels_path, labels)

        print("Feature extraction complete.")
        return img_features, tab_features, ids, labels
