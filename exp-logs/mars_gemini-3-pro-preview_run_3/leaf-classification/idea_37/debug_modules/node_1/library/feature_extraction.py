import os
import numpy as np
import pandas as pd
import cv2
import torch
import timm
from PIL import Image
from library.config import Config
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles the extraction of deep learning features using DINOv2 and ConvNeXt models.
    Implements multi-view generation (12 equidistant rotations) and caching mechanisms.
    """

    def __init__(self):
        # Ensure reproducibility
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        print(f"Initializing FeatureExtractor on {self.device}...")

        # Load DINOv2 Model (Global Geometry)
        print(f"Loading {Config.MODEL_DINOV2}...")
        self.dino_model = (
            timm.create_model(Config.MODEL_DINOV2, pretrained=True, num_classes=0)
            .to(self.device)
            .eval()
        )

        # Load ConvNeXt Model (Local Texture)
        print(f"Loading {Config.MODEL_CONVNEXT}...")
        self.conv_model = (
            timm.create_model(Config.MODEL_CONVNEXT, pretrained=True, num_classes=0)
            .to(self.device)
            .eval()
        )

        # Create Preprocessing Transforms
        # We use the configuration from DINOv2 as the baseline for both (usually 224x224, ImageNet Norm)
        # This handles Resize, CenterCrop, Normalize, and ToTensor
        data_config = timm.data.resolve_model_data_config(self.dino_model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def preprocess_views(self, image_path):
        """
        Loads an image and generates 12 equidistant rotated views.

        Args:
            image_path (str): Relative path to the image (e.g., 'images/10.jpg').

        Returns:
            torch.Tensor: Batch of preprocessed views with shape [12, 3, 224, 224].
        """
        full_path = os.path.join(Config.INPUT_DIR, image_path)

        # Load image (BGR)
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        views = []
        # Generate angles: 0, 30, 60, ..., 330
        angles = np.linspace(0, 360, Config.N_VIEWS, endpoint=False)

        for angle in angles:
            # Rotation Matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Warp with white border (255, 255, 255) to match background
            rotated_img = cv2.warpAffine(
                img,
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Convert to PIL for timm transforms compatibility
            pil_img = Image.fromarray(rotated_img)

            # Apply transforms (Resize -> CenterCrop -> Normalize -> ToTensor)
            tensor_img = self.transforms(pil_img)
            views.append(tensor_img)

        # Stack into a batch [N_VIEWS, C, H, W]
        return torch.stack(views)

    def extract_and_cache(
        self, metadata_path, dataset_name, load_cached_data=True, limit=None
    ):
        """
        Extracts features for all images in the metadata file and caches them.

        Args:
            metadata_path (str): Path to the metadata CSV (train/val/test).
            dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.
            limit (int, optional): Limit processing to N samples (for debugging).

        Returns:
            tuple: (dino_features, conv_features, ids)
                dino_features: np.ndarray [N, 12, 1024]
                conv_features: np.ndarray [N, 12, 1536]
                ids: np.ndarray [N]
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Define cache file paths
        cache_dino_path = os.path.join(Config.CACHE_DIR, f"{dataset_name}_dino.npy")
        cache_conv_path = os.path.join(Config.CACHE_DIR, f"{dataset_name}_conv.npy")
        cache_ids_path = os.path.join(Config.CACHE_DIR, f"{dataset_name}_ids.npy")

        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(cache_dino_path)
                and os.path.exists(cache_conv_path)
                and os.path.exists(cache_ids_path)
            ):

                print(
                    f"Loading cached features for '{dataset_name}' from {Config.CACHE_DIR}..."
                )
                dino_feats = np.load(cache_dino_path)
                conv_feats = np.load(cache_conv_path)
                ids = np.load(cache_ids_path)

                # If limit was requested, slice the cached data
                if limit is not None:
                    print(f"Limiting cached data to first {limit} samples.")
                    return dino_feats[:limit], conv_feats[:limit], ids[:limit]
                return dino_feats, conv_feats, ids

        # 2. Compute from scratch
        print(f"Starting feature extraction for '{dataset_name}'...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        if limit is not None:
            print(f"Limiting processing to first {limit} samples.")
            df = df.head(limit)

        dino_features_list = []
        conv_features_list = []
        ids_list = []

        # Iterate through images
        # We process one image (12 views) at a time.
        # This creates a batch of size 12 on the GPU, which is efficient enough.
        with torch.no_grad():
            for _, row in df.iterrows():
                img_path = row["file_path"]
                img_id = row["id"]

                # Get batch of 12 views
                # Shape: [12, 3, 224, 224]
                batch_views = self.preprocess_views(img_path).to(self.device)

                # Extract DINOv2 features
                # Shape: [12, 1024]
                dino_out = self.dino_model(batch_views).cpu().numpy()

                # Extract ConvNeXt features
                # Shape: [12, 1536]
                conv_out = self.conv_model(batch_views).cpu().numpy()

                dino_features_list.append(dino_out)
                conv_features_list.append(conv_out)
                ids_list.append(img_id)

        # Stack results
        # Final Shapes: [N_samples, 12, Feature_Dim]
        all_dino_features = np.stack(dino_features_list)
        all_conv_features = np.stack(conv_features_list)
        all_ids = np.array(ids_list)

        print(f"Extraction complete.")
        print(f"  DINOv2 Shape: {all_dino_features.shape}")
        print(f"  ConvNeXt Shape: {all_conv_features.shape}")

        # 3. Save to cache (only if not limited, to avoid overwriting full cache with partial data)
        if limit is None:
            print(f"Saving features to {Config.CACHE_DIR}...")
            np.save(cache_dino_path, all_dino_features)
            np.save(cache_conv_path, all_conv_features)
            np.save(cache_ids_path, all_ids)
        else:
            print("Skipping cache save (limit applied).")

        return all_dino_features, all_conv_features, all_ids
