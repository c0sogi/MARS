import os
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from library.config import Config


class FeatureExtractor:
    """
    Handles the extraction of visual features using DINOv2 and ConvNeXt models.
    Implements Multi-View extraction (12 rotations) and caching.
    """

    def __init__(self):
        """
        Initialize models and transforms.
        """
        self.device = Config.DEVICE
        self.image_size = Config.IMAGE_SIZE
        self.rotation_angles = Config.ROTATION_ANGLES

        # Define normalization transform (ImageNet stats)
        # We separate Resize logic to handle it explicitly before rotation
        self.transform_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load DINOv2 (ViT-Large)
        print(f"Loading DINOv2 model: {Config.MODEL_DINO}")
        self.model_dino = timm.create_model(
            Config.MODEL_DINO, pretrained=True, num_classes=0, img_size=self.image_size
        ).to(self.device)
        self.model_dino.eval()

        # Load ConvNeXt Large
        print(f"Loading ConvNeXt model: {Config.MODEL_CONVNEXT}")
        self.model_conv = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        ).to(self.device)
        self.model_conv.eval()

    def _load_and_process_image(self, image_path):
        """
        Loads an image, resizes it, generates 12 rotated views, and applies normalization.

        Args:
            image_path (str): Relative path to the image (e.g., 'images/1.jpg').

        Returns:
            torch.Tensor: Batch of 12 views with shape (12, 3, 224, 224).
        """
        full_path = os.path.join(Config.INPUT_DIR, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Load image and ensure RGB (dataset is binary black/white but models need 3 channels)
        img = Image.open(full_path).convert("RGB")

        # Resize to target size first.
        # We resize before rotation to ensure consistent canvas size (224x224).
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        views = []
        for angle in self.rotation_angles:
            # Rotate the image.
            # expand=False keeps the size fixed at 224x224.
            # fillcolor=(255, 255, 255) ensures the background remains white after rotation.
            img_rot = img.rotate(
                angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
            )

            # Convert to tensor and normalize
            img_tensor = self.transform_tensor(img_rot)
            views.append(img_tensor)

        # Stack into a single batch: (12, 3, 224, 224)
        return torch.stack(views)

    def extract_features(self, metadata_path, cache_prefix, load_cached_data=True):
        """
        Extracts features for all images listed in the metadata CSV.
        Checks for cached .npy files first.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (ids, dino_features, conv_features)
                ids (np.array): Shape (N,)
                dino_features (np.array): Shape (N, 12, 1024)
                conv_features (np.array): Shape (N, 12, 1536)
        """
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        path_ids = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")
        path_dino = os.path.join(cache_dir, f"{cache_prefix}_dino.npy")
        path_conv = os.path.join(cache_dir, f"{cache_prefix}_conv.npy")

        # 1. Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(path_ids)
                and os.path.exists(path_dino)
                and os.path.exists(path_conv)
            ):
                print(f"[{cache_prefix}] Loading features from cache: {cache_dir}")
                ids = np.load(path_ids)
                dino_feats = np.load(path_dino)
                conv_feats = np.load(path_conv)
                return ids, dino_feats, conv_feats
            else:
                print(f"[{cache_prefix}] Cache miss. Starting feature extraction...")
        else:
            print(
                f"[{cache_prefix}] Force re-computation. Starting feature extraction..."
            )

        # 2. Compute features
        df = pd.read_csv(metadata_path)

        # Debugging: Limit sample size if configured
        if Config.DEBUG_SAMPLE_LIMIT is not None:
            print(
                f"[{cache_prefix}] DEBUG MODE: Limiting to {Config.DEBUG_SAMPLE_LIMIT} samples."
            )
            # Cite debug_lesson_12: Preserve Statistical Invariants in Debug Subsets
            if "species" in df.columns:
                df = df.sort_values("species")
                df = df.groupby("species").head(Config.N_SPLITS)
                df = df.head(Config.DEBUG_SAMPLE_LIMIT)
                df = df.groupby("species").filter(lambda x: len(x) >= Config.N_SPLITS)
            else:
                df = df.head(Config.DEBUG_SAMPLE_LIMIT)

        ids_list = []
        dino_list = []
        conv_list = []

        print(f"[{cache_prefix}] Processing {len(df)} images...")

        with torch.no_grad():
            for i, row in df.iterrows():
                img_id = row["id"]
                rel_path = row["file_path"]

                # Get 12 views (Shape: 12, 3, 224, 224)
                # We process one image (12 views) at a time to be memory safe on GPU
                batch_imgs = self._load_and_process_image(rel_path).to(self.device)

                # DINOv2 Inference (Shape: 12, 1024)
                feat_dino = self.model_dino(batch_imgs).cpu().numpy()

                # ConvNeXt Inference (Shape: 12, 1536)
                feat_conv = self.model_conv(batch_imgs).cpu().numpy()

                ids_list.append(img_id)
                dino_list.append(feat_dino)
                conv_list.append(feat_conv)

                if (i + 1) % 50 == 0:
                    print(f"[{cache_prefix}] Processed {i + 1}/{len(df)}")

        # Convert to numpy arrays
        ids_arr = np.array(ids_list)
        dino_arr = np.array(dino_list)
        conv_arr = np.array(conv_list)

        # 3. Save to cache
        np.save(path_ids, ids_arr)
        np.save(path_dino, dino_arr)
        np.save(path_conv, conv_arr)
        print(f"[{cache_prefix}] Saved features to {cache_dir}")

        return ids_arr, dino_arr, conv_arr

    def run(self, load_cached_data=True):
        """
        Executes the extraction pipeline for Train, Validation, and Test sets.

        Args:
            load_cached_data (bool): Whether to use cached data.

        Returns:
            dict: Dictionary containing tuples of (ids, dino, conv) for each split.
        """
        print("Initializing Feature Extraction Pipeline...")

        # Extract Train
        train_data = self.extract_features(
            Config.TRAIN_METADATA, "train", load_cached_data=load_cached_data
        )

        # Extract Validation
        val_data = self.extract_features(
            Config.VAL_METADATA, "val", load_cached_data=load_cached_data
        )

        # Extract Test
        test_data = self.extract_features(
            Config.TEST_METADATA, "test", load_cached_data=load_cached_data
        )

        return {"train": train_data, "val": val_data, "test": test_data}
