import os
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    setup_logger,
    load_image,
    rotate_image,
    save_numpy,
    load_numpy,
)


class FeatureExtractor:
    """
    Handles the loading of pretrained models (DINOv2, ConvNeXt),
    image preprocessing (rotation, resizing), and feature extraction.
    Implements caching to avoid redundant computation.
    """

    def __init__(self):
        self.logger = setup_logger("FeatureExtractor")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Initializing FeatureExtractor on device: {self.device}")

        # Load Models
        self.logger.info(f"Loading DINOv2 model: {Config.MODEL_DINO}")
        self.model_dino = timm.create_model(
            Config.MODEL_DINO, pretrained=True, num_classes=0
        )
        self.model_dino.to(self.device)
        self.model_dino.eval()

        self.logger.info(f"Loading ConvNeXt model: {Config.MODEL_CONVNEXT}")
        self.model_convnext = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        )
        self.model_convnext.to(self.device)
        self.model_convnext.eval()

        # Define Transforms
        # Standard ImageNet normalization and resizing
        self.transform = transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_cache_filenames(self, dataset_type):
        """Returns the cache filenames based on dataset type."""
        if dataset_type == "train":
            return {
                "features": Config.CACHE_TRAIN_IMG_FEATURES,
                "tabular": Config.CACHE_TRAIN_TABULAR,
                "ids": Config.CACHE_TRAIN_IDS,
                "labels": Config.CACHE_TRAIN_LABELS,
                "classes": Config.CACHE_CLASSES,
            }
        elif dataset_type == "test":
            return {
                "features": Config.CACHE_TEST_IMG_FEATURES,
                "tabular": Config.CACHE_TEST_TABULAR,
                "ids": Config.CACHE_TEST_IDS,
            }
        else:
            raise ValueError("dataset_type must be 'train' or 'test'")

    def _load_cached(self, cache_files):
        """Attempts to load all files in the cache_files dictionary."""
        data = {}
        for key, filename in cache_files.items():
            loaded = load_numpy(filename)
            if loaded is None:
                return None
            data[key] = loaded
        return data

    def _extract_tabular_features(self, df):
        """Extracts the 192 tabular features from the dataframe."""
        # Identify feature columns
        margin_cols = [c for c in df.columns if c.startswith("margin")]
        shape_cols = [c for c in df.columns if c.startswith("shape")]
        texture_cols = [c for c in df.columns if c.startswith("texture")]
        feature_cols = margin_cols + shape_cols + texture_cols

        # Ensure correct order and presence
        if len(feature_cols) != 192:
            self.logger.warning(
                f"Expected 192 tabular features, found {len(feature_cols)}. "
                "Check column names in metadata."
            )

        return df[feature_cols].values.astype(np.float32)

    def process_image_batch(self, image_path):
        """
        Loads an image, generates 12 rotated views, and prepares a tensor batch.
        Returns: Tensor of shape (12, 3, H, W)
        """
        # Load image (BGR)
        img_bgr = load_image(image_path)

        # Convert to RGB for PIL/Transforms
        img_rgb = cv2_img = img_bgr[:, :, ::-1]

        tensors = []
        for angle in Config.ROTATION_ANGLES:
            # Rotate (using the utility that handles padding)
            # Note: rotate_image expects numpy array
            rotated_np = rotate_image(img_rgb, angle)

            # Convert to PIL for transforms
            rotated_pil = Image.fromarray(rotated_np)

            # Apply transforms (Resize, Normalize, ToTensor)
            tensor = self.transform(rotated_pil)
            tensors.append(tensor)

        # Stack into batch: (12, 3, 224, 224)
        return torch.stack(tensors)

    def extract_features(self, dataset_type="train", load_cached_data=True):
        """
        Main method to extract features for the specified dataset.

        Args:
            dataset_type (str): 'train' or 'test'.
            load_cached_data (bool): If True, attempts to load from disk first.

        Returns:
            dict: Dictionary containing 'features', 'tabular', 'ids', and optionally 'labels'.
        """
        cache_files = self._get_cache_filenames(dataset_type)

        # 1. Try Loading Cache
        if load_cached_data:
            self.logger.info(f"Attempting to load cached data for {dataset_type}...")
            cached_data = self._load_cached(cache_files)
            if cached_data is not None:
                self.logger.info("Cache hit. Data loaded successfully.")
                return cached_data
            else:
                self.logger.info("Cache miss or incomplete. Starting extraction...")

        # 2. Prepare for Extraction
        metadata_path = (
            Config.TRAIN_METADATA_PATH
            if dataset_type == "train"
            else Config.TEST_METADATA_PATH
        )
        df = pd.read_csv(metadata_path)

        # Debugging: Subset data if enabled
        if Config.DEBUG:
            self.logger.info(
                f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows."
            )
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        # Initialize containers
        all_img_features = []
        all_ids = []
        all_labels = []

        # Extract Tabular Data upfront
        tabular_features = self._extract_tabular_features(df)

        # 3. Extraction Loop
        self.logger.info(f"Starting feature extraction for {len(df)} images...")

        # We process one image (12 views) at a time.
        # 12 views fits easily in GPU memory.

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Extracting {dataset_type}"
        ):
            img_id = row["id"]
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Prepare batch of 12 rotated views
            try:
                img_batch = self.process_image_batch(full_path)  # (12, 3, 224, 224)
                img_batch = img_batch.to(self.device)

                with torch.no_grad():
                    # Stream 1: DINOv2
                    feats_dino = self.model_dino(img_batch)  # (12, 1024)

                    # Stream 2: ConvNeXt
                    feats_conv = self.model_convnext(img_batch)  # (12, 1536)

                    # Concatenate features
                    # Shape: (12, 2560)
                    combined = torch.cat([feats_dino, feats_conv], dim=1)

                all_img_features.append(combined.cpu().numpy())
                all_ids.append(img_id)

                if dataset_type == "train":
                    all_labels.append(row["species"])

            except Exception as e:
                self.logger.error(
                    f"Error processing image {img_id} at {full_path}: {e}"
                )
                # In case of error, we might skip or fill with zeros.
                # For this competition, we assume data integrity but logging is good.
                continue

        # 4. Finalize and Save
        all_img_features = np.array(all_img_features, dtype=np.float32)  # (N, 12, 2560)
        all_ids = np.array(all_ids)

        output_data = {
            "features": all_img_features,
            "tabular": tabular_features,
            "ids": all_ids,
        }

        save_numpy(all_img_features, cache_files["features"])
        save_numpy(tabular_features, cache_files["tabular"])
        save_numpy(all_ids, cache_files["ids"])

        if dataset_type == "train":
            all_labels = np.array(all_labels)
            classes = np.unique(all_labels)

            output_data["labels"] = all_labels
            output_data["classes"] = classes

            save_numpy(all_labels, cache_files["labels"])
            save_numpy(classes, cache_files["classes"])

        self.logger.info(
            f"Extraction complete. Features shape: {all_img_features.shape}"
        )
        return output_data
