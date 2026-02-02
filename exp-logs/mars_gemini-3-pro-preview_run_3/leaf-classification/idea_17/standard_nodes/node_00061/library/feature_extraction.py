import os
import numpy as np
import pandas as pd
import cv2
import torch
import timm
from PIL import Image
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles loading of images, generation of multi-view rotations,
    and extraction of features using DINOv2 and ConvNeXt backbones.
    Also handles loading and caching of tabular features.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.image_size = Config.IMAGE_SIZE

        # Standard ImageNet normalization
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_models(self):
        """
        Initializes the DINOv2 and ConvNeXt models in evaluation mode.
        """
        print(f"Initializing models on {self.device}...")

        # DINOv2 (ViT-Large)
        # num_classes=0 removes the classifier and returns the pooled features (usually CLS token for ViT)
        # Cite debug_lesson_5: Explicitly configure resolution for ViT to trigger position embedding interpolation
        model_dino = timm.create_model(
            Config.MODEL_DINO,
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )
        model_dino = model_dino.to(self.device)
        model_dino.eval()

        # ConvNeXt Large
        model_conv = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        )
        model_conv = model_conv.to(self.device)
        model_conv.eval()

        return model_dino, model_conv

    def _process_image_views(self, image_rel_path):
        """
        Reads an image, resizes it to Config.IMAGE_SIZE, and generates
        Config.NUM_ROTATIONS views by rotating.

        Args:
            image_rel_path (str): Relative path to the image (e.g., 'images/1.jpg')

        Returns:
            torch.Tensor: Batch of rotated images, shape (NUM_ROTATIONS, 3, H, W)
        """
        full_path = os.path.join(Config.INPUT_DIR, image_rel_path)

        # Read image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size (224x224)
        # We resize before rotation to ensure consistent input size.
        img = cv2.resize(
            img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        views = []

        for angle in Config.ROTATION_ANGLES:
            if angle == 0:
                rotated = img.copy()
            else:
                # Rotate
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Use white border (255, 255, 255) as these are leaf images on white background
                rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

            # Convert to Tensor and Normalize
            # PIL is needed for ToTensor to handle range [0, 255] -> [0, 1] correctly automatically
            pil_img = Image.fromarray(rotated)
            tensor_img = self.tensor_transform(pil_img)
            views.append(tensor_img)

        # Stack into a batch: (36, 3, 224, 224)
        return torch.stack(views)

    def _extract_features_from_df(self, df, model_dino, model_conv, desc="Data"):
        """
        Iterates over the dataframe, extracts image features for all views,
        and collects tabular features and labels.
        """
        dino_features = []
        conv_features = []
        tabular_features = []
        ids = []
        labels = []

        # Identify tabular columns
        # Filter columns that start with 'margin', 'shape', 'texture'
        tab_cols = [
            c
            for c in df.columns
            if any(c.startswith(p) for p in Config.TABULAR_PREFIXES)
        ]

        # Processing loop
        # We process one image (36 views) at a time.
        # 36 views fit comfortably in A100 GPU memory.
        print(f"Extracting features for {desc} ({len(df)} samples)...")

        with torch.no_grad():
            for idx, row in df.iterrows():
                # 1. Image Processing
                img_path = row["file_path"]
                views_batch = self._process_image_views(img_path)
                views_batch = views_batch.to(self.device)

                # 2. Model Inference
                # DINOv2
                feat_d = model_dino(views_batch)  # Shape: (36, Embed_Dim)
                dino_features.append(feat_d.cpu().numpy())

                # ConvNeXt
                feat_c = model_conv(views_batch)  # Shape: (36, Embed_Dim)
                conv_features.append(feat_c.cpu().numpy())

                # 3. Tabular Data
                # Ensure float32
                tab_data = row[tab_cols].values.astype(np.float32)
                tabular_features.append(tab_data)

                # 4. Metadata
                ids.append(row["id"])
                if "species" in row:
                    labels.append(row["species"])
                else:
                    labels.append(None)

        return (
            np.array(dino_features),  # (N, 36, D1)
            np.array(conv_features),  # (N, 36, D2)
            np.array(tabular_features),  # (N, 192)
            np.array(ids),  # (N,)
            np.array(labels),  # (N,)
        )

    def extract_all(self, load_cached_data=True):
        """
        Main entry point. Loads metadata, extracts features (or loads from cache),
        and returns a dictionary containing all data.

        Args:
            load_cached_data (bool): If True, attempts to load from .npy files.

        Returns:
            dict: Dictionary with keys 'train_dino', 'train_conv', 'train_tab', 'train_ids', 'train_lbl', etc.
        """
        Config.setup_directories()

        # Define cache file paths
        cache_paths = {
            "train_dino": os.path.join(Config.CACHE_DIR, "train_dino.npy"),
            "train_conv": os.path.join(Config.CACHE_DIR, "train_conv.npy"),
            "train_tab": os.path.join(Config.CACHE_DIR, "train_tab.npy"),
            "train_ids": os.path.join(Config.CACHE_DIR, "train_ids.npy"),
            "train_lbl": os.path.join(Config.CACHE_DIR, "train_lbl.npy"),
            "val_dino": os.path.join(Config.CACHE_DIR, "val_dino.npy"),
            "val_conv": os.path.join(Config.CACHE_DIR, "val_conv.npy"),
            "val_tab": os.path.join(Config.CACHE_DIR, "val_tab.npy"),
            "val_ids": os.path.join(Config.CACHE_DIR, "val_ids.npy"),
            "val_lbl": os.path.join(Config.CACHE_DIR, "val_lbl.npy"),
            "test_dino": os.path.join(Config.CACHE_DIR, "test_dino.npy"),
            "test_conv": os.path.join(Config.CACHE_DIR, "test_conv.npy"),
            "test_tab": os.path.join(Config.CACHE_DIR, "test_tab.npy"),
            "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
        }

        # Check cache existence
        cache_exists = all(os.path.exists(p) for p in cache_paths.values())

        if load_cached_data and cache_exists:
            print("Loading features from cache...")
            data = {}
            for key, path in cache_paths.items():
                # Allow pickle for object arrays (labels)
                data[key] = np.load(path, allow_pickle=True)
            return data

        # If not cached, compute from scratch
        print("Cache not found or forced reload. Extracting features...")

        # Load metadata
        df_train = pd.read_csv(Config.METADATA_TRAIN)
        df_val = pd.read_csv(Config.METADATA_VAL)
        df_test = pd.read_csv(Config.METADATA_TEST)

        # Initialize models
        model_dino, model_conv = self._get_models()

        # Extract features
        train_dino, train_conv, train_tab, train_ids, train_lbl = (
            self._extract_features_from_df(df_train, model_dino, model_conv, "Train")
        )

        val_dino, val_conv, val_tab, val_ids, val_lbl = self._extract_features_from_df(
            df_val, model_dino, model_conv, "Validation"
        )

        test_dino, test_conv, test_tab, test_ids, _ = self._extract_features_from_df(
            df_test, model_dino, model_conv, "Test"
        )

        # Construct data dictionary
        data = {
            "train_dino": train_dino,
            "train_conv": train_conv,
            "train_tab": train_tab,
            "train_ids": train_ids,
            "train_lbl": train_lbl,
            "val_dino": val_dino,
            "val_conv": val_conv,
            "val_tab": val_tab,
            "val_ids": val_ids,
            "val_lbl": val_lbl,
            "test_dino": test_dino,
            "test_conv": test_conv,
            "test_tab": test_tab,
            "test_ids": test_ids,
        }

        # Save to cache
        print("Saving features to cache...")
        for key, path in cache_paths.items():
            np.save(path, data[key])

        return data
