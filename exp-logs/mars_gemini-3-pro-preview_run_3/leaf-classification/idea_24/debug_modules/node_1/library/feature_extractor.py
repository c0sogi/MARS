import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import timm
import library.config as cfg
import library.utils as utils

# Suppress warnings
warnings.filterwarnings("ignore")


class DualStreamExtractor:
    def __init__(self, device=cfg.DEVICE):
        """
        Initializes the DualStreamExtractor with DINOv2 and ConvNeXt models.
        """
        self.device = device
        self.img_size = cfg.IMG_SIZE

        print(f"Initializing Feature Extractor on {self.device}...")

        # Load DINOv2 Model (Global Geometry)
        print(f"Loading DINOv2 model: {cfg.MODEL_DINO}")
        self.dino_model = timm.create_model(
            cfg.MODEL_DINO, pretrained=True, num_classes=0
        ).to(self.device)
        self.dino_model.eval()

        # Load ConvNeXt Model (Local Texture)
        print(f"Loading ConvNeXt model: {cfg.MODEL_CONVNEXT}")
        self.convnext_model = timm.create_model(
            cfg.MODEL_CONVNEXT, pretrained=True, num_classes=0
        ).to(self.device)
        self.convnext_model.eval()

        # Data Transforms
        self.mean = cfg.IMAGENET_MEAN
        self.std = cfg.IMAGENET_STD

        self.to_tensor_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )

    def preprocess_image(self, img_path):
        """
        Reads an image and pads it to a square canvas based on its diagonal.
        This ensures that no part of the leaf is clipped during rotation.
        """
        full_path = os.path.join(cfg.INPUT_DIR, img_path)

        # Open and convert to RGB (standardizing input)
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a blank white image in case of error to avoid crash
            return Image.new("RGB", (224, 224), (255, 255, 255))

        w, h = img.size

        # Calculate diagonal length
        diagonal = int((w**2 + h**2) ** 0.5) + 1

        # Create white canvas (background color of dataset is white)
        new_img = Image.new("RGB", (diagonal, diagonal), (255, 255, 255))

        # Paste original image in center
        offset_x = (diagonal - w) // 2
        offset_y = (diagonal - h) // 2
        new_img.paste(img, (offset_x, offset_y))

        return new_img

    def get_rotated_crops(self, pil_img, n_rotations=cfg.N_ROTATIONS):
        """
        Generates N equidistant rotated versions of the image.
        Returns a batch tensor of shape (N, 3, H, W).
        """
        crops = []
        step = 360 / n_rotations

        for i in range(n_rotations):
            angle = i * step
            # Rotate the image. Since we padded to diagonal, we don't need to expand.
            # Fill with white (255) to match background.
            rot_img = F.rotate(pil_img, angle, fill=[255, 255, 255])

            # Resize to model input size
            rot_img = F.resize(rot_img, (self.img_size, self.img_size))

            # Convert to tensor and normalize
            tensor_img = self.to_tensor_norm(rot_img)
            crops.append(tensor_img)

        return torch.stack(crops)

    def extract_features_batch(self, image_batch):
        """
        Extracts and concatenates features from both models.
        Input: (B, 3, H, W)
        Output: (B, D_dino + D_convnext)
        """
        with torch.no_grad():
            # DINOv2 Features
            dino_feats = self.dino_model(image_batch)

            # ConvNeXt Features
            conv_feats = self.convnext_model(image_batch)

            # Concatenate features
            combined = torch.cat([dino_feats, conv_feats], dim=1)

        return combined.cpu().numpy()

    def process_dataset(self, df, desc="Processing"):
        """
        Iterates over the dataframe, processing each image to generate 36 views
        and extracting features.
        """
        all_img_features = []
        all_ids = []
        all_labels = []
        all_tab_features = []

        # Identify tabular columns
        margin_cols = [c for c in df.columns if c.startswith("margin")]
        shape_cols = [c for c in df.columns if c.startswith("shape")]
        texture_cols = [c for c in df.columns if c.startswith("texture")]
        tab_cols = margin_cols + shape_cols + texture_cols

        total = len(df)
        print(f"Starting {desc} for {total} images...")

        for idx, row in df.iterrows():
            if idx % 50 == 0 and idx > 0:
                print(f"  {desc}: Processed {idx}/{total} images")

            img_path = row["file_path"]
            img_id = row["id"]

            # Extract tabular features
            tab_feat = row[tab_cols].values.astype(np.float32)

            # Extract label if available
            label = row["species"] if "species" in row else "unknown"

            # 1. Preprocess (Pad)
            pil_img = self.preprocess_image(img_path)

            # 2. Generate 36 Rotated Views
            # Shape: (36, 3, 224, 224)
            crops_tensor = self.get_rotated_crops(pil_img)
            crops_tensor = crops_tensor.to(self.device)

            # 3. Extract Features
            # Shape: (36, Feature_Dim)
            features = self.extract_features_batch(crops_tensor)

            # 4. Accumulate
            all_img_features.append(features)
            all_ids.append(img_id)
            all_labels.append(label)
            all_tab_features.append(tab_feat)

        # Convert to numpy arrays
        # Shape: (N_samples, 36, Feature_Dim)
        final_img_features = np.stack(all_img_features, axis=0)
        final_ids = np.array(all_ids)
        final_labels = np.array(all_labels)
        # Shape: (N_samples, 192)
        final_tab_features = np.stack(all_tab_features, axis=0)

        print(f"Finished {desc}. Feature shape: {final_img_features.shape}")
        return final_img_features, final_ids, final_labels, final_tab_features

    def extract_all_rotations(self, load_cached_data=True):
        """
        Main pipeline function. Checks for cached data, and if not found,
        runs the full extraction process for both train (combined with val) and test sets.
        """
        # Define cache file paths
        cache_files = [
            cfg.CACHE_TRAIN_IMG_FEATURES,
            cfg.CACHE_TRAIN_IDS,
            cfg.CACHE_TRAIN_LABELS,
            cfg.CACHE_TRAIN_TAB_FEATURES,
            cfg.CACHE_TEST_IMG_FEATURES,
            cfg.CACHE_TEST_IDS,
            cfg.CACHE_TEST_TAB_FEATURES,
            cfg.CACHE_CLASSES,
        ]

        # Check if all cache files exist
        cache_exists = all(os.path.exists(f) for f in cache_files)

        if load_cached_data and cache_exists:
            print("Cache found. Loading features from disk...")
            train_img = np.load(cfg.CACHE_TRAIN_IMG_FEATURES)
            train_ids = np.load(cfg.CACHE_TRAIN_IDS)
            train_labels = np.load(cfg.CACHE_TRAIN_LABELS)
            train_tab = np.load(cfg.CACHE_TRAIN_TAB_FEATURES)

            test_img = np.load(cfg.CACHE_TEST_IMG_FEATURES)
            test_ids = np.load(cfg.CACHE_TEST_IDS)
            test_tab = np.load(cfg.CACHE_TEST_TAB_FEATURES)
            classes = np.load(cfg.CACHE_CLASSES)

            return {
                "train_img": train_img,
                "train_ids": train_ids,
                "train_labels": train_labels,
                "train_tab": train_tab,
                "test_img": test_img,
                "test_ids": test_ids,
                "test_tab": test_tab,
                "classes": classes,
            }

        print("Cache miss or force reload. Starting feature extraction pipeline...")

        # 1. Load Metadata
        # Combine Train and Val to form the full training set for Cross-Validation
        df_train_part = pd.read_csv(cfg.TRAIN_METADATA_PATH)
        df_val_part = pd.read_csv(cfg.VAL_METADATA_PATH)
        df_train_full = pd.concat([df_train_part, df_val_part], ignore_index=True)

        df_test = pd.read_csv(cfg.TEST_METADATA_PATH)

        # 2. Extract Training Features
        train_img, train_ids, train_labels, train_tab = self.process_dataset(
            df_train_full, desc="Train Extraction"
        )

        # 3. Extract Test Features
        test_img, test_ids, _, test_tab = self.process_dataset(
            df_test, desc="Test Extraction"
        )

        # 4. Extract Classes
        classes = np.unique(train_labels)

        # 5. Save to Cache
        print(f"Saving features to {cfg.WORKING_DIR}...")
        os.makedirs(cfg.WORKING_DIR, exist_ok=True)

        np.save(cfg.CACHE_TRAIN_IMG_FEATURES, train_img)
        np.save(cfg.CACHE_TRAIN_IDS, train_ids)
        np.save(cfg.CACHE_TRAIN_LABELS, train_labels)
        np.save(cfg.CACHE_TRAIN_TAB_FEATURES, train_tab)

        np.save(cfg.CACHE_TEST_IMG_FEATURES, test_img)
        np.save(cfg.CACHE_TEST_IDS, test_ids)
        np.save(cfg.CACHE_TEST_TAB_FEATURES, test_tab)
        np.save(cfg.CACHE_CLASSES, classes)

        print("Feature extraction complete.")

        return {
            "train_img": train_img,
            "train_ids": train_ids,
            "train_labels": train_labels,
            "train_tab": train_tab,
            "test_img": test_img,
            "test_ids": test_ids,
            "test_tab": test_tab,
            "classes": classes,
        }
