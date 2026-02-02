import os
import numpy as np
import pandas as pd
import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image
import logging

from library.config import Config
from library.utils import seed_everything


class DualStreamExtractor:
    """
    Extracts features using a Dual-Stream architecture (DINOv2 + ConvNeXt)
    and processes tabular data. Implements Manifold Densification by extracting
    features for multiple rotated views of each image.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self._init_models()
        self._init_transforms()

    def _init_models(self):
        """Initializes the DINOv2 and ConvNeXt models."""
        logging.info(f"Loading models on {self.device}...")

        # Global Geometry Stream: DINOv2
        # num_classes=0 returns the feature embedding (CLS token for ViT)
        self.dino = timm.create_model(
            self.config.MODEL_DINO, pretrained=True, num_classes=0
        ).to(self.device)
        self.dino.eval()

        # Local Texture Stream: ConvNeXt
        # num_classes=0 returns the pooled feature embedding
        self.convnext = timm.create_model(
            self.config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        ).to(self.device)
        self.convnext.eval()

    def _init_transforms(self):
        """Defines the image preprocessing transforms."""
        # Standard ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.config.IMG_SIZE, self.config.IMG_SIZE), antialias=True
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _rotate_image(self, image, angle):
        """
        Rotates an image by a specific angle around its center.
        Fills borders with white (255) to match the leaf background.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        # Scale=1.0 maintains size
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Border constant white
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated

    def process_dataset(self, metadata_path, dataset_name, load_cached_data=True):
        """
        Processes a dataset defined by the metadata CSV.
        Extracts visual features for 12 rotations and loads tabular features.

        Args:
            metadata_path (str): Path to the metadata CSV.
            dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing numpy arrays for ids, features, and labels.
        """
        # Define cache paths
        cache_prefix = os.path.join(self.config.WORKING_DIR, dataset_name)
        paths = {
            "ids": f"{cache_prefix}_ids.npy",
            "dino": f"{cache_prefix}_dino.npy",
            "conv": f"{cache_prefix}_conv.npy",
            "tab": f"{cache_prefix}_tab.npy",
            "labels": f"{cache_prefix}_labels.npy",
        }

        # Attempt to load from cache
        if load_cached_data:
            # Check if essential feature files exist
            if (
                os.path.exists(paths["ids"])
                and os.path.exists(paths["dino"])
                and os.path.exists(paths["conv"])
                and os.path.exists(paths["tab"])
            ):

                logging.info(
                    f"Loading cached features for '{dataset_name}' from {self.config.WORKING_DIR}..."
                )
                data = {
                    "ids": np.load(paths["ids"]),
                    "dino": np.load(paths["dino"]),
                    "conv": np.load(paths["conv"]),
                    "tab": np.load(paths["tab"]),
                }
                if os.path.exists(paths["labels"]):
                    data["labels"] = np.load(paths["labels"])
                return data

        # Process from scratch
        logging.info(f"Extracting features for '{dataset_name}' from source...")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Limit dataset for debugging if configured
        if self.config.LIMIT_DATASET:
            logging.info(
                f"Limiting {dataset_name} dataset to {self.config.LIMIT_DATASET} samples."
            )
            df = df.head(self.config.LIMIT_DATASET)

        # Identify tabular columns
        tab_cols = []
        for prefix in self.config.TABULAR_PREFIXES:
            tab_cols.extend([c for c in df.columns if c.startswith(prefix)])

        # Containers
        all_ids = []
        all_labels = []
        all_dino_feats = []
        all_conv_feats = []
        all_tab_feats = []

        has_labels = "species" in df.columns

        # Iterate over samples
        # Note: Using a loop here is feasible because N is small (~1000).
        for _, row in df.iterrows():
            # 1. Store ID and Label
            all_ids.append(row["id"])
            if has_labels:
                all_labels.append(row["species"])

            # 2. Extract Tabular Features
            # Ensure float32 for consistency
            tab_feat = row[tab_cols].values.astype(np.float32)
            all_tab_feats.append(tab_feat)

            # 3. Process Image (Visual Features)
            img_rel_path = row["file_path"]
            img_full_path = os.path.join(self.config.INPUT_DIR, img_rel_path)

            # Read image (BGR)
            img = cv2.imread(img_full_path)
            if img is None:
                # Fallback for missing images (should not happen based on metadata check)
                logging.warning(
                    f"Image not found: {img_full_path}. Using white placeholder."
                )
                img = np.full((224, 224, 3), 255, dtype=np.uint8)

            # Generate Rotations
            batch_tensors = []
            for angle in self.config.ROTATION_ANGLES:
                # Rotate
                rot_img_bgr = self._rotate_image(img, angle)
                # Convert BGR to RGB
                rot_img_rgb = cv2.cvtColor(rot_img_bgr, cv2.COLOR_BGR2RGB)
                # Convert to PIL
                pil_img = Image.fromarray(rot_img_rgb)
                # Transform
                tensor_img = self.transform(pil_img)
                batch_tensors.append(tensor_img)

            # Stack into batch: (12, 3, 224, 224)
            batch_input = torch.stack(batch_tensors).to(self.device)

            # Inference
            with torch.no_grad():
                # Extract DINO features
                dino_out = (
                    self.dino(batch_input).cpu().numpy()
                )  # Shape: (12, Embed_Dim)
                # Extract ConvNeXt features
                conv_out = (
                    self.convnext(batch_input).cpu().numpy()
                )  # Shape: (12, Embed_Dim)

            all_dino_feats.append(dino_out)
            all_conv_feats.append(conv_out)

        # Convert to NumPy arrays
        ids_np = np.array(all_ids)
        dino_np = np.array(all_dino_feats)  # Shape: (N, 12, D_dino)
        conv_np = np.array(all_conv_feats)  # Shape: (N, 12, D_conv)
        tab_np = np.array(all_tab_feats)  # Shape: (N, 192)

        data = {"ids": ids_np, "dino": dino_np, "conv": conv_np, "tab": tab_np}

        if has_labels:
            labels_np = np.array(all_labels)
            data["labels"] = labels_np

        # Save to cache
        logging.info(f"Saving extracted features to {self.config.WORKING_DIR}...")
        np.save(paths["ids"], ids_np)
        np.save(paths["dino"], dino_np)
        np.save(paths["conv"], conv_np)
        np.save(paths["tab"], tab_np)
        if has_labels:
            np.save(paths["labels"], labels_np)

        return data
