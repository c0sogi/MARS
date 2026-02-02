import os
import numpy as np
import pandas as pd
import torch
import timm
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from library.config import Config
from library.utils import load_data, get_logger, seed_everything


class DeepFeatureExtractor:
    """
    Handles the extraction of deep features (DINOv2, ConvNeXt) and tabular features
    with support for multi-view rotation (Manifold Densification).
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.logger = get_logger("feature_extractor")

        # Standard ImageNet normalization required for pre-trained models
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._init_models()

    def _init_models(self):
        """Initializes the pre-trained models in evaluation mode."""
        self.logger.info(f"Initializing models on {self.device}...")

        # Load DINOv2 (ViT-Large)
        # num_classes=0 returns the pooled feature vector (CLS token for ViT)
        self.dino = timm.create_model(
            Config.DINO_MODEL_NAME,
            pretrained=True,
            num_classes=0,
            img_size=Config.IMG_SIZE,
        )
        self.dino.to(self.device)
        self.dino.eval()

        # Load ConvNeXt Large
        # num_classes=0 returns the global average pooled feature
        self.convnext = timm.create_model(
            Config.CONVNEXT_MODEL_NAME, pretrained=True, num_classes=0
        )
        self.convnext.to(self.device)
        self.convnext.eval()

    def _get_rotations(self, rel_path):
        """
        Generates 12 equidistant rotated views of the image.
        Handles padding to ensure no cropping of the leaf occurs.

        Args:
            rel_path (str): Relative path to the image file.

        Returns:
            torch.Tensor: Batch of rotated images (12, 3, 224, 224).
        """
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pad to square (white background) to prevent cropping when rotating
        h, w = img.shape[:2]
        dim = max(h, w)

        # Create white canvas (255)
        square_img = np.full((dim, dim, 3), 255, dtype=np.uint8)

        # Center the original image on the canvas
        x_off = (dim - w) // 2
        y_off = (dim - h) // 2
        square_img[y_off : y_off + h, x_off : x_off + w] = img

        rotated_tensors = []
        center = (dim // 2, dim // 2)

        # Generate rotations
        for i in range(Config.N_ROTATIONS):
            angle = i * (360.0 / Config.N_ROTATIONS)

            # Rotate
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                square_img,
                M,
                (dim, dim),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Resize to model input size
            resized = cv2.resize(
                rotated,
                (Config.IMG_SIZE, Config.IMG_SIZE),
                interpolation=cv2.INTER_AREA,
            )

            # Convert to tensor and normalize
            tensor = self.transform(Image.fromarray(resized))
            rotated_tensors.append(tensor)

        # Stack into a batch: (12, 3, 224, 224)
        return torch.stack(rotated_tensors)

    def extract_features(self, split, load_cached_data=True):
        """
        Extracts features for a given dataset split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from disk if available.

        Returns:
            dict: Dictionary containing numpy arrays for features, ids, and labels.
        """
        # Construct cache filenames
        # We append _debug to the filename if running in debug mode to avoid collisions
        suffix = "_debug" if Config.DEBUG else ""
        cache_base = os.path.join(Config.CACHE_DIR, f"{split}{suffix}")

        paths = {
            "dino": f"{cache_base}_dino.npy",
            "conv": f"{cache_base}_conv.npy",
            "tab": f"{cache_base}_tab.npy",
            "ids": f"{cache_base}_ids.npy",
            "labels": f"{cache_base}_labels.npy",
        }

        # 1. Try Loading from Cache
        required_keys = ["dino", "conv", "tab", "ids"]
        if split != "test":
            required_keys.append("labels")

        if load_cached_data:
            if all(os.path.exists(paths[k]) for k in required_keys):
                self.logger.info(
                    f"Loading cached features for '{split}' from {Config.CACHE_DIR}..."
                )
                data = {
                    "dino_features": np.load(paths["dino"]),
                    "conv_features": np.load(paths["conv"]),
                    "tabular_features": np.load(paths["tab"]),
                    "ids": np.load(paths["ids"]),
                }
                if split != "test":
                    data["labels"] = np.load(paths["labels"])
                return data

        # 2. Compute Features
        self.logger.info(f"Starting feature extraction for '{split}'...")

        # Load metadata
        df = load_data(split, load_cached_data=load_cached_data, debug=Config.DEBUG)

        # Identify tabular columns
        tab_cols = [
            c for c in df.columns if c.startswith(("margin", "shape", "texture"))
        ]
        if len(tab_cols) != 192:
            self.logger.warning(
                f"Expected 192 tabular features, found {len(tab_cols)}."
            )

        # Storage
        dino_feats_list = []
        conv_feats_list = []
        tab_feats_list = []
        ids_list = []
        labels_list = []

        # Iterate
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
            # --- Visual Features ---
            try:
                # Get batch of 12 rotations
                img_batch = self._get_rotations(row["file_path"])
                img_batch = img_batch.to(self.device)

                with torch.no_grad():
                    # DINOv2 Inference
                    d_emb = self.dino(img_batch)  # Shape: (12, Embed_Dim)
                    dino_feats_list.append(d_emb.cpu().numpy())

                    # ConvNeXt Inference
                    c_emb = self.convnext(img_batch)  # Shape: (12, Embed_Dim)
                    conv_feats_list.append(c_emb.cpu().numpy())

            except Exception as e:
                self.logger.error(f"Failed to process image {row['file_path']}: {e}")
                raise e

            # --- Tabular Features ---
            # Ensure float32 for consistency
            tab_feats_list.append(row[tab_cols].values.astype(np.float32))

            # --- Metadata ---
            ids_list.append(row["id"])
            if "species" in row:
                labels_list.append(row["species"])

        # Convert to numpy arrays
        dino_arr = np.stack(dino_feats_list)  # (N, 12, D_dino)
        conv_arr = np.stack(conv_feats_list)  # (N, 12, D_conv)
        tab_arr = np.stack(tab_feats_list)  # (N, 192)
        ids_arr = np.array(ids_list)

        # 3. Save to Cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.save(paths["dino"], dino_arr)
        np.save(paths["conv"], conv_arr)
        np.save(paths["tab"], tab_arr)
        np.save(paths["ids"], ids_arr)

        result = {
            "dino_features": dino_arr,
            "conv_features": conv_arr,
            "tabular_features": tab_arr,
            "ids": ids_arr,
        }

        if labels_list:
            labels_arr = np.array(labels_list)
            np.save(paths["labels"], labels_arr)
            result["labels"] = labels_arr

        self.logger.info(f"Feature extraction complete. Saved to {Config.CACHE_DIR}")
        return result
