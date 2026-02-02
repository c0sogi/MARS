import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from library.config import Config
from library.utils import setup_logger, load_metadata

# Initialize logger
logger = setup_logger("feature_extraction")


class DualStreamExtractor:
    """
    Handles the dual-stream feature extraction using DINOv2 and ConvNeXt.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device

        logger.info(f"Initializing models on {self.device}...")

        # 1. Global Geometry Stream: DINOv2 (ViT-Large)
        self.dino_model = timm.create_model(
            Config.MODEL_DINO_NAME, pretrained=True, num_classes=0  # Get feature vector
        )
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # 2. Local Texture Stream: ConvNeXt Large
        self.convnext_model = timm.create_model(
            Config.MODEL_CONVNEXT_NAME,
            pretrained=True,
            num_classes=0,  # Get feature vector
        )
        self.convnext_model.to(self.device)
        self.convnext_model.eval()

        # Preprocessing transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.MEAN, std=Config.STD),
            ]
        )

        logger.info("Models initialized successfully.")

    def _rotate_image(self, image, angle):
        """
        Rotates an image by a specific angle around its center.
        Fills the border with white (255, 255, 255).
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        # Border value is white because background is white
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated

    def process_image(self, image_path):
        """
        Loads an image, generates 12 rotated views, and extracts features.

        Returns:
            np.ndarray: Shape (12, D_dino + D_convnext)
        """
        # Load image (OpenCV loads as BGR)
        full_path = os.path.join(Config.INPUT_DIR, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        img = cv2.imread(full_path)
        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")

        # Convert BGR to RGB for models
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        batch_tensors = []

        # Generate rotated views
        for angle in Config.ROTATION_ANGLES:
            rot_img = self._rotate_image(img, angle)
            tensor = self.transform(rot_img)
            batch_tensors.append(tensor)

        # Stack into a batch: (12, 3, H, W)
        batch = torch.stack(batch_tensors).to(self.device)

        # Inference
        with torch.no_grad():
            # Extract features
            feats_dino = self.dino_model(batch)  # (12, 1024)
            feats_conv = self.convnext_model(batch)  # (12, 1536)

            # Concatenate features: (12, 2560)
            combined = torch.cat([feats_dino, feats_conv], dim=1)

        return combined.cpu().numpy()


def extract_dataset(split: str, load_cached_data: bool = True):
    """
    Extracts features for the specified dataset split (train/val/test).
    Handles caching to disk.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (img_features, tab_features, ids, labels)
            - img_features: (N, 12, Feature_Dim)
            - tab_features: (N, 192)
            - ids: (N,)
            - labels: (N,) or None for test
    """
    # Determine cache paths based on split
    if split == "train":
        path_img = Config.CACHE_TRAIN_IMG_FEATURES
        path_tab = Config.CACHE_TRAIN_TAB_FEATURES
        path_ids = Config.CACHE_TRAIN_IDS
        path_lbl = Config.CACHE_TRAIN_LABELS
    elif split == "val":
        path_img = Config.CACHE_VAL_IMG_FEATURES
        path_tab = Config.CACHE_VAL_TAB_FEATURES
        path_ids = Config.CACHE_VAL_IDS
        path_lbl = Config.CACHE_VAL_LABELS
    elif split == "test":
        path_img = Config.CACHE_TEST_IMG_FEATURES
        path_tab = Config.CACHE_TEST_TAB_FEATURES
        path_ids = Config.CACHE_TEST_IDS
        path_lbl = None  # No labels for test
    else:
        raise ValueError(f"Unknown split: {split}")

    # Check cache
    if load_cached_data:
        files_exist = (
            os.path.exists(path_img)
            and os.path.exists(path_tab)
            and os.path.exists(path_ids)
        )

        if split != "test":
            files_exist = files_exist and os.path.exists(path_lbl)

        if files_exist:
            logger.info(
                f"Loading cached features for {split} from {Config.WORKING_DIR}..."
            )
            img_features = np.load(path_img)
            tab_features = np.load(path_tab)
            ids = np.load(path_ids)
            labels = np.load(path_lbl) if split != "test" else None
            return img_features, tab_features, ids, labels
        else:
            logger.info(
                f"Cache missing or incomplete for {split}. Starting extraction..."
            )
    else:
        logger.info(f"Ignoring cache. Starting extraction for {split}...")

    # Load Metadata
    df = load_metadata(split=split)

    # Initialize Extractor
    extractor = DualStreamExtractor()

    # Storage
    img_features_list = []
    tab_features_list = []
    ids_list = []
    labels_list = []

    # Identify tabular columns
    # margin_1..64, shape_1..64, texture_1..64
    margin_cols = [c for c in df.columns if c.startswith("margin")]
    shape_cols = [c for c in df.columns if c.startswith("shape")]
    texture_cols = [c for c in df.columns if c.startswith("texture")]
    tabular_cols = margin_cols + shape_cols + texture_cols

    logger.info(f"Processing {len(df)} images for {split} set...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
        # 1. Image Features
        # Returns (12, Dim)
        img_feats = extractor.process_image(row["file_path"])
        img_features_list.append(img_feats)

        # 2. Tabular Features
        # (192,)
        tab_feats = row[tabular_cols].values.astype(np.float32)
        tab_features_list.append(tab_feats)

        # 3. ID
        ids_list.append(row["id"])

        # 4. Label (if exists)
        if split != "test":
            labels_list.append(row["species"])

    # Convert to numpy arrays
    img_features = np.array(img_features_list, dtype=np.float32)  # (N, 12, Dim)
    tab_features = np.array(tab_features_list, dtype=np.float32)  # (N, 192)
    ids = np.array(ids_list, dtype=np.int64)

    if split != "test":
        labels = np.array(labels_list)
    else:
        labels = None

    # Save to cache
    logger.info(f"Saving extracted features to {Config.WORKING_DIR}...")
    np.save(path_img, img_features)
    np.save(path_tab, tab_features)
    np.save(path_ids, ids)

    if labels is not None:
        np.save(path_lbl, labels)

    logger.info("Extraction and caching complete.")

    return img_features, tab_features, ids, labels
