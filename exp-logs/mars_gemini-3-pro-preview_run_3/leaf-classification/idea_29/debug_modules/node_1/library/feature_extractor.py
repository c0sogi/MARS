import os
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms.functional as TF

from library.config import Config
from library.utils import setup_logger, save_numpy, load_numpy


class DeepFeatureExtractor:
    """
    Handles loading images, generating multi-view rotations, and extracting
    deep features using DINOv2 and ConvNeXt models.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("FeatureExtractor")

        # Models and transforms are initialized lazily
        self.dino_model = None
        self.conv_model = None
        self.dino_transform = None
        self.conv_transform = None

    def _init_models(self):
        """
        Initializes the deep learning models and their specific data transforms.
        """
        self.logger.info(f"Initializing models on device: {self.device}")

        # 1. DINOv2 (Global Geometry Stream)
        self.logger.info(f"Loading DINOv2 model: {Config.MODEL_DINO}")
        self.dino_model = timm.create_model(
            Config.MODEL_DINO,
            pretrained=True,
            num_classes=0,  # Use pooling/token for feature extraction
        ).to(self.device)
        self.dino_model.eval()

        dino_config = resolve_data_config({}, model=self.dino_model)
        self.dino_transform = create_transform(**dino_config)

        # 2. ConvNeXt (Local Texture Stream)
        self.logger.info(f"Loading ConvNeXt model: {Config.MODEL_CONV}")
        self.conv_model = timm.create_model(
            Config.MODEL_CONV, pretrained=True, num_classes=0
        ).to(self.device)
        self.conv_model.eval()

        conv_config = resolve_data_config({}, model=self.conv_model)
        self.conv_transform = create_transform(**conv_config)

    def _process_image(self, rel_path):
        """
        Loads an image from disk and converts it to RGB.
        """
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Open image and convert to RGB (models expect 3 channels)
        img = Image.open(full_path).convert("RGB")
        return img

    def _generate_rotated_views(self, img):
        """
        Generates N_ROTATIONS equidistant rotated views of the image.
        Fills the background with white (255, 255, 255).
        """
        views = []
        # Calculate angles: 0, 30, 60, ..., 330 for N=12
        angles = np.linspace(0, 360, Config.N_ROTATIONS, endpoint=False)

        for angle in angles:
            # Rotate image. fill=[255, 255, 255] ensures white background for new areas
            rot_img = TF.rotate(img, angle, fill=[255, 255, 255])
            views.append(rot_img)

        return views

    def extract_features(self, metadata_df, dataset_name, load_cached_data=True):
        """
        Main pipeline to extract features for a given dataset (train/val/test).

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
            dataset_name (str): Name of the dataset (e.g., 'train', 'test') for cache naming.
            load_cached_data (bool): If True, attempts to load from disk first.

        Returns:
            ids (np.ndarray): Shape (N,)
            dino_features (np.ndarray): Shape (N, N_ROTATIONS, D_dino)
            conv_features (np.ndarray): Shape (N, N_ROTATIONS, D_conv)
        """
        # Define cache filenames
        cache_ids_name = f"{dataset_name}_ids"
        cache_dino_name = f"{dataset_name}_dino_features"
        cache_conv_name = f"{dataset_name}_conv_features"

        # 1. Try loading from cache
        if load_cached_data:
            ids = load_numpy(cache_ids_name)
            dino_feats = load_numpy(cache_dino_name)
            conv_feats = load_numpy(cache_conv_name)

            if ids is not None and dino_feats is not None and conv_feats is not None:
                self.logger.info(
                    f"Successfully loaded cached features for '{dataset_name}'."
                )
                return ids, dino_feats, conv_feats

        # 2. Compute from scratch
        self.logger.info(
            f"Starting feature extraction for '{dataset_name}' ({len(metadata_df)} samples)..."
        )

        # Initialize models if needed
        if self.dino_model is None:
            self._init_models()

        ids_list = []
        dino_features_list = []
        conv_features_list = []

        paths = metadata_df["file_path"].values
        image_ids = metadata_df["id"].values
        num_samples = len(paths)
        batch_size = Config.BATCH_SIZE

        # Process in batches
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_paths = paths[i : i + batch_size]
                batch_ids = image_ids[i : i + batch_size]

                # Lists to hold transformed tensors for the batch
                batch_dino_inputs = []
                batch_conv_inputs = []

                for p in batch_paths:
                    img = self._process_image(p)
                    views = self._generate_rotated_views(img)

                    # Apply specific transforms for each model
                    # Each image generates N_ROTATIONS views
                    dino_views = [self.dino_transform(v) for v in views]
                    conv_views = [self.conv_transform(v) for v in views]

                    batch_dino_inputs.extend(dino_views)
                    batch_conv_inputs.extend(conv_views)

                # Serialized Inference with AMP to save memory
                current_batch_len = len(batch_paths)

                # 1. Process DINOv2
                dino_tensor = torch.stack(batch_dino_inputs).to(self.device)
                with torch.amp.autocast("cuda"):
                    dino_out = self.dino_model(dino_tensor)

                dino_reshaped = (
                    dino_out.view(current_batch_len, Config.N_ROTATIONS, -1)
                    .float()
                    .cpu()
                    .numpy()
                )

                # Free memory immediately
                del dino_tensor, dino_out
                torch.cuda.empty_cache()

                # 2. Process ConvNeXt
                conv_tensor = torch.stack(batch_conv_inputs).to(self.device)
                with torch.amp.autocast("cuda"):
                    conv_out = self.conv_model(conv_tensor)

                conv_reshaped = (
                    conv_out.view(current_batch_len, Config.N_ROTATIONS, -1)
                    .float()
                    .cpu()
                    .numpy()
                )

                # Free memory
                del conv_tensor, conv_out
                torch.cuda.empty_cache()

                # Store
                dino_features_list.append(dino_reshaped)
                conv_features_list.append(conv_reshaped)
                ids_list.append(batch_ids)

                # Log progress periodically
                if (i // batch_size) % 5 == 0:
                    self.logger.info(
                        f"Processed {min(i + batch_size, num_samples)}/{num_samples} images"
                    )

        # Concatenate all batches
        all_ids = np.concatenate(ids_list, axis=0)
        all_dino = np.concatenate(dino_features_list, axis=0)
        all_conv = np.concatenate(conv_features_list, axis=0)

        self.logger.info(f"Extraction complete for '{dataset_name}'.")
        self.logger.info(f"DINOv2 Shape: {all_dino.shape}")
        self.logger.info(f"ConvNeXt Shape: {all_conv.shape}")

        # 3. Save to cache
        self.logger.info("Saving features to cache...")
        save_numpy(all_ids, cache_ids_name)
        save_numpy(all_dino, cache_dino_name)
        save_numpy(all_conv, cache_conv_name)

        return all_ids, all_dino, all_conv
