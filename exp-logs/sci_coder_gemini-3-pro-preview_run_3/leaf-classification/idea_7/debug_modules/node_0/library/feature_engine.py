import os
import numpy as np
import torch
import timm
from library.config import Config
from library.image_ops import ImagePreprocessor
from library.data_loader import LeafDataManager
from library.utils import seed_everything


class DualStreamExtractor:
    """
    Extracts features using DINOv2 (Global) and ConvNeXt (Local) backbones.
    Handles multi-view averaging and caching.
    """

    def __init__(self):
        # Ensure reproducibility
        seed_everything(Config.SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DualStreamExtractor using device: {self.device}")

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Initialize Preprocessor
        self.preprocessor = ImagePreprocessor()

        # Load Models
        # Stream A: DINOv2 for global geometry
        print(f"Loading DINOv2 model: {Config.MODEL_DINO}")
        self.dino_model = timm.create_model(
            Config.MODEL_DINO, pretrained=True, num_classes=0  # Get feature embeddings
        ).to(self.device)
        self.dino_model.eval()

        # Stream B: ConvNeXt for local texture/margins
        print(f"Loading ConvNeXt model: {Config.MODEL_CONVNEXT}")
        self.conv_model = timm.create_model(
            Config.MODEL_CONVNEXT,
            pretrained=True,
            num_classes=0,  # Get feature embeddings
        ).to(self.device)
        self.conv_model.eval()

    def get_train_features(self, load_cached_data=True):
        """
        Retrieves training features. Loads from cache if available and requested,
        otherwise computes them from scratch and caches them.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (dino_features, conv_features)
                dino_features (np.ndarray): (N, D1)
                conv_features (np.ndarray): (N, D2)
        """
        # Check cache existence
        cache_exists = os.path.exists(Config.CACHE_TRAIN_DINO) and os.path.exists(
            Config.CACHE_TRAIN_CONV
        )

        if load_cached_data and cache_exists:
            print("Loading training features from cache...")
            dino_features = np.load(Config.CACHE_TRAIN_DINO)
            conv_features = np.load(Config.CACHE_TRAIN_CONV)
            return dino_features, conv_features

        print("Extracting training features from images...")
        # Load file paths
        dm = LeafDataManager()
        # We use load_cached_data=True for the tabular part to avoid re-parsing CSVs
        _, _, file_paths, _ = dm.load_train_data(load_cached_data=True)

        dino_features, conv_features = self._extract_dataset(file_paths)

        # Save to cache
        print("Caching training features...")
        np.save(Config.CACHE_TRAIN_DINO, dino_features)
        np.save(Config.CACHE_TRAIN_CONV, conv_features)

        return dino_features, conv_features

    def get_test_features(self, load_cached_data=True):
        """
        Retrieves test features. Loads from cache if available and requested,
        otherwise computes them from scratch and caches them.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (dino_features, conv_features)
        """
        cache_exists = os.path.exists(Config.CACHE_TEST_DINO) and os.path.exists(
            Config.CACHE_TEST_CONV
        )

        if load_cached_data and cache_exists:
            print("Loading test features from cache...")
            dino_features = np.load(Config.CACHE_TEST_DINO)
            conv_features = np.load(Config.CACHE_TEST_CONV)
            return dino_features, conv_features

        print("Extracting test features from images...")
        dm = LeafDataManager()
        _, _, file_paths = dm.load_test_data(load_cached_data=True)

        dino_features, conv_features = self._extract_dataset(file_paths)

        # Save to cache
        print("Caching test features...")
        np.save(Config.CACHE_TEST_DINO, dino_features)
        np.save(Config.CACHE_TEST_CONV, conv_features)

        return dino_features, conv_features

    def _extract_dataset(self, file_paths):
        """
        Internal method to iterate over file paths, batch process images,
        and extract features using both models.
        """
        all_dino = []
        all_conv = []

        total_files = len(file_paths)
        batch_size = Config.BATCH_SIZE

        # Process in batches
        for i in range(0, total_files, batch_size):
            batch_paths = file_paths[i : i + batch_size]
            current_batch_size = len(batch_paths)

            # Print progress periodically
            if i % (batch_size * 5) == 0:
                print(f"Processing images {i}/{total_files}...")

            batch_dino_tensors = []
            batch_conv_tensors = []

            # Load and preprocess
            for p in batch_paths:
                # Returns (4, 3, 518, 518) and (4, 3, 1024, 1024)
                d_t, c_t = self.preprocessor.load_and_preprocess(p)
                batch_dino_tensors.append(d_t)
                batch_conv_tensors.append(c_t)

            # Stack into single tensors: (B*4, 3, H, W)
            # This combines the batch dimension and the view dimension
            dino_input = torch.cat(batch_dino_tensors, dim=0).to(self.device)
            conv_input = torch.cat(batch_conv_tensors, dim=0).to(self.device)

            with torch.no_grad():
                # DINO Inference
                # Shape: (B*4, EmbedDim)
                dino_out = self.dino_model(dino_input)

                # ConvNeXt Inference
                # Shape: (B*4, EmbedDim)
                conv_out = self.conv_model(conv_input)

            # Reshape to (B, 4, EmbedDim) and average over views (dim 1)
            # DINO
            d_dim = dino_out.shape[1]
            dino_feats = dino_out.view(current_batch_size, 4, d_dim).mean(dim=1)

            # ConvNeXt
            c_dim = conv_out.shape[1]
            conv_feats = conv_out.view(current_batch_size, 4, c_dim).mean(dim=1)

            # Move to CPU and store
            all_dino.append(dino_feats.cpu().numpy())
            all_conv.append(conv_feats.cpu().numpy())

            # Clean up GPU memory
            del dino_input, conv_input, dino_out, conv_out
            torch.cuda.empty_cache()

        # Concatenate all batches
        final_dino = np.concatenate(all_dino, axis=0)
        final_conv = np.concatenate(all_conv, axis=0)

        print(
            f"Extraction complete. DINO shape: {final_dino.shape}, ConvNeXt shape: {final_conv.shape}"
        )

        return final_dino, final_conv
