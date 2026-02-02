import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
import timm
from library.config import Config
from library.utils import seed_everything, save_npy, load_npy


class FeatureExtractor:
    """
    Handles the extraction of deep learning features from images using
    DINOv2 and ConvNeXt models with multi-view rotation augmentation.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.image_size = Config.IMAGE_SIZE
        self.batch_size = Config.BATCH_SIZE
        self.num_rotations = Config.NUM_ROTATIONS
        self.rotation_step = Config.ROTATION_STEP

        # Image preprocessing pipeline
        self.preprocess = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Models are loaded lazily
        self.dino = None
        self.convnext = None

    def _load_models(self):
        """Loads models to GPU if not already loaded."""
        if self.dino is not None and self.convnext is not None:
            return

        print(f"Loading models on {self.device}...")

        # Load DINOv2 (ViT-Large)
        self.dino = timm.create_model(
            Config.MODEL_DINO,
            pretrained=True,
            num_classes=0,  # Remove classification head
        ).to(self.device)
        self.dino.eval()

        # Load ConvNeXt Large
        self.convnext = timm.create_model(
            Config.MODEL_CONVNEXT,
            pretrained=True,
            num_classes=0,  # Remove classification head
        ).to(self.device)
        self.convnext.eval()

    def _get_rotations(self, img_tensor):
        """
        Generates N equidistant rotated views of the input image tensor.
        Args:
            img_tensor: Tensor of shape (C, H, W)
        Returns:
            Tensor of shape (N_rotations, C, H, W)
        """
        rotations = []
        for i in range(self.num_rotations):
            angle = i * self.rotation_step
            # rotate expects angle in degrees
            img_rot = T.functional.rotate(img_tensor, angle)
            rotations.append(img_rot)
        return torch.stack(rotations)

    def extract_features(self, df):
        """
        Extracts features for all images in the dataframe.
        Args:
            df: DataFrame containing 'file_path', 'id', and tabular features.
        Returns:
            img_features: (N, 12, D_total)
            tab_features: (N, 192)
            ids: (N,)
            labels: (N,) or None
        """
        self._load_models()

        img_features_list = []
        tab_features_list = []
        ids_list = []
        labels_list = []

        # Identify tabular columns
        margin_cols = [c for c in df.columns if c.startswith("margin")]
        shape_cols = [c for c in df.columns if c.startswith("shape")]
        texture_cols = [c for c in df.columns if c.startswith("texture")]
        tab_cols = margin_cols + shape_cols + texture_cols

        num_samples = len(df)
        print(f"Extracting features for {num_samples} samples...")

        # Process in batches
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_df = df.iloc[start_idx:end_idx]

            batch_imgs = []
            valid_mask = []

            for _, row in batch_df.iterrows():
                img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
                try:
                    # Open and convert to RGB
                    img = Image.open(img_path).convert("RGB")
                    img_t = self.preprocess(img)

                    # Generate rotations
                    rotations = self._get_rotations(img_t)  # (12, 3, 224, 224)
                    batch_imgs.append(rotations)
                    valid_mask.append(True)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    valid_mask.append(False)

            if not batch_imgs:
                continue

            # Create batch tensor: (B, 12, 3, 224, 224)
            batch_tensor = torch.stack(batch_imgs)
            B, R, C, H, W = batch_tensor.shape

            # Flatten to (B*12, 3, 224, 224) for efficient batch inference
            batch_flat = batch_tensor.view(B * R, C, H, W).to(self.device)

            with torch.no_grad():
                # Extract DINO features
                feat_dino = self.dino(batch_flat)  # (B*R, 1024)

                # Extract ConvNeXt features
                feat_conv = self.convnext(batch_flat)  # (B*R, 1536)

                # Concatenate features
                feat_combined = torch.cat([feat_dino, feat_conv], dim=1)  # (B*R, 2560)

                # Reshape back to (B, 12, 2560)
                feat_reshaped = feat_combined.view(B, R, -1).cpu().numpy()

            img_features_list.append(feat_reshaped)

            # Collect tabular features and IDs for valid images
            batch_valid = batch_df[valid_mask]
            tab_features_list.append(batch_valid[tab_cols].values.astype(np.float32))
            ids_list.append(batch_valid["id"].values)

            if "species" in batch_valid.columns:
                labels_list.append(batch_valid["species"].values)

        # Concatenate all batches
        img_features = np.concatenate(img_features_list, axis=0)
        tab_features = np.concatenate(tab_features_list, axis=0)
        ids = np.concatenate(ids_list, axis=0)

        labels = None
        if labels_list:
            labels = np.concatenate(labels_list, axis=0)

        return img_features, tab_features, ids, labels

    def run(self, load_cached_data=True):
        """
        Main execution method. Checks cache, and if missing, runs extraction pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load from disk first.
        """
        seed_everything()
        Config.setup()  # Ensure directories exist

        # Define keys for cache retrieval
        train_keys = [
            "train_img_features",
            "train_tab_features",
            "train_ids",
            "train_labels",
        ]
        test_keys = ["test_img_features", "test_tab_features", "test_ids"]

        # Check if all cache files exist
        cache_complete = True
        if load_cached_data:
            for key in train_keys + test_keys:
                if not os.path.exists(Config.get_cache_path(key)):
                    cache_complete = False
                    break
        else:
            cache_complete = False

        if cache_complete:
            print("All features found in cache. Skipping extraction.")
            return

        print("Starting feature extraction pipeline...")

        # 1. Process Training Data (Train + Val)
        # We combine them to maximize data for the Cross-Validation strategy
        print("Loading training metadata...")
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_full_train = pd.concat([df_train, df_val], ignore_index=True)

        train_img, train_tab, train_ids, train_labels = self.extract_features(
            df_full_train
        )

        print("Saving training features...")
        save_npy(train_img, Config.get_cache_path("train_img_features"))
        save_npy(train_tab, Config.get_cache_path("train_tab_features"))
        save_npy(train_ids, Config.get_cache_path("train_ids"))
        save_npy(train_labels, Config.get_cache_path("train_labels"))

        # 2. Process Test Data
        print("Loading test metadata...")
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        test_img, test_tab, test_ids, _ = self.extract_features(df_test)

        print("Saving test features...")
        save_npy(test_img, Config.get_cache_path("test_img_features"))
        save_npy(test_tab, Config.get_cache_path("test_tab_features"))
        save_npy(test_ids, Config.get_cache_path("test_ids"))

        print("Feature extraction pipeline finished.")
