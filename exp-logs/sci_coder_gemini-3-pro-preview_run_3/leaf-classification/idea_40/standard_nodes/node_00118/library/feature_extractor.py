import os
import torch
import timm
import pandas as pd
import numpy as np
from torchvision import transforms
from library.config import Config
from library.image_utils import load_image, generate_rotated_views


class FeatureExtractor:
    """
    Handles the extraction of deep learning features using DINOv2 and ConvNeXt models.
    Implements caching and batch processing for efficiency.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FeatureExtractor initialized on device: {self.device}")

        # Initialize DINOv2 (Global Geometry Stream)
        print(f"Loading DINOv2 model: {Config.MODEL_DINO_NAME}...")
        self.dino_model = timm.create_model(
            Config.MODEL_DINO_NAME,
            pretrained=True,
            num_classes=0,  # Get features, not logits
            img_size=Config.IMAGE_SIZE,
        )
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Initialize ConvNeXt (Local Texture Stream)
        print(f"Loading ConvNeXt model: {Config.MODEL_CONVNEXT_NAME}...")
        self.convnext_model = timm.create_model(
            Config.MODEL_CONVNEXT_NAME,
            pretrained=True,
            num_classes=0,  # Get features, not logits
        )
        self.convnext_model.to(self.device)
        self.convnext_model.eval()

        # Define standard ImageNet normalization
        # Note: image_utils.load_image already handles resizing to Config.IMAGE_SIZE
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _process_batch(self, image_paths, image_ids):
        """
        Internal method to process a batch of images.
        Generates rotated views and runs inference.
        """
        batch_tensors = []
        batch_meta = []  # Stores (id, angle) tuples corresponding to tensors

        # 1. Prepare Batch with Rotations
        for img_path, img_id in zip(image_paths, image_ids):
            # Load base image
            full_path = os.path.join(Config.INPUT_DIR, img_path)
            img_np = load_image(full_path)

            if img_np is None:
                # Skip missing images (though metadata check says none are missing)
                continue

            # Generate 12 rotated views
            rotated_views = generate_rotated_views(img_np, Config.ROTATION_ANGLES)

            for angle, view in zip(Config.ROTATION_ANGLES, rotated_views):
                # Convert to tensor and normalize
                # view is H,W,C (RGB) -> ToTensor makes it C,H,W [0,1]
                tensor = self.transform(view)
                batch_tensors.append(tensor)
                batch_meta.append((img_id, angle))

        if not batch_tensors:
            return [], [], [], []

        # Stack into a single tensor for the batch
        # Shape: (Batch_Size * 12, 3, H, W)
        input_tensor = torch.stack(batch_tensors).to(self.device)

        # 2. Inference
        with torch.no_grad():
            # Extract DINO features
            dino_feats = self.dino_model(input_tensor).cpu().numpy()

            # Extract ConvNeXt features
            convnext_feats = self.convnext_model(input_tensor).cpu().numpy()

        # 3. Unpack results
        ids = [m[0] for m in batch_meta]
        angles = [m[1] for m in batch_meta]

        return ids, angles, dino_feats, convnext_feats

    def extract_dataset_features(
        self, metadata_path, cache_path, load_cached_data=True, limit=None
    ):
        """
        Extracts features for the dataset specified in the metadata file.

        Args:
            metadata_path (str): Path to the CSV containing 'id' and 'file_path'.
            cache_path (str): Path to save/load the Parquet cache.
            load_cached_data (bool): If True, attempts to load from cache first.
            limit (int, optional): Limit the number of images processed (for debugging).

        Returns:
            pd.DataFrame: DataFrame containing IDs, angles, and feature vectors.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading features from cache: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Load Metadata
        print(f"Reading metadata from {metadata_path}...")
        df = pd.read_csv(metadata_path)

        if limit:
            df = df.head(limit)
            print(f"Limiting processing to first {limit} images.")

        # 3. Processing Loop
        all_ids = []
        all_angles = []
        all_dino = []
        all_conv = []

        # Process in chunks of Config.BATCH_SIZE
        # Note: Effective batch size for model is BATCH_SIZE * len(ROTATION_ANGLES)
        # e.g., 32 * 12 = 384 images per forward pass.

        paths = df["file_path"].tolist()
        ids = df["id"].tolist()
        num_samples = len(paths)

        print(f"Starting feature extraction for {num_samples} images...")

        for i in range(0, num_samples, Config.BATCH_SIZE):
            batch_paths = paths[i : i + Config.BATCH_SIZE]
            batch_ids = ids[i : i + Config.BATCH_SIZE]

            b_ids, b_angles, b_dino, b_conv = self._process_batch(
                batch_paths, batch_ids
            )

            all_ids.extend(b_ids)
            all_angles.extend(b_angles)
            # Store features as lists in the dataframe cells to be Parquet compatible
            all_dino.extend(list(b_dino))
            all_conv.extend(list(b_conv))

            if (i + 1) % (Config.BATCH_SIZE * 5) == 0:
                print(f"Processed {i} / {num_samples} images...")

        # 4. Construct DataFrame
        print("Constructing DataFrame...")
        # Convert numpy arrays to lists for storage if necessary,
        # but pandas/parquet handles numpy arrays in columns well usually.
        # However, to ensure compatibility, we'll keep them as arrays or lists.
        # Here we treat each feature vector as a single object (array).

        out_df = pd.DataFrame(
            {
                "id": all_ids,
                "view_angle": all_angles,
                "dino_features": [x for x in all_dino],
                "convnext_features": [x for x in all_conv],
            }
        )

        # 5. Save Cache
        print(f"Saving features to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out_df.to_parquet(cache_path, index=False)

        print("Feature extraction complete.")
        return out_df
