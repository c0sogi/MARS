import os
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything, load_image, rotate_image


class DualStreamExtractor:
    """
    Handles the loading of DINOv2 and ConvNeXt models and extracts concatenated features.
    """

    def __init__(self, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load DINOv2 (Global Geometry)
        # num_classes=0 returns the embedding (pooled features)
        self.dino = timm.create_model(
            Config.MODEL_DINO_NAME,
            pretrained=True,
            num_classes=0,
            img_size=Config.IMG_SIZE,
        )
        self.dino.to(self.device)
        self.dino.eval()

        # Load ConvNeXt (Local Texture)
        self.convnext = timm.create_model(
            Config.MODEL_CONVNEXT_NAME, pretrained=True, num_classes=0
        )
        self.convnext.to(self.device)
        self.convnext.eval()

        # Standard ImageNet Preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def extract_batch(self, images):
        """
        Extracts features from a batch of images using both models.
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
        Returns:
            np.ndarray: Concatenated features (B, Dim_DINO + Dim_ConvNeXt).
        """
        images = images.to(self.device)

        # Extract features
        feat_dino = self.dino(images)
        feat_conv = self.convnext(images)

        # Concatenate features (Early Fusion at extraction level)
        # Dimensions: DINO (~1024) + ConvNeXt (~1536) = ~2560
        features = torch.cat([feat_dino, feat_conv], dim=1)

        return features.cpu().numpy()


def extract_dataset_features(metadata_df, extractor, sample_limit=None):
    """
    Iterates over the dataset, generates 12 rotated views per image, and extracts features.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
        extractor (DualStreamExtractor): Initialized extractor instance.
        sample_limit (int, optional): Limit number of images for debugging.

    Returns:
        pd.DataFrame: DataFrame containing IDs, view indices, extracted features, and original metadata.
    """
    all_ids = []
    all_view_indices = []
    all_features = []

    # Apply sample limit if provided
    if sample_limit:
        metadata_df = metadata_df.iloc[:sample_limit]
        print(f"Debugging: Limiting extraction to first {sample_limit} images.")

    print(f"Starting feature extraction for {len(metadata_df)} images...")

    for idx, row in metadata_df.iterrows():
        img_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        try:
            # Load original image
            original_img = load_image(full_path)

            # Prepare batch of 12 rotations
            batch_tensors = []
            for view_idx, angle in enumerate(Config.ROTATION_ANGLES):
                # Rotate image
                rot_img = rotate_image(original_img, angle)
                # Preprocess to tensor
                tensor_img = extractor.transform(rot_img)
                batch_tensors.append(tensor_img)

            # Stack into a batch: (12, 3, 224, 224)
            batch_tensor = torch.stack(batch_tensors)

            # Extract features: (12, Feature_Dim)
            feats = extractor.extract_batch(batch_tensor)

            # Store results
            all_ids.extend([img_id] * Config.NUM_ROTATIONS)
            all_view_indices.extend(range(Config.NUM_ROTATIONS))
            all_features.append(feats)

        except Exception as e:
            print(f"Error processing image {img_id} at {full_path}: {e}")
            continue

    if not all_features:
        print("Warning: No features extracted.")
        return pd.DataFrame()

    # Concatenate all features into a large matrix: (N_images * 12, Feature_Dim)
    all_features = np.concatenate(all_features, axis=0)

    # Create feature column names
    # We name them f_0, f_1, ... to avoid creating 2500+ string objects manually in code
    feature_cols = [f"feat_{i}" for i in range(all_features.shape[1])]

    # Create DataFrame components
    df_meta_keys = pd.DataFrame({"id": all_ids, "view_idx": all_view_indices})

    df_features = pd.DataFrame(all_features, columns=feature_cols)

    # Combine keys and features
    df_result = pd.concat([df_meta_keys, df_features], axis=1)

    # Merge with original metadata to retain tabular features and labels
    # The original metadata has 1 row per ID. The result has 12 rows per ID.
    # Left merge ensures we duplicate the tabular features for each view.
    df_result = df_result.merge(metadata_df, on="id", how="left")

    return df_result


def get_or_compute_features(mode="train", load_cached_data=True, sample_limit=None):
    """
    Main entry point to get features. Handles caching logic.

    Args:
        mode (str): 'train' (includes val) or 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.
        sample_limit (int, optional): Limit dataset size for debugging.

    Returns:
        pd.DataFrame: The dataframe containing features for all views.
    """
    # 1. Determine paths and source data
    if mode == "train":
        cache_path = Config.CACHE_PATH_TRAIN_FEATURES
        # Combine Train and Val for full training set processing
        if os.path.exists(Config.TRAIN_METADATA_PATH) and os.path.exists(
            Config.VAL_METADATA_PATH
        ):
            df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
            df_val = pd.read_csv(Config.VAL_METADATA_PATH)
            metadata_df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        else:
            raise FileNotFoundError("Train/Val metadata files not found.")

    elif mode == "test":
        cache_path = Config.CACHE_PATH_TEST_FEATURES
        if os.path.exists(Config.TEST_METADATA_PATH):
            metadata_df = pd.read_csv(Config.TEST_METADATA_PATH)
        else:
            raise FileNotFoundError("Test metadata file not found.")

    else:
        raise ValueError("Mode must be 'train' or 'test'.")

    # Use global debug limit if not specified locally
    if sample_limit is None:
        sample_limit = Config.DEBUG_SAMPLE_LIMIT

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} features from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from scratch...")

    # 3. Compute Features
    print(f"Computing features for {mode} set (Sample Limit: {sample_limit})...")

    # Ensure reproducibility
    seed_everything(Config.RANDOM_SEED)

    # Initialize extractor
    extractor = DualStreamExtractor()

    # Run extraction
    df_result = extract_dataset_features(metadata_df, extractor, sample_limit)

    # 4. Save to Cache
    if not df_result.empty:
        print(f"Saving {len(df_result)} rows to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_result.to_parquet(cache_path, index=False)

    return df_result
