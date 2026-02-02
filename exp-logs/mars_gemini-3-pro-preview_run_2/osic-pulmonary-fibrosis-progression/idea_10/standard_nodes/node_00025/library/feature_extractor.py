import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from tqdm import tqdm

from library.config import Config
from library.dicom_loader import load_scan
from library.image_processing import (
    get_patient_zones,
    select_variance_slice,
    compute_density_hist,
    preprocess_image,
)
from library.utils import seed_everything


class VisualBackbone(nn.Module):
    """
    A wrapper around timm's EfficientNet-B0 to act as a fixed feature extractor.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        # Load pre-trained EfficientNet-B0
        # num_classes=0 removes the classifier head and pooling, returning the features
        # global_pool='' returns the spatial features, but usually we want the pooled vector for simple extraction.
        # By default timm with num_classes=0 returns the pooled feature vector (Global Average Pooling).
        self.model = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )
        self.model.eval()  # Set to evaluation mode

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        # output shape: (Batch, 1280) for EfficientNet-B0
        return self.model(x)


class FeatureExtractor:
    """
    Orchestrates the extraction of zonal visual features and density histograms.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone = VisualBackbone().to(self.device)
        self.backbone.eval()

        # Seed for reproducibility
        seed_everything(Config.SEED)

    def extract_patient_features(self, patient_id, dcm_rel_path):
        """
        Extracts the hybrid feature vector for a single patient.

        Vector composition:
        [Zone1_Embed (1280), Zone2_Embed (1280), Zone3_Embed (1280),
         Zone1_Hist (4), Zone2_Hist (4), Zone3_Hist (4)]

        Total dimension: 1280*3 + 4*3 = 3852
        """
        full_path = os.path.join(Config.INPUT_DIR, dcm_rel_path)

        # 1. Load 3D Volume
        # load_scan handles caching internally for the raw volume
        volume = load_scan(full_path, load_cached_data=True)

        # 2. Split into Zonal Sub-volumes
        zones = get_patient_zones(volume)

        zone_embeddings = []
        zone_histograms = []

        # Prepare batch of images for the backbone
        images_to_process = []

        # 3. Process each zone
        for zone_vol in zones:
            # A. Structural Feature: Density Histogram
            hist = compute_density_hist(zone_vol)
            zone_histograms.append(hist)

            # B. Visual Feature Preparation: Max Variance Slice
            best_slice = select_variance_slice(zone_vol)
            processed_img = preprocess_image(best_slice)  # (3, 224, 224)
            images_to_process.append(processed_img)

        # 4. Run Deep Learning Backbone
        # Stack into a single batch: (3, 3, 224, 224)
        batch_tensor = torch.tensor(
            np.stack(images_to_process), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            # (3, 1280)
            features = self.backbone(batch_tensor)
            features_np = features.cpu().numpy()

        # Flatten embeddings: [Zone1, Zone2, Zone3]
        flat_embeddings = features_np.flatten()  # (3840,)

        # Flatten histograms: [Zone1_h, Zone2_h, Zone3_h]
        flat_histograms = np.concatenate(zone_histograms)  # (12,)

        # 5. Concatenate
        final_vector = np.concatenate([flat_embeddings, flat_histograms])

        return final_vector

    def generate_features(self, metadata_df, save_name, load_cached_data=True):
        """
        Generates features for all patients in the metadata dataframe.
        Implements caching to disk.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'Patient' and 'dcm_path'.
            save_name (str): Filename for the cache (e.g., 'train_features').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (features_matrix, patient_ids)
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        feat_cache_path = os.path.join(Config.WORKING_DIR, f"{save_name}_features.npy")
        ids_cache_path = os.path.join(Config.WORKING_DIR, f"{save_name}_ids.npy")

        # We only need unique patients, as features are static per patient
        # (The metadata might have multiple rows per patient for different weeks)
        unique_patients_df = metadata_df[["Patient", "dcm_path"]].drop_duplicates()

        # 1. Try Load from Cache
        if (
            load_cached_data
            and os.path.exists(feat_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            print(f"Loading cached features from {feat_cache_path}...")
            try:
                features = np.load(feat_cache_path)
                ids = np.load(ids_cache_path)

                # Cite debug_lesson_3: Invalidate Stale Caches When Toggling Debug Modes
                if len(ids) == len(unique_patients_df):
                    return features, ids
                else:
                    print(
                        f"Stale feature cache detected (Patients: {len(ids)} vs {len(unique_patients_df)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        print(f"Generating features for {len(metadata_df)} patients...")

        feature_list = []
        patient_id_list = []

        # Iterate with index to handle potential failures gracefully if needed
        # Using tqdm is not allowed per prompt "Only print the required information. Do not print progress bars"
        # So we iterate silently or with simple print
        total = len(unique_patients_df)

        for idx, row in unique_patients_df.iterrows():
            pid = row["Patient"]
            path = row["dcm_path"]

            try:
                feat_vec = self.extract_patient_features(pid, path)
                feature_list.append(feat_vec)
                patient_id_list.append(pid)
            except Exception as e:
                print(f"Error extracting features for {pid}: {e}")
                # Append zero vector to maintain consistency?
                # Better to skip and let downstream handle alignment, or fill zeros.
                # Given constraints, filling with zeros is safer to avoid shape mismatch later.
                # Expected dim: 1280*3 + 12 = 3852
                feature_list.append(np.zeros(3852, dtype=np.float32))
                patient_id_list.append(pid)

            if (len(feature_list)) % 50 == 0:
                print(f"Processed {len(feature_list)}/{total} patients.")

        features_matrix = np.array(feature_list, dtype=np.float32)
        ids_array = np.array(patient_id_list)

        # 3. Save to Cache
        print(f"Saving features to {feat_cache_path}...")
        np.save(feat_cache_path, features_matrix)
        np.save(ids_cache_path, ids_array)

        return features_matrix, ids_array
