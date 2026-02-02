import os
import numpy as np
import pandas as pd
import torch
import timm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from library.config import Config, seed_everything
from library.dicom_processing import process_patient

# Ensure reproducibility
seed_everything(Config.SEED)


class ImageEmbedder:
    """
    Wraps a pre-trained EfficientNet-B0 to extract features from lung CT slices.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        # Load EfficientNet-B0, num_classes=0 returns the global pool features (1280 dim)
        self.model = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )
        self.model.to(self.device)
        self.model.eval()

        # ImageNet Normalization Constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def extract(self, images: np.ndarray) -> np.ndarray:
        """
        Extracts embeddings for a batch of images.
        Args:
            images: Numpy array of shape (N, H, W, 3) in [0, 1] range.
        Returns:
            Numpy array of shape (N, 1280).
        """
        if images.shape[0] == 0:
            return np.zeros((0, 1280), dtype=np.float32)

        # Convert to Tensor: (N, H, W, 3) -> (N, 3, H, W)
        img_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(self.device)

        # Normalize
        img_tensor = (img_tensor - self.mean) / self.std

        with torch.no_grad():
            # Forward pass
            features = self.model(img_tensor)

        return features.cpu().numpy()


class TabularProcessor:
    """
    Handles encoding of clinical metadata.
    """

    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.cat_cols = ["Sex", "SmokingStatus"]

    def fit(self, df):
        self.encoder.fit(df[self.cat_cols])
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("TabularProcessor must be fitted before transform.")

        # Encode categoricals
        cat_feats = self.encoder.transform(df[self.cat_cols])

        # Get numericals (Age)
        # We treat Age as a raw feature here; scaling happens globally later
        age_feats = df[["Age"]].values.astype(np.float32)

        return np.hstack([age_feats, cat_feats])


class PCAReducer:
    """
    Manages scaling and dimensionality reduction without using pickle for storage.
    """

    def __init__(self, n_components=None):
        self.n_components = (
            n_components if n_components is not None else Config.N_PCA_COMPONENTS
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=Config.SEED)
        self.is_fitted = False

    def fit(self, X):
        print(f"Fitting PCA on data with shape {X.shape}...")

        # Cite debug_lesson_7: Adapt Static Hyperparameters to Runtime Data Dimensions
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
        if self.n_components > max_components:
            print(
                f"Adjusting n_components from {self.n_components} to {max_components} to match data dimensions."
            )
            self.n_components = max_components
            self.pca.n_components = max_components

        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.is_fitted = True

        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA Explained Variance (Top {self.n_components}): {explained_var:.6f}")

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted before transform.")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def save(self, dir_path):
        """Saves PCA and Scaler parameters to npy files."""
        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path, "scaler_mean.npy"), self.scaler.mean_)
        np.save(os.path.join(dir_path, "scaler_scale.npy"), self.scaler.scale_)
        np.save(os.path.join(dir_path, "pca_components.npy"), self.pca.components_)
        np.save(os.path.join(dir_path, "pca_mean.npy"), self.pca.mean_)
        np.save(os.path.join(dir_path, "pca_var.npy"), self.pca.explained_variance_)

    def load(self, dir_path):
        """Reconstructs PCA and Scaler from npy files."""
        self.scaler.mean_ = np.load(os.path.join(dir_path, "scaler_mean.npy"))
        self.scaler.scale_ = np.load(os.path.join(dir_path, "scaler_scale.npy"))
        self.pca.components_ = np.load(os.path.join(dir_path, "pca_components.npy"))
        self.pca.mean_ = np.load(os.path.join(dir_path, "pca_mean.npy"))
        self.pca.explained_variance_ = np.load(os.path.join(dir_path, "pca_var.npy"))
        self.is_fitted = True


class FeatureAggregator:
    """
    Orchestrates the generation of the full feature set.
    """

    def __init__(self):
        self.embedder = ImageEmbedder()
        self.tabular_processor = TabularProcessor()
        self.pca_reducer = PCAReducer()

    def _get_patient_features(self, patient_id, dcm_path, load_cached_data):
        """
        Generates the static feature vector for a single patient:
        [Flattened_Visual_Embeddings, Density_Profile]
        """
        # 1. Get processed images and density
        data = process_patient(patient_id, dcm_path, load_cached_data=load_cached_data)
        images = data["images"]  # (N_views, 224, 224, 3)
        density = data["density"]  # (4,)

        # 2. Extract Visual Embeddings
        # Shape: (N_views, 1280)
        embeddings = self.embedder.extract(images)

        # Flatten embeddings: we expect fixed number of views (Config.TOTAL_SLICES)
        # If fewer views were returned (e.g. read error), pad with zeros
        expected_views = Config.TOTAL_SLICES
        current_views = embeddings.shape[0]

        if current_views < expected_views:
            padding = np.zeros((expected_views - current_views, 1280), dtype=np.float32)
            embeddings = np.vstack([embeddings, padding])
        elif current_views > expected_views:
            embeddings = embeddings[:expected_views]

        flat_embeddings = embeddings.flatten()  # (N_views * 1280,)

        # 3. Concatenate
        return np.concatenate([flat_embeddings, density])

    def generate_raw_features(self, df, dataset_name, load_cached_data=True):
        """
        Generates high-dimensional features for a dataset.
        Caches the result to disk.
        """
        cache_file = os.path.join(
            Config.WORKING_DIR, f"raw_features_{dataset_name}.npy"
        )
        patient_ids_file = os.path.join(
            Config.WORKING_DIR, f"patient_ids_{dataset_name}.npy"
        )

        # Check cache
        if (
            load_cached_data
            and os.path.exists(cache_file)
            and os.path.exists(patient_ids_file)
        ):
            print(f"Loading cached raw features for {dataset_name}...")
            X_raw = np.load(cache_file)
            cached_pids = np.load(patient_ids_file, allow_pickle=True)

            # Verify alignment (simple check)
            unique_patients = df["Patient"].unique()
            if len(cached_pids) == len(unique_patients):
                # We assume the order is correct if the count matches and we rely on the map logic below
                # To be safe, we return a dictionary mapping PatientID -> FeatureVector
                return dict(zip(cached_pids, X_raw))

        print(f"Generating raw features for {dataset_name}...")
        unique_patients = df["Patient"].unique()

        # Debug subset
        if Config.DEBUG:
            unique_patients = unique_patients[: Config.DEBUG_SAMPLE_SIZE]
            print(f"DEBUG: Processing subset of {len(unique_patients)} patients.")

        features_list = []
        processed_pids = []

        for pid in unique_patients:
            # Get path from the first occurrence of this patient
            patient_row = df[df["Patient"] == pid].iloc[0]
            dcm_path = patient_row["dcm_path"]

            feat_vec = self._get_patient_features(pid, dcm_path, load_cached_data)
            features_list.append(feat_vec)
            processed_pids.append(pid)

        X_raw = np.array(features_list, dtype=np.float32)
        processed_pids = np.array(processed_pids)

        # Save cache
        np.save(cache_file, X_raw)
        np.save(patient_ids_file, processed_pids)

        return dict(zip(processed_pids, X_raw))

    def prepare_datasets(self, load_cached_data=True):
        """
        Main pipeline execution method.
        1. Loads metadata.
        2. Generates/Loads raw features (Images + Density).
        3. Encodes tabular data.
        4. Fits PCA on Train.
        5. Transforms all sets.
        6. Returns aligned DataFrames and Feature Arrays.
        """
        # 1. Load Metadata
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        if Config.DEBUG:
            train_df = train_df.head(
                Config.DEBUG_SAMPLE_SIZE * 3
            )  # approx 3 visits per patient
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE * 3)
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE * 3)

        # 2. Generate Raw Image+Density Features (Dictionary: PID -> Vector)
        # Note: We process unique patients
        train_feats_map = self.generate_raw_features(
            train_df, "train", load_cached_data
        )
        val_feats_map = self.generate_raw_features(val_df, "val", load_cached_data)
        test_feats_map = self.generate_raw_features(test_df, "test", load_cached_data)

        # 3. Fit Tabular Encoder on Train
        print("Fitting Tabular Encoder...")
        self.tabular_processor.fit(train_df)

        # 4. Construct Full Feature Matrices (Aligning with DataFrame rows)
        def construct_matrix(df, feats_map):
            # Filter df to only include patients we successfully processed (relevant for DEBUG mode)
            valid_mask = df["Patient"].isin(feats_map.keys())
            df_filtered = df[valid_mask].copy()

            # 1. Retrieve Image+Density features
            img_density_feats = np.stack(
                [feats_map[pid] for pid in df_filtered["Patient"]]
            )

            # 2. Retrieve Tabular features
            tabular_feats = self.tabular_processor.transform(df_filtered)

            # Concatenate
            X_full = np.hstack([img_density_feats, tabular_feats])
            return df_filtered, X_full.astype(np.float32)

        train_df, X_train_full = construct_matrix(train_df, train_feats_map)
        val_df, X_val_full = construct_matrix(val_df, val_feats_map)
        test_df, X_test_full = construct_matrix(test_df, test_feats_map)

        # 5. PCA Dimensionality Reduction
        # Check if PCA is already cached
        pca_cache_dir = os.path.join(Config.WORKING_DIR, "pca_model")
        pca_cached = os.path.exists(os.path.join(pca_cache_dir, "pca_components.npy"))

        if load_cached_data and pca_cached:
            print("Loading cached PCA model...")
            self.pca_reducer.load(pca_cache_dir)
            # If loaded, we assume it matches the data.
            # If dimensions mismatch, transform will fail, which is acceptable behavior.
        else:
            self.pca_reducer.fit(X_train_full)
            self.pca_reducer.save(pca_cache_dir)

        print("Transforming datasets with PCA...")
        X_train_pca = self.pca_reducer.transform(X_train_full)
        X_val_pca = self.pca_reducer.transform(X_val_full)
        X_test_pca = self.pca_reducer.transform(X_test_full)

        # 6. Save Final Processed Features (Optional, for inspection or modularity)
        np.save(os.path.join(Config.WORKING_DIR, "X_train_pca.npy"), X_train_pca)
        np.save(os.path.join(Config.WORKING_DIR, "X_val_pca.npy"), X_val_pca)
        np.save(os.path.join(Config.WORKING_DIR, "X_test_pca.npy"), X_test_pca)

        print(f"Feature Extraction Complete.")
        print(f"Train Shape: {X_train_pca.shape}")
        print(f"Val Shape:   {X_val_pca.shape}")
        print(f"Test Shape:  {X_test_pca.shape}")

        return {
            "train": (train_df, X_train_pca),
            "val": (val_df, X_val_pca),
            "test": (test_df, X_test_pca),
        }


def run_feature_extraction(load_cached_data=True):
    aggregator = FeatureAggregator()
    return aggregator.prepare_datasets(load_cached_data=load_cached_data)
