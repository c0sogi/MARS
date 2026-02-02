import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from library.config import Config
from library.image_extractor import ImageExtractor


class DataManager:
    """
    Manages data loading, feature extraction, preprocessing, and caching.
    Implements the Dual-Moment Axial-Quantile Pipeline data strategy.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.img_extractor = ImageExtractor(device)
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define paths for cached datasets
        self.train_cache = os.path.join(self.cache_dir, "dataset_train.npz")
        self.val_cache = os.path.join(self.cache_dir, "dataset_val.npz")
        self.test_cache = os.path.join(self.cache_dir, "dataset_test.npz")

    def prepare_data(self, load_cached_data=True):
        """
        Main method to prepare datasets.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed data from disk.

        Returns:
            train_data (dict): Contains 'X_fvc', 'y', 'X_unc', 'weeks', 'horizon'
            val_data (dict): Contains 'X_fvc', 'y', 'X_unc', 'weeks', 'horizon'
            test_data (dict): Contains 'X_fvc', 'X_unc', 'patient_week', 'weeks', 'horizon'
        """
        # 1. Try Loading Cache
        if load_cached_data:
            if all(
                os.path.exists(p)
                for p in [self.train_cache, self.val_cache, self.test_cache]
            ):
                print(f"Loading cached datasets from {self.cache_dir}...")
                try:
                    train_data = self._load_npz(self.train_cache)
                    val_data = self._load_npz(self.val_cache)
                    test_data = self._load_npz(self.test_cache)
                    return train_data, val_data, test_data
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing from scratch...")

        print("Processing data from scratch...")

        # 2. Load Metadata
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # 3. Extract Image Features
        # Combine all dfs to get unique patients for batch processing
        all_meta = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
        # This handles caching internally per patient
        img_features_map = self.img_extractor.extract_features(
            all_meta, load_cached_data=load_cached_data
        )

        # 4. Prepare Raw Feature Matrices
        def get_raw_features(df, is_test=False):
            # Image Features: (N, 2560)
            img_feats = np.stack([img_features_map[p] for p in df["Patient"].values])

            # Clinical Features
            # Cite debug_lesson_2: Harmonize Data Schemas via Runtime Feature Derivation
            if "Percent" not in df.columns and "Baseline_Percent" in df.columns:
                df = df.assign(Percent=df["Baseline_Percent"])

            clinical_df = df[["Age", "Sex", "SmokingStatus", "Percent"]].copy()

            # Time Features
            weeks = df["Weeks"].values.astype(np.float32)

            if is_test:
                # For test, Horizon = |Weeks - Baseline_Weeks|
                baseline = df["Baseline_Weeks"].values.astype(np.float32)
                horizon = np.abs(weeks - baseline)
                y = np.zeros(len(df))  # Dummy
                identifiers = df["Patient_Week"].values
            else:
                # For train/val, Baseline is implicitly 0 (relative weeks), so Horizon = |Weeks|
                horizon = np.abs(weeks)
                y = df["FVC"].values.astype(np.float32)
                identifiers = df["Patient"].values  # Just for tracking

            return img_feats, clinical_df, weeks, horizon, y, identifiers

        X_img_train, X_clin_train, t_train, h_train, y_train, _ = get_raw_features(
            df_train, is_test=False
        )
        X_img_val, X_clin_val, t_val, h_val, y_val, _ = get_raw_features(
            df_val, is_test=False
        )
        X_img_test, X_clin_test, t_test, h_test, y_test, ids_test = get_raw_features(
            df_test, is_test=True
        )

        # 5. PCA on Image Features (Dimensionality Reduction)
        print(f"Fitting PCA on training images (components={Config.PCA_COMPONENTS})...")
        pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        X_img_train_pca = pca.fit_transform(X_img_train)
        X_img_val_pca = pca.transform(X_img_val)
        X_img_test_pca = pca.transform(X_img_test)

        # 6. Preprocess Clinical Features
        print("Preprocessing Clinical Features...")
        numeric_features = ["Age", "Percent"]
        categorical_features = ["Sex", "SmokingStatus"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )

        X_clin_train_enc = preprocessor.fit_transform(X_clin_train)
        X_clin_val_enc = preprocessor.transform(X_clin_val)
        X_clin_test_enc = preprocessor.transform(X_clin_test)

        # 7. Construct Base Feature Matrix X_base
        # X_base = [PCA_Image_Features, Encoded_Clinical_Features]
        X_base_train = np.hstack([X_img_train_pca, X_clin_train_enc])
        X_base_val = np.hstack([X_img_val_pca, X_clin_val_enc])
        X_base_test = np.hstack([X_img_test_pca, X_clin_test_enc])

        # 8. Feature Engineering for Models

        # Scale Time and Horizon for numerical stability in linear models
        t_scaler = StandardScaler()
        t_train_scaled = t_scaler.fit_transform(t_train.reshape(-1, 1)).flatten()
        t_val_scaled = t_scaler.transform(t_val.reshape(-1, 1)).flatten()
        t_test_scaled = t_scaler.transform(t_test.reshape(-1, 1)).flatten()

        h_scaler = StandardScaler()
        h_train_scaled = h_scaler.fit_transform(h_train.reshape(-1, 1)).flatten()
        h_val_scaled = h_scaler.transform(h_val.reshape(-1, 1)).flatten()
        h_test_scaled = h_scaler.transform(h_test.reshape(-1, 1)).flatten()

        # FVC Features: [X_base, t, X_base * t]
        # This creates interaction terms allowing patient-specific slopes
        def create_fvc_features(X_base, t_scaled):
            N, D = X_base.shape
            t_col = t_scaled.reshape(-1, 1)
            interaction = X_base * t_col
            return np.hstack([X_base, t_col, interaction])

        # Uncertainty Features: [X_base, horizon]
        # Horizon captures temporal heteroscedasticity
        def create_unc_features(X_base, h_scaled):
            h_col = h_scaled.reshape(-1, 1)
            return np.hstack([X_base, h_col])

        X_fvc_train = create_fvc_features(X_base_train, t_train_scaled)
        X_fvc_val = create_fvc_features(X_base_val, t_val_scaled)
        X_fvc_test = create_fvc_features(X_base_test, t_test_scaled)

        X_unc_train = create_unc_features(X_base_train, h_train_scaled)
        X_unc_val = create_unc_features(X_base_val, h_val_scaled)
        X_unc_test = create_unc_features(X_base_test, h_test_scaled)

        # 9. Pack and Save
        train_data = {
            "X_fvc": X_fvc_train,
            "y": y_train,
            "X_unc": X_unc_train,
            "weeks": t_train,
            "horizon": h_train,
        }
        val_data = {
            "X_fvc": X_fvc_val,
            "y": y_val,
            "X_unc": X_unc_val,
            "weeks": t_val,
            "horizon": h_val,
        }
        test_data = {
            "X_fvc": X_fvc_test,
            "X_unc": X_unc_test,
            "patient_week": ids_test,
            "weeks": t_test,
            "horizon": h_test,
        }

        self._save_npz(self.train_cache, train_data)
        self._save_npz(self.val_cache, val_data)
        self._save_npz(self.test_cache, test_data)

        print("Data preparation complete.")
        return train_data, val_data, test_data

    def _save_npz(self, path, data_dict):
        """Saves dictionary of numpy arrays to compressed npz."""
        np.savez_compressed(path, **data_dict)

    def _load_npz(self, path):
        """Loads compressed npz into dictionary."""
        loaded = np.load(path, allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}
