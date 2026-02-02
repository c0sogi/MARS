import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import Config
from library.cnn_feature_extractor import VisualFeatureExtractor
from library.dicom_processing import load_scan, get_pixels_hu


class DatasetBuilder:
    """
    Orchestrates the creation of regression datasets for the Orthogonal MIP-Texture pipeline.
    Handles metadata augmentation, feature extraction (CNN + Volumetrics), PCA reduction,
    and the construction of specialized matrices for FVC and Uncertainty models.
    """

    def __init__(self):
        self.pca = PCA(n_components=Config.N_PCA_COMPONENTS, random_state=Config.SEED)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.cnn_extractor = VisualFeatureExtractor()

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def _load_and_augment_metadata(self):
        """
        Loads metadata and ensures Train/Val have Baseline columns similar to Test.
        """
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        def augment_baseline(df):
            # If already has baseline columns, skip (Test set)
            if "Baseline_FVC" in df.columns:
                return df

            # Find baseline (min weeks) for each patient
            # We assume the row with min weeks is the baseline visit
            baseline_info = df.loc[df.groupby("Patient")["Weeks"].idxmin()]
            baseline_info = baseline_info[["Patient", "FVC", "Percent", "Weeks"]]
            baseline_info = baseline_info.rename(
                columns={
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Weeks": "Baseline_Weeks",
                }
            )

            # Merge back
            df_aug = pd.merge(df, baseline_info, on="Patient", how="left")
            return df_aug

        train_df = augment_baseline(train_df)
        val_df = augment_baseline(val_df)

        # Calculate Time Delta (Weeks from Baseline)
        # Note: In test set, 'Weeks' is the target week, 'Baseline_Weeks' is the CT week.
        for df in [train_df, val_df, test_df]:
            df["Time_Delta"] = df["Weeks"] - df["Baseline_Weeks"]

        return train_df, val_df, test_df

    def _get_volumetrics(self, df, split_name, load_cached_data=True):
        """
        Calculates explicit volumetrics (Mean Density, Approx Volume) from DICOMs.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"volumetrics_{split_name}.npy")

        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                if len(data) == len(df):
                    return data
            except Exception:
                pass

        print(f"Generating volumetrics for {split_name}...")

        # Get unique patients to avoid redundant IO
        unique_patients = df[["Patient", "dcm_path"]].drop_duplicates()
        patient_vol_map = {}

        for _, row in unique_patients.iterrows():
            pid = row["Patient"]
            rel_path = row["dcm_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            try:
                # We use the library functions but we need the full volume
                # This is computationally expensive, hence caching is vital
                scans = load_scan(full_path)
                volume = get_pixels_hu(scans)

                # Mean Density (Mean HU of lung tissue)
                # Volume is masked to [HU_MIN, HU_MAX] in get_pixels_hu
                mean_density = np.mean(volume)

                # Approx Volume (Voxel count > -1000, i.e., not air)
                # Since get_pixels_hu clips to HU_MIN (-1000), we check > HU_MIN
                # Adding a small epsilon or checking strictly greater
                lung_vol = np.sum(volume > Config.HU_MIN)

                patient_vol_map[pid] = [mean_density, lung_vol]

            except Exception as e:
                print(f"Error calculating volumetrics for {pid}: {e}")
                patient_vol_map[pid] = [0.0, 0.0]  # Fallback

        # Map back to dataframe
        vol_features = []
        for pid in df["Patient"]:
            vol_features.append(patient_vol_map.get(pid, [0.0, 0.0]))

        vol_features = np.array(vol_features, dtype=np.float32)
        np.save(cache_path, vol_features)
        return vol_features

    def _process_tabular(self, train_df, val_df, test_df, train_vol, val_vol, test_vol):
        """
        Encodes and scales tabular data.
        """
        # Define columns
        num_cols = ["Age", "Baseline_FVC", "Baseline_Percent"]
        cat_cols = ["Sex", "SmokingStatus"]

        # 1. Numerical Features
        # Combine Clinical Num + Volumetrics
        X_num_train = np.hstack([train_df[num_cols].values, train_vol])
        X_num_val = np.hstack([val_df[num_cols].values, val_vol])
        X_num_test = np.hstack([test_df[num_cols].values, test_vol])

        # Fit Scaler on Train
        self.scaler.fit(X_num_train)

        X_num_train = self.scaler.transform(X_num_train)
        X_num_val = self.scaler.transform(X_num_val)
        X_num_test = self.scaler.transform(X_num_test)

        # 2. Categorical Features
        self.encoder.fit(train_df[cat_cols])

        X_cat_train = self.encoder.transform(train_df[cat_cols])
        X_cat_val = self.encoder.transform(val_df[cat_cols])
        X_cat_test = self.encoder.transform(test_df[cat_cols])

        # Concatenate
        X_tab_train = np.hstack([X_num_train, X_cat_train])
        X_tab_val = np.hstack([X_num_val, X_cat_val])
        X_tab_test = np.hstack([X_num_test, X_cat_test])

        return X_tab_train, X_tab_val, X_tab_test

    def _apply_pca(self, train_feats, val_feats, test_feats, load_cached_data=True):
        """
        Fits PCA on training features and transforms all sets.
        """
        cache_path_train = os.path.join(Config.CACHE_DIR, "pca_features_train.npy")
        cache_path_val = os.path.join(Config.CACHE_DIR, "pca_features_val.npy")
        cache_path_test = os.path.join(Config.CACHE_DIR, "pca_features_test.npy")

        if (
            load_cached_data
            and os.path.exists(cache_path_train)
            and os.path.exists(cache_path_val)
            and os.path.exists(cache_path_test)
        ):
            return (
                np.load(cache_path_train),
                np.load(cache_path_val),
                np.load(cache_path_test),
            )

        print("Fitting PCA on training visual features...")
        self.pca.fit(train_feats)

        pca_train = self.pca.transform(train_feats)
        pca_val = self.pca.transform(val_feats)
        pca_test = self.pca.transform(test_feats)

        np.save(cache_path_train, pca_train)
        np.save(cache_path_val, pca_val)
        np.save(cache_path_test, pca_test)

        return pca_train, pca_val, pca_test

    def generate_datasets(self, load_cached_data=True):
        """
        Main entry point to generate all datasets required for training and inference.

        Returns:
            train_data (dict): Keys 'X_fvc', 'y', 'X_unc', 'meta'
            val_data (dict): Keys 'X_fvc', 'y', 'X_unc', 'meta'
            test_data (dict): Keys 'X_fvc', 'X_unc', 'meta'
        """
        # 1. Metadata
        train_df, val_df, test_df = self._load_and_augment_metadata()

        # 2. Visual Features (CNN)
        train_cnn = self.cnn_extractor.generate_features(
            train_df, "train", load_cached_data
        )
        val_cnn = self.cnn_extractor.generate_features(val_df, "val", load_cached_data)
        test_cnn = self.cnn_extractor.generate_features(
            test_df, "test", load_cached_data
        )

        # 3. Volumetrics
        train_vol = self._get_volumetrics(train_df, "train", load_cached_data)
        val_vol = self._get_volumetrics(val_df, "val", load_cached_data)
        test_vol = self._get_volumetrics(test_df, "test", load_cached_data)

        # 4. PCA Reduction
        train_pca, val_pca, test_pca = self._apply_pca(
            train_cnn, val_cnn, test_cnn, load_cached_data
        )

        # 5. Tabular Processing
        train_tab, val_tab, test_tab = self._process_tabular(
            train_df, val_df, test_df, train_vol, val_vol, test_vol
        )

        # 6. Construct Matrices
        def build_matrices(df, pca_feats, tab_feats):
            time_delta = df["Time_Delta"].values.reshape(-1, 1)

            # --- FVC Model Matrix ---
            # Features: [Tabular, PCA, Time, Interactions(PCA*Time)]
            # Interactions: Multiply each PCA component by Time_Delta
            interactions = pca_feats * time_delta

            X_fvc = np.hstack([tab_feats, pca_feats, time_delta, interactions])

            # --- Uncertainty Model Matrix ---
            # Features: [Tabular, PCA, Horizon]
            # Horizon is abs(Time_Delta)
            horizon = np.abs(time_delta)

            X_unc = np.hstack([tab_feats, pca_feats, horizon])

            return X_fvc.astype(np.float32), X_unc.astype(np.float32)

        X_fvc_train, X_unc_train = build_matrices(train_df, train_pca, train_tab)
        X_fvc_val, X_unc_val = build_matrices(val_df, val_pca, val_tab)
        X_fvc_test, X_unc_test = build_matrices(test_df, test_pca, test_tab)

        # Targets
        y_train = train_df["FVC"].values.astype(np.float32)
        y_val = val_df["FVC"].values.astype(np.float32)

        return {
            "train": {
                "X_fvc": X_fvc_train,
                "y": y_train,
                "X_unc": X_unc_train,
                "meta": train_df,
            },
            "val": {"X_fvc": X_fvc_val, "y": y_val, "X_unc": X_unc_val, "meta": val_df},
            "test": {"X_fvc": X_fvc_test, "X_unc": X_unc_test, "meta": test_df},
        }
