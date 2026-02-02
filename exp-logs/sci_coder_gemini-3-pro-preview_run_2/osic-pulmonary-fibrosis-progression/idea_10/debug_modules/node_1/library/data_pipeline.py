import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from library.config import Config
from library.utils import seed_everything
from library.feature_extractor import FeatureExtractor


class TabularProcessor:
    """
    Handles preprocessing of tabular clinical data.
    1. Generates Baseline columns (FVC, Percent, Weeks) for training data.
    2. Computes relative time (Weeks_Passed).
    3. Encodes and scales static features.
    """

    def __init__(self):
        self.preprocessor = None
        self.feature_cols = []

    def _generate_baselines(self, df):
        """
        For training/val data, identifies the earliest visit as the baseline.
        For test data, assumes Baseline columns already exist.
        """
        df = df.copy()

        # Check if Baseline columns already exist (Test set structure)
        required_cols = ["Baseline_Weeks", "Baseline_FVC", "Baseline_Percent"]
        if all(col in df.columns for col in required_cols):
            # Calculate Weeks_Passed
            df["Weeks_Passed"] = df["Weeks"] - df["Baseline_Weeks"]
            return df

        # Logic for Train/Val sets: Find min weeks per patient
        # Group by Patient and find the row with min Weeks
        # We sort by Weeks and take the first one
        baseline_df = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )

        # Select relevant columns and rename
        baseline_cols = {
            "Weeks": "Baseline_Weeks",
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
        }
        baseline_df = baseline_df[["Patient"] + list(baseline_cols.keys())].rename(
            columns=baseline_cols
        )

        # Merge back to original dataframe
        df = pd.merge(df, baseline_df, on="Patient", how="left")

        # Calculate Weeks_Passed
        df["Weeks_Passed"] = df["Weeks"] - df["Baseline_Weeks"]

        return df

    def fit(self, df):
        """
        Fits the scaler and encoder on the training dataframe.
        """
        df_processed = self._generate_baselines(df)

        # Define columns
        num_cols = ["Age", "Baseline_FVC", "Baseline_Percent"]
        cat_cols = ["Sex", "SmokingStatus"]

        # Build Pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    cat_cols,
                ),
            ]
        )

        self.preprocessor.fit(df_processed)
        return self

    def transform(self, df):
        """
        Transforms the dataframe to get static feature vector and time feature.
        Returns:
            static_features (np.array): Scaled/Encoded features.
            weeks_passed (np.array): Relative time.
            targets (np.array): FVC values (if available).
        """
        df_processed = self._generate_baselines(df)

        # Extract Static Features
        static_features = self.preprocessor.transform(df_processed)

        # Extract Time
        weeks_passed = df_processed["Weeks_Passed"].values.astype(np.float32)

        # Extract Target if available
        targets = None
        if "FVC" in df_processed.columns:
            targets = df_processed["FVC"].values.astype(np.float32)

        return static_features, weeks_passed, targets, df_processed["Patient"].values


class DataPipeline:
    """
    Orchestrates the data preparation process:
    1. Extracts/Loads Visual Features.
    2. Processes Tabular Data.
    3. Fuses Features and applies PCA.
    4. Generates Interaction Terms.
    5. Caches results.
    """

    def __init__(self):
        self.tabular_processor = TabularProcessor()
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        self.feature_extractor = FeatureExtractor()

    def _load_or_generate_visual_features(
        self, metadata_path, split_name, load_cached_data
    ):
        """
        Wrapper to load visual features using the library's FeatureExtractor.
        """
        df = pd.read_csv(metadata_path)
        features, patient_ids = self.feature_extractor.generate_features(
            df, save_name=split_name, load_cached_data=load_cached_data
        )
        # Convert to dictionary for fast lookup
        # features shape: (N_patients, 3852)
        feat_dict = {pid: features[i] for i, pid in enumerate(patient_ids)}
        return feat_dict

    def _align_features(self, df_path, visual_feat_dict, is_training=False):
        """
        Aligns visual features with tabular rows.
        """
        df = pd.read_csv(df_path)

        # Fit tabular processor if training
        if is_training:
            self.tabular_processor.fit(df)

        static_tabular, weeks_passed, targets, patient_ids = (
            self.tabular_processor.transform(df)
        )

        # Align visual features
        aligned_visual = []
        valid_indices = []

        # Create a zero vector for missing patients (fallback)
        # Get dimension from an arbitrary value in dict
        feat_dim = next(iter(visual_feat_dict.values())).shape[0]
        zero_vec = np.zeros(feat_dim, dtype=np.float32)

        for i, pid in enumerate(patient_ids):
            if pid in visual_feat_dict:
                aligned_visual.append(visual_feat_dict[pid])
            else:
                # This should ideally not happen if metadata is consistent
                aligned_visual.append(zero_vec)

        aligned_visual = np.array(aligned_visual, dtype=np.float32)

        # Concatenate: [Visual (3852) | Tabular (~10)]
        combined_features = np.hstack([aligned_visual, static_tabular])

        return combined_features, weeks_passed, targets

    def process_dataset(
        self, metadata_path, split_name, load_cached_data=True, is_training=False
    ):
        """
        Main entry point to process a dataset (train/val/test).
        Returns dictionary containing X_fvc, X_unc, y (if available).
        """
        seed_everything()
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Cache paths
        cache_prefix = os.path.join(Config.WORKING_DIR, split_name)
        path_X_fvc = f"{cache_prefix}_X_fvc.npy"
        path_X_unc = f"{cache_prefix}_X_unc.npy"
        path_y = f"{cache_prefix}_y.npy"
        path_meta = f"{cache_prefix}_meta.parquet"  # Cache processed metadata for submission ID alignment

        # 1. Try Loading from Cache
        if load_cached_data:
            if os.path.exists(path_X_fvc) and os.path.exists(path_X_unc):
                print(f"[{split_name}] Loading processed data from cache...")
                X_fvc = np.load(path_X_fvc)
                X_unc = np.load(path_X_unc)
                y = np.load(path_y) if os.path.exists(path_y) else None
                return {"X_fvc": X_fvc, "X_unc": X_unc, "y": y}

        # 2. Compute from Scratch
        print(f"[{split_name}] Processing data pipeline...")

        # A. Get Visual Features
        visual_feat_dict = self._load_or_generate_visual_features(
            metadata_path, split_name, load_cached_data
        )

        # B. Align and Process Tabular Data
        # Note: self.tabular_processor must be fitted on train before calling this on val/test
        # This is handled by the caller ensuring process_dataset(train, is_training=True) is called first
        raw_X, weeks, targets = self._align_features(
            metadata_path, visual_feat_dict, is_training=is_training
        )

        # C. PCA Dimensionality Reduction
        if is_training:
            # Adjust PCA components if samples are fewer than configured components
            n_samples, n_features = raw_X.shape
            n_components = min(n_samples, n_features, Config.PCA_COMPONENTS)

            if n_components < self.pca.n_components:
                print(
                    f"[{split_name}] Adjusting PCA components from {self.pca.n_components} to {n_components} due to data size."
                )
                self.pca = PCA(n_components=n_components, random_state=Config.SEED)

            print(f"[{split_name}] Fitting PCA on {raw_X.shape[0]} samples...")
            self.pca.fit(raw_X)

        X_pca = self.pca.transform(raw_X)  # (N_samples, n_components)

        # D. Feature Engineering for Models

        # FVC Model Features: [PCA, Weeks, PCA * Weeks]
        # Reshape weeks for broadcasting: (N, 1)
        weeks_2d = weeks[:, np.newaxis]

        # Interaction terms
        interactions = (
            X_pca * weeks_2d
        )  # Element-wise broadcast: (N, 40) * (N, 1) -> (N, 40)

        # Concatenate: PCA (40) + Weeks (1) + Interactions (40) = 81 features
        X_fvc = np.hstack([X_pca, weeks_2d, interactions])

        # Uncertainty Model Features: [PCA, |Weeks|]
        # Horizon = Absolute time delta
        horizon_2d = np.abs(weeks_2d)

        # Concatenate: PCA (40) + Horizon (1) = 41 features
        X_unc = np.hstack([X_pca, horizon_2d])

        # 3. Save to Cache
        print(f"[{split_name}] Saving processed data to cache...")
        np.save(path_X_fvc, X_fvc)
        np.save(path_X_unc, X_unc)
        if targets is not None:
            np.save(path_y, targets)

        return {"X_fvc": X_fvc, "X_unc": X_unc, "y": targets}
