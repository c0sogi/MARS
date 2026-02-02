import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from library.config import Config
from library.utils import get_logger, seed_everything


class DimensionalityReducer:
    """
    Handles Independent Component Compression (PCA) for multi-stream features
    and fuses them with metadata.
    """

    def __init__(self):
        self.config = Config
        self.logger = get_logger("DimensionalityReducer")
        self.working_dir = self.config.WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Define paths for the final processed data
        self.final_paths = {
            "X_train": os.path.join(self.working_dir, "final_X_train.npy"),
            "y_train": os.path.join(self.working_dir, "final_y_train.npy"),
            "X_val": os.path.join(self.working_dir, "final_X_val.npy"),
            "y_val": os.path.join(self.working_dir, "final_y_val.npy"),
            "X_test": os.path.join(self.working_dir, "final_X_test.npy"),
            "ids_test": os.path.join(self.working_dir, "final_ids_test.npy"),
        }

    def _load_metadata_features(self, split):
        """
        Loads metadata from CSV, extracts specific columns, and scales them.
        Returns features (np.array) and targets (np.array) or ids.
        """
        if split == "train":
            path = self.config.TRAIN_META_PATH
        elif split == "val":
            path = self.config.VAL_META_PATH
        else:
            path = self.config.TEST_META_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)

        # Extract Metadata Features
        # Ensure columns exist (robustness)
        for col in self.config.METADATA_COLS:
            if col not in df.columns:
                df[col] = 0

        meta_features = df[self.config.METADATA_COLS].values.astype(np.float32)
        meta_features = meta_features * self.config.METADATA_SCALE

        # Extract Targets or IDs
        if split in ["train", "val"]:
            if "Pawpularity" not in df.columns:
                raise ValueError(
                    f"Target column 'Pawpularity' missing in {split} metadata."
                )
            targets = df["Pawpularity"].values.astype(np.float32)
            return meta_features, targets
        else:
            ids = df["Id"].values
            return meta_features, ids

    def run(self, load_cached_data=True):
        """
        Main execution method.
        1. Checks cache for final fused datasets.
        2. If not cached:
           - Fits PCA per backbone on Train features.
           - Transforms Train/Val/Test features.
           - Concatenates compressed features from all backbones.
           - Appends scaled metadata.
           - Saves final arrays to disk.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, ids_test)
        """
        self.logger.info("Starting Dimensionality Reduction & Feature Fusion...")

        # 1. Check Cache
        all_cached = all(os.path.exists(p) for p in self.final_paths.values())
        if load_cached_data and all_cached:
            self.logger.info("Found cached final datasets. Loading...")
            try:
                data = {}
                for k, v in self.final_paths.items():
                    data[k] = np.load(
                        v, allow_pickle=True
                    )  # allow_pickle needed for object arrays (IDs)

                self.logger.info("Loaded cached data successfully.")
                return (
                    data["X_train"],
                    data["y_train"],
                    data["X_val"],
                    data["y_val"],
                    data["X_test"],
                    data["ids_test"],
                )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Process Backbones
        X_train_parts = []
        X_val_parts = []
        X_test_parts = []

        # Iterate through backbones defined in Config
        for friendly_name in self.config.BACKBONES.keys():
            self.logger.info(f"Processing backbone stream: {friendly_name}")

            # Construct paths for raw features extracted by FeatureEngine
            train_feat_path = os.path.join(
                self.working_dir, f"{friendly_name}_train_features.npy"
            )
            val_feat_path = os.path.join(
                self.working_dir, f"{friendly_name}_val_features.npy"
            )
            test_feat_path = os.path.join(
                self.working_dir, f"{friendly_name}_test_features.npy"
            )

            # Verify existence
            if not (
                os.path.exists(train_feat_path)
                and os.path.exists(val_feat_path)
                and os.path.exists(test_feat_path)
            ):
                raise FileNotFoundError(
                    f"Raw features for '{friendly_name}' not found in {self.working_dir}. "
                    "Please run FeatureEngine first."
                )

            # Load raw features
            raw_train = np.load(train_feat_path)
            raw_val = np.load(val_feat_path)
            raw_test = np.load(test_feat_path)

            self.logger.info(f"  Raw Train Shape: {raw_train.shape}")

            # Initialize and Fit PCA
            # Retain specified variance (e.g., 0.95)
            pca = PCA(
                n_components=self.config.PCA_VARIANCE, random_state=self.config.SEED
            )

            self.logger.info(f"  Fitting PCA (Variance={self.config.PCA_VARIANCE})...")
            pca.fit(raw_train)

            n_components = pca.n_components_
            self.logger.info(f"  PCA retained {n_components} components.")

            # Save PCA model for reproducibility/inference
            pca_path = os.path.join(self.working_dir, f"pca_{friendly_name}.joblib")
            joblib.dump(pca, pca_path)

            # Transform all splits
            t_train = pca.transform(raw_train).astype(np.float32)
            t_val = pca.transform(raw_val).astype(np.float32)
            t_test = pca.transform(raw_test).astype(np.float32)

            X_train_parts.append(t_train)
            X_val_parts.append(t_val)
            X_test_parts.append(t_test)

            # Explicitly free memory
            del raw_train, raw_val, raw_test, t_train, t_val, t_test

        # 3. Concatenate Image Features
        self.logger.info("Concatenating compressed backbone features...")
        X_train_img = np.concatenate(X_train_parts, axis=1)
        X_val_img = np.concatenate(X_val_parts, axis=1)
        X_test_img = np.concatenate(X_test_parts, axis=1)

        self.logger.info(f"Fused Image Features Shape (Train): {X_train_img.shape}")

        # 4. Load and Concatenate Metadata
        self.logger.info("Loading and appending metadata...")

        meta_train, y_train = self._load_metadata_features("train")
        meta_val, y_val = self._load_metadata_features("val")
        meta_test, ids_test = self._load_metadata_features("test")

        # Concatenate [Image Features | Metadata]
        X_train_final = np.concatenate([X_train_img, meta_train], axis=1)
        X_val_final = np.concatenate([X_val_img, meta_val], axis=1)
        X_test_final = np.concatenate([X_test_img, meta_test], axis=1)

        self.logger.info(f"Final Feature Matrix Shape (Train): {X_train_final.shape}")

        # 5. Save to Cache
        self.logger.info("Saving final datasets to cache...")
        np.save(self.final_paths["X_train"], X_train_final)
        np.save(self.final_paths["y_train"], y_train)
        np.save(self.final_paths["X_val"], X_val_final)
        np.save(self.final_paths["y_val"], y_val)
        np.save(self.final_paths["X_test"], X_test_final)
        np.save(self.final_paths["ids_test"], ids_test)

        return X_train_final, y_train, X_val_final, y_val, X_test_final, ids_test
