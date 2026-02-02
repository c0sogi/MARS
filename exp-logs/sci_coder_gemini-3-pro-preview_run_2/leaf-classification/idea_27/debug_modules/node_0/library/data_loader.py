import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.features import ZernikeExtractor
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


class DataLoader:
    """
    Handles data ingestion, feature extraction coordination, and formatting
    for the Orthogonal-Basis Hybrid-Complexity Ensemble.
    """

    def __init__(self):
        self.zernike_extractor = ZernikeExtractor()

    def _load_metadata(self):
        """Loads the train, validation, and test metadata CSVs."""
        if not all(
            [
                pd.io.common.file_exists(Config.TRAIN_METADATA_PATH),
                pd.io.common.file_exists(Config.VAL_METADATA_PATH),
                pd.io.common.file_exists(Config.TEST_METADATA_PATH),
            ]
        ):
            raise FileNotFoundError("One or more metadata files are missing.")

        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        return train_df, val_df, test_df

    def _extract_global_features(self, df):
        """
        Extracts the provided tabular features (Margin, Shape, Texture).
        Returns a float64 numpy array.
        """
        # Select columns that start with margin, shape, or texture
        feature_cols = [
            c
            for c in df.columns
            if c.startswith("margin")
            or c.startswith("shape")
            or c.startswith("texture")
        ]
        return df[feature_cols].values.astype(np.float64)

    def _get_aligned_zernike_features(self, meta_df, zernike_df):
        """
        Aligns Zernike features with the metadata DataFrame using 'id'.
        """
        # Merge on ID to ensure rows match exactly
        merged = pd.merge(
            meta_df[["id"]], zernike_df, on="id", how="left", validate="one_to_one"
        )

        # Drop ID and convert to numpy array
        feat_cols = [c for c in merged.columns if c != "id"]

        # Fill NaNs with 0.0 (in case of extraction failure, though unlikely)
        if merged[feat_cols].isna().any().any():
            logger.warning("NaNs detected in Zernike features. Filling with 0.0.")
            merged[feat_cols] = merged[feat_cols].fillna(0.0)

        return merged[feat_cols].values.astype(np.float64)

    def load_data(self, load_cached=True):
        """
        Main entry point to load all data splits and feature views.

        Args:
            load_cached (bool): Whether to use cached Zernike features.

        Returns:
            dict: A dictionary containing 'train', 'val', 'test' dictionaries
                  and the 'label_encoder'.
        """
        logger.info("Starting data loading process...")

        # 1. Load Metadata
        train_df, val_df, test_df = self._load_metadata()
        logger.info(
            f"Metadata loaded. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        # 2. Extract Global Features (Tier 1)
        logger.info("Processing Global features (Margin, Shape, Texture)...")
        X_train_global = self._extract_global_features(train_df)
        X_val_global = self._extract_global_features(val_df)
        X_test_global = self._extract_global_features(test_df)

        # 3. Extract Zernike Features (Tier 2)
        logger.info("Processing Zernike features (Spectral Shape)...")

        # Train
        zernike_train_df = self.zernike_extractor.extract_features(
            metadata_path=Config.TRAIN_METADATA_PATH,
            cache_path=Config.CACHE_ZERNIKE_TRAIN,
            load_cached=load_cached,
        )
        X_train_zernike = self._get_aligned_zernike_features(train_df, zernike_train_df)

        # Val
        zernike_val_df = self.zernike_extractor.extract_features(
            metadata_path=Config.VAL_METADATA_PATH,
            cache_path=Config.CACHE_ZERNIKE_VAL,
            load_cached=load_cached,
        )
        X_val_zernike = self._get_aligned_zernike_features(val_df, zernike_val_df)

        # Test
        zernike_test_df = self.zernike_extractor.extract_features(
            metadata_path=Config.TEST_METADATA_PATH,
            cache_path=Config.CACHE_ZERNIKE_TEST,
            load_cached=load_cached,
        )
        X_test_zernike = self._get_aligned_zernike_features(test_df, zernike_test_df)

        # 4. Process Labels
        logger.info("Encoding target labels...")
        le = LabelEncoder()
        # Fit on combined train and val to ensure all classes are known
        all_species = pd.concat([train_df["species"], val_df["species"]]).unique()
        le.fit(all_species)

        y_train = le.transform(train_df["species"])
        y_val = le.transform(val_df["species"])

        # 5. Extract IDs
        ids_train = train_df["id"].values
        ids_val = val_df["id"].values
        ids_test = test_df["id"].values

        # 6. Construct Output
        data = {
            "train": {
                "X_global": X_train_global,
                "X_zernike": X_train_zernike,
                "y": y_train,
                "ids": ids_train,
            },
            "val": {
                "X_global": X_val_global,
                "X_zernike": X_val_zernike,
                "y": y_val,
                "ids": ids_val,
            },
            "test": {
                "X_global": X_test_global,
                "X_zernike": X_test_zernike,
                "ids": ids_test,
            },
            "label_encoder": le,
        }

        logger.info("Data loading complete.")
        logger.info(f"Global Feature Dim: {X_train_global.shape[1]}")
        logger.info(f"Zernike Feature Dim: {X_train_zernike.shape[1]}")

        return data
