import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from library.config import Config
from library.image_features import process_images


class GaussianPreprocessor:
    """
    Applies Yeo-Johnson transformation and enforces Double Precision (float64).
    This ensures features conform to the normality assumptions of LDA and
    maintains numerical stability.
    """

    def __init__(self):
        self.pt = PowerTransformer(method=Config.POWER_TRANSFORMER_METHOD)

    def fit(self, X):
        self.pt.fit(X)
        return self

    def transform(self, X):
        X_trans = self.pt.transform(X)
        return X_trans.astype(np.float64)

    def fit_transform(self, X):
        X_trans = self.pt.fit_transform(X)
        return X_trans.astype(np.float64)


class DataLoader:
    """
    Manages data loading, feature extraction, and preprocessing for the ensemble.
    Handles the creation of Anchor, Orthogonal, and Synergistic views.
    """

    def __init__(self):
        self.anchor_cols = Config.MARGIN_COLS + Config.SHAPE_COLS + Config.TEXTURE_COLS
        self.ortho_cols = Config.HU_MOMENTS_COLS + Config.GEOMETRIC_SCALARS_COLS
        self.le = LabelEncoder()

    def _load_raw_data(
        self, metadata_path, cache_path, is_test=False, max_samples=None
    ):
        """
        Loads raw data from CSV and extracts/loads morphometric features.
        Handles caching and subsetting while maintaining row alignment.
        """
        # Load full metadata to ensure alignment with cache logic
        df_full = pd.read_csv(metadata_path)
        full_paths = df_full["image_path"].tolist()

        # Get full morphometric features (Source B)
        # process_images handles the caching mechanism internally (loading from parquet if exists)
        df_ortho_full = process_images(
            full_paths, load_cached_data=True, cache_path=cache_path
        )

        # Subset if debugging or max_samples specified
        if max_samples is not None and max_samples < len(df_full):
            df_main = df_full.sample(n=max_samples, random_state=Config.RANDOM_SEED)
            # Use index to align extracted features with sampled metadata
            df_ortho = df_ortho_full.loc[df_main.index]
        else:
            df_main = df_full
            df_ortho = df_ortho_full

        # Extract Provided Features (Source A)
        X_anchor = df_main[self.anchor_cols].values.astype(np.float64)

        # Extract Extracted Features (Source B)
        X_ortho = df_ortho[self.ortho_cols].values.astype(np.float64)

        ids = df_main["id"].values

        if not is_test:
            y = df_main["species"].values
            return X_anchor, X_ortho, y, ids
        else:
            return X_anchor, X_ortho, None, ids

    def get_phase1_data(self, max_samples=None):
        """
        Prepares data for Phase 1: Selection (Train/Val Split).
        Fits preprocessors on Train, transforms Train and Val.
        Returns dictionaries containing 'anchor', 'orthogonal', and 'synergistic' views.
        """
        # Load Raw Data
        X_anc_tr, X_ort_tr, y_tr, _ = self._load_raw_data(
            Config.TRAIN_METADATA_PATH, Config.CACHE_TRAIN_PATH, max_samples=max_samples
        )
        X_anc_val, X_ort_val, y_val, _ = self._load_raw_data(
            Config.VAL_METADATA_PATH, Config.CACHE_VAL_PATH, max_samples=max_samples
        )

        # Encode Labels
        # Fit on train, transform val (stratified split in metadata ensures class coverage)
        y_tr_enc = self.le.fit_transform(y_tr)
        y_val_enc = self.le.transform(y_val)

        # Initialize Preprocessors for each view
        pt_anc = GaussianPreprocessor()
        pt_ort = GaussianPreprocessor()

        # Fit & Transform Train
        X_anc_tr_trans = pt_anc.fit_transform(X_anc_tr)
        X_ort_tr_trans = pt_ort.fit_transform(X_ort_tr)

        # Transform Val (No fitting, prevents leakage)
        X_anc_val_trans = pt_anc.transform(X_anc_val)
        X_ort_val_trans = pt_ort.transform(X_ort_val)

        # Create Synergistic View (Concatenation of Transformed Views)
        X_syn_tr_trans = np.hstack([X_anc_tr_trans, X_ort_tr_trans])
        X_syn_val_trans = np.hstack([X_anc_val_trans, X_ort_val_trans])

        data = {
            "train": {
                "anchor": X_anc_tr_trans,
                "orthogonal": X_ort_tr_trans,
                "synergistic": X_syn_tr_trans,
                "y": y_tr_enc,
            },
            "val": {
                "anchor": X_anc_val_trans,
                "orthogonal": X_ort_val_trans,
                "synergistic": X_syn_val_trans,
                "y": y_val_enc,
            },
            "classes": self.le.classes_,
        }
        return data

    def get_phase2_data(self, max_samples=None):
        """
        Prepares data for Phase 2: Final Retraining (Full Train + Test).
        Fits preprocessors on Combined Train+Val, transforms Test.
        Returns dictionaries containing 'anchor', 'orthogonal', and 'synergistic' views.
        """
        # Load Raw Data (Train and Val)
        X_anc_tr, X_ort_tr, y_tr, _ = self._load_raw_data(
            Config.TRAIN_METADATA_PATH, Config.CACHE_TRAIN_PATH, max_samples=max_samples
        )
        X_anc_val, X_ort_val, y_val, _ = self._load_raw_data(
            Config.VAL_METADATA_PATH, Config.CACHE_VAL_PATH, max_samples=max_samples
        )

        # Combine Train and Val for Full Training
        X_anc_full = np.vstack([X_anc_tr, X_anc_val])
        X_ort_full = np.vstack([X_ort_tr, X_ort_val])
        y_full = np.concatenate([y_tr, y_val])

        # Load Test Data
        X_anc_test, X_ort_test, _, ids_test = self._load_raw_data(
            Config.TEST_METADATA_PATH,
            Config.CACHE_TEST_PATH,
            is_test=True,
            max_samples=max_samples,
        )

        # Encode Labels
        y_full_enc = self.le.fit_transform(y_full)

        # Initialize Preprocessors
        pt_anc = GaussianPreprocessor()
        pt_ort = GaussianPreprocessor()

        # Fit & Transform Full Train
        X_anc_full_trans = pt_anc.fit_transform(X_anc_full)
        X_ort_full_trans = pt_ort.fit_transform(X_ort_full)

        # Transform Test
        X_anc_test_trans = pt_anc.transform(X_anc_test)
        X_ort_test_trans = pt_ort.transform(X_ort_test)

        # Create Synergistic View
        X_syn_full_trans = np.hstack([X_anc_full_trans, X_ort_full_trans])
        X_syn_test_trans = np.hstack([X_anc_test_trans, X_ort_test_trans])

        data = {
            "train": {
                "anchor": X_anc_full_trans,
                "orthogonal": X_ort_full_trans,
                "synergistic": X_syn_full_trans,
                "y": y_full_enc,
            },
            "test": {
                "anchor": X_anc_test_trans,
                "orthogonal": X_ort_test_trans,
                "synergistic": X_syn_test_trans,
                "ids": ids_test,
            },
            "classes": self.le.classes_,
        }
        return data
