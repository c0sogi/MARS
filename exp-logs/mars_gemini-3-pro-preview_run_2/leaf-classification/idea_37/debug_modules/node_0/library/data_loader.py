import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, LabelEncoder
from library.config import Config
from library.utils import load_metadata


class DataLoader:
    """
    Handles data ingestion, dual-stream preprocessing, and splitting for the
    Constrained-Basis Dual-Stream Generative Ensemble.
    """

    def __init__(self):
        self.target_col = "species"
        self.id_col = "id"
        self.image_path_col = "image_path"

    def _get_feature_cols(self, df):
        """
        Identifies feature columns (Margin, Shape, Texture) by excluding metadata.
        """
        exclude = {self.id_col, self.target_col, self.image_path_col}
        cols = [c for c in df.columns if c not in exclude]
        # Sort to ensure consistent column ordering across splits
        cols.sort()
        return cols

    def _process_streams(self, X_fit, X_transform_list):
        """
        Fits the Dual-Stream transformers on X_fit and transforms all datasets in X_transform_list.

        Stream A: Parametric (Yeo-Johnson) - The robust anchor.
        Stream B: Constrained Non-Parametric (Quantile, n=50) - The flexible expert.

        Args:
            X_fit: Data to fit transformers on (Train in Phase 1, Full in Phase 2).
            X_transform_list: List of datasets to transform.

        Returns:
            List of dictionaries, each containing 'stream_a' and 'stream_b' numpy arrays.
        """
        # Stream A: Parametric
        pt = PowerTransformer(
            method=Config.PT_METHOD, standardize=Config.PT_STANDARDIZE
        )
        pt.fit(X_fit)

        # Stream B: Constrained Non-Parametric
        # n_quantiles is strictly constrained (e.g., 50) to prevent overfitting small datasets
        qt = QuantileTransformer(
            output_distribution=Config.QT_OUTPUT_DIST,
            n_quantiles=Config.QT_N_QUANTILES,
            random_state=Config.RANDOM_STATE,
        )
        qt.fit(X_fit)

        results = []
        for X in X_transform_list:
            # Transform and strictly enforce float64 precision
            xa = pt.transform(X).astype(Config.NP_DTYPE)
            xb = qt.transform(X).astype(Config.NP_DTYPE)
            results.append({"stream_a": xa, "stream_b": xb})

        return results

    def load_phase1_data(self, load_cached_data=True):
        """
        Loads and processes data for Phase 1: Model Selection.
        Uses the standard Train (80%) / Val (20%) split.

        Returns:
            train_data (dict): {'stream_a': X, 'stream_b': X, 'y': y}
            val_data (dict):   {'stream_a': X, 'stream_b': X, 'y': y}
            classes (array):   List of class names.
        """
        cache_dir = Config.WORKING_DIR
        paths = {
            "train_a": os.path.join(cache_dir, "p1_train_a.npy"),
            "train_b": os.path.join(cache_dir, "p1_train_b.npy"),
            "train_y": os.path.join(cache_dir, "p1_train_y.npy"),
            "val_a": os.path.join(cache_dir, "p1_val_a.npy"),
            "val_b": os.path.join(cache_dir, "p1_val_b.npy"),
            "val_y": os.path.join(cache_dir, "p1_val_y.npy"),
            "classes": os.path.join(cache_dir, "p1_classes.npy"),
        }

        # 1. Try Loading from Cache
        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            print("Loading Phase 1 data from cache...")
            train_data = {
                "stream_a": np.load(paths["train_a"]),
                "stream_b": np.load(paths["train_b"]),
                "y": np.load(paths["train_y"]),
            }
            val_data = {
                "stream_a": np.load(paths["val_a"]),
                "stream_b": np.load(paths["val_b"]),
                "y": np.load(paths["val_y"]),
            }
            classes = np.load(paths["classes"], allow_pickle=True)
            return train_data, val_data, classes

        # 2. Process from Scratch
        print("Processing Phase 1 data from scratch...")
        df_train = load_metadata("train")
        df_val = load_metadata("val")

        # Extract features
        cols = self._get_feature_cols(df_train)
        X_train_raw = df_train[cols].values.astype(Config.NP_DTYPE)
        X_val_raw = df_val[cols].values.astype(Config.NP_DTYPE)

        # Apply Dual-Stream Preprocessing (Fit on Train, Transform Train & Val)
        processed = self._process_streams(X_train_raw, [X_train_raw, X_val_raw])
        train_feats = processed[0]
        val_feats = processed[1]

        # Encode Labels
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.target_col])
        y_val = le.transform(df_val[self.target_col])
        classes = le.classes_

        # Save to Cache
        np.save(paths["train_a"], train_feats["stream_a"])
        np.save(paths["train_b"], train_feats["stream_b"])
        np.save(paths["train_y"], y_train)
        np.save(paths["val_a"], val_feats["stream_a"])
        np.save(paths["val_b"], val_feats["stream_b"])
        np.save(paths["val_y"], y_val)
        np.save(paths["classes"], classes)

        train_data = {**train_feats, "y": y_train}
        val_data = {**val_feats, "y": y_val}

        return train_data, val_data, classes

    def load_phase2_data(self, load_cached_data=True):
        """
        Loads and processes data for Phase 2: Final Retraining.
        Combines Train + Val into a single training set. Loads Test set.

        Returns:
            full_data (dict): {'stream_a': X, 'stream_b': X, 'y': y}
            test_data (dict): {'stream_a': X, 'stream_b': X, 'ids': ids}
            classes (array):  List of class names.
        """
        cache_dir = Config.WORKING_DIR
        paths = {
            "full_a": os.path.join(cache_dir, "p2_full_a.npy"),
            "full_b": os.path.join(cache_dir, "p2_full_b.npy"),
            "full_y": os.path.join(cache_dir, "p2_full_y.npy"),
            "test_a": os.path.join(cache_dir, "p2_test_a.npy"),
            "test_b": os.path.join(cache_dir, "p2_test_b.npy"),
            "test_ids": os.path.join(cache_dir, "p2_test_ids.npy"),
            "classes": os.path.join(cache_dir, "p2_classes.npy"),
        }

        # 1. Try Loading from Cache
        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            print("Loading Phase 2 data from cache...")
            full_data = {
                "stream_a": np.load(paths["full_a"]),
                "stream_b": np.load(paths["full_b"]),
                "y": np.load(paths["full_y"]),
            }
            test_data = {
                "stream_a": np.load(paths["test_a"]),
                "stream_b": np.load(paths["test_b"]),
                "ids": np.load(paths["test_ids"]),
            }
            classes = np.load(paths["classes"], allow_pickle=True)
            return full_data, test_data, classes

        # 2. Process from Scratch
        print("Processing Phase 2 data from scratch...")
        df_train = load_metadata("train")
        df_val = load_metadata("val")
        df_test = load_metadata("test")

        # Combine Train and Val
        df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

        # Extract Features
        cols = self._get_feature_cols(df_full)
        X_full_raw = df_full[cols].values.astype(Config.NP_DTYPE)
        X_test_raw = df_test[cols].values.astype(Config.NP_DTYPE)

        # Apply Dual-Stream Preprocessing (Fit on Full Train, Transform Full Train & Test)
        processed = self._process_streams(X_full_raw, [X_full_raw, X_test_raw])
        full_feats = processed[0]
        test_feats = processed[1]

        # Encode Labels
        le = LabelEncoder()
        y_full = le.fit_transform(df_full[self.target_col])
        classes = le.classes_
        test_ids = df_test[self.id_col].values

        # Save to Cache
        np.save(paths["full_a"], full_feats["stream_a"])
        np.save(paths["full_b"], full_feats["stream_b"])
        np.save(paths["full_y"], y_full)
        np.save(paths["test_a"], test_feats["stream_a"])
        np.save(paths["test_b"], test_feats["stream_b"])
        np.save(paths["test_ids"], test_ids)
        np.save(paths["classes"], classes)

        full_data = {**full_feats, "y": y_full}
        test_data = {**test_feats, "ids": test_ids}

        return full_data, test_data, classes
