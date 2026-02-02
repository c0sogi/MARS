import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from library.utils import (
    seed_everything,
    ensure_dir,
    load_metadata,
    calculate_metric,
    WORKING_DIR,
)
from library.feature_extractor import FeatureExtractor
from library.densification import Densifier
from library.model_factory import create_pipeline


class CrossValidator:
    """
    Manages the 10-Fold Stratified Cross-Validation workflow.
    Handles data preparation, densification, model training, and evaluation.
    """

    def __init__(self, n_splits=10, random_state=42, cache_subdir="idea_42"):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cache_subdir = cache_subdir
        self.models_dir = os.path.join(WORKING_DIR, cache_subdir, "models")
        ensure_dir(self.models_dir)
        seed_everything(random_state)

    def run_cv(self, load_cached_data=True, limit=None):
        """
        Executes the Cross-Validation loop.

        Args:
            load_cached_data (bool): Whether to use cached features/densified data.
            limit (int): Optional limit on data size for debugging.

        Returns:
            float: The average log loss across all folds.
        """
        print(f"Starting {self.n_splits}-Fold Cross-Validation...")

        # 1. Prepare Data (Load & Concat Train/Val)
        dino_all, conv_all, tab_all, ids_all, y_all, label_encoder = (
            self._prepare_full_dataset(load_cached_data=load_cached_data, limit=limit)
        )

        # Save classes for submission generation later
        joblib.dump(
            label_encoder.classes_, os.path.join(self.models_dir, "classes.pkl")
        )

        # 2. Setup CV
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        fold_scores = []
        densifier = Densifier(cache_subdir=self.cache_subdir)

        # 3. CV Loop
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dino_all, y_all)):
            print(f"\n--- Fold {fold_idx} ---")

            # --- Data Slicing ---
            # Train Slice
            dino_train = dino_all[train_idx]
            conv_train = conv_all[train_idx]
            tab_train = tab_all[train_idx]
            y_train_raw = y_all[train_idx]
            ids_train = ids_all[train_idx]

            # Val Slice
            dino_val = dino_all[val_idx]
            conv_val = conv_all[val_idx]
            tab_val = tab_all[val_idx]
            y_val_raw = y_all[val_idx]
            ids_val = ids_all[val_idx]

            # --- Densification (Train: Convex-Hull 6x) ---
            # We use a unique split name per fold to cache the densified arrays correctly
            dino_train_dense, conv_train_dense, tab_train_dense, y_train_dense, _ = (
                densifier.densify_training_data(
                    dino_train,
                    conv_train,
                    tab_train,
                    y_train_raw,
                    ids_train,
                    split_name=f"fold_{fold_idx}_train",
                    load_cached_data=load_cached_data,
                )
            )

            # Construct X_train by horizontally stacking features
            # [DINO (1024) | ConvNeXt (1536) | Tabular (192)]
            X_train = np.hstack([dino_train_dense, conv_train_dense, tab_train_dense])

            # --- Densification (Val: Canonical 3x) ---
            dino_val_canon, conv_val_canon, tab_val_canon, _ = (
                densifier.densify_inference_data(
                    dino_val,
                    conv_val,
                    tab_val,
                    ids_val,
                    split_name=f"fold_{fold_idx}_val",
                    load_cached_data=load_cached_data,
                )
            )

            X_val_expanded = np.hstack([dino_val_canon, conv_val_canon, tab_val_canon])

            # --- Model Training ---
            # Create pipeline
            pipeline = create_pipeline(dino_dim=1024, conv_dim=1536, tab_dim=192)

            # Fit
            pipeline.fit(X_train, y_train_dense)

            # --- Evaluation (Aggregation) ---
            # Predict on expanded validation set (3 views per image)
            probs_expanded = pipeline.predict_proba(
                X_val_expanded
            )  # (N_val * 3, n_classes)

            # Reshape to (N_val, 3, n_classes) and average across views
            n_val_samples = len(y_val_raw)
            n_classes = len(label_encoder.classes_)

            probs_reshaped = probs_expanded.reshape(n_val_samples, 3, n_classes)
            probs_final = np.mean(probs_reshaped, axis=1)  # (N_val, n_classes)

            # Calculate Metric
            score = calculate_metric(y_val_raw, probs_final)
            fold_scores.append(score)
            print(f"Fold {fold_idx} Log Loss: {score:.15f}")

            # --- Save Model ---
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold_idx}.pkl")
            joblib.dump(pipeline, model_path)

        # 4. Summary
        avg_score = np.mean(fold_scores)
        print(f"\nCV Complete. Average Log Loss: {avg_score:.15f}")
        return avg_score

    def _prepare_full_dataset(self, load_cached_data=True, limit=None):
        """
        Loads and concatenates train and validation sets to form the full labeled dataset.
        """
        extractor = FeatureExtractor()

        # Load 'train' split
        print("Loading 'train' split features...")
        dino_train, conv_train, tab_train, ids_train = (
            extractor.extract_and_save_features(
                "train", load_cached_data=load_cached_data, limit=limit
            )
        )
        df_train = load_metadata("train")
        if limit:
            df_train = df_train.head(limit)
        y_train_str = df_train["species"].values

        # Load 'val' split
        print("Loading 'val' split features...")
        dino_val, conv_val, tab_val, ids_val = extractor.extract_and_save_features(
            "val", load_cached_data=load_cached_data, limit=limit
        )
        df_val = load_metadata("val")
        if limit:
            df_val = df_val.head(limit)
        y_val_str = df_val["species"].values

        # Concatenate
        dino_all = np.concatenate([dino_train, dino_val], axis=0)
        conv_all = np.concatenate([conv_train, conv_val], axis=0)
        tab_all = np.concatenate([tab_train, tab_val], axis=0)
        ids_all = np.concatenate([ids_train, ids_val], axis=0)
        y_str_all = np.concatenate([y_train_str, y_val_str], axis=0)

        # Encode Labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_str_all)

        return dino_all, conv_all, tab_all, ids_all, y_encoded, le
