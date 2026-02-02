import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config
from library.densification import get_densified_data
from library.custom_transformers import DualStreamPreprocessor


class Trainer:
    """
    Manages the training of the Convex-Hull Densified Selective-Topology LDA ensemble.
    Implements Stratified K-Fold Cross-Validation on the densified dataset.
    """

    def __init__(self):
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED

    def train_ensemble(self, load_cached_data=True):
        """
        Executes the Stratified K-Fold Cross-Validation training pipeline.

        Args:
            load_cached_data (bool): Whether to load features/densified data from cache.
                                     If False, re-computes features and densification.
        """
        print("Initializing training pipeline...")

        # 1. Load Densified Training Data
        # This returns the 6x expanded dataset (3 primary + 3 interpolated centroids per image)
        ids, X_dino, X_conv, X_tab, y = get_densified_data(
            split="train", load_cached_data=load_cached_data
        )

        if y is None:
            raise ValueError("Training data must have labels.")

        # 2. Concatenate Features for the Preprocessor
        # The DualStreamPreprocessor expects [DINO, ConvNeXt, Tabular]
        print(
            f"Data shapes - DINO: {X_dino.shape}, ConvNeXt: {X_conv.shape}, Tabular: {X_tab.shape}"
        )
        X = np.concatenate([X_dino, X_conv, X_tab], axis=1)

        # Determine dimensions for the preprocessor
        dino_dim = X_dino.shape[1]
        conv_dim = X_conv.shape[1]
        tab_dim = X_tab.shape[1]

        # 3. Prepare for Stratified Splitting
        # We must split based on unique Image IDs to prevent data leakage.
        # Since the data is densified (multiple rows per ID), we extract unique IDs and their corresponding labels.
        unique_ids, unique_indices = np.unique(ids, return_index=True)
        unique_labels = y[unique_indices]

        print(f"Unique training samples: {len(unique_ids)}")
        print(f"Total densified samples: {len(ids)}")

        # 4. K-Fold Cross Validation
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        fold_metrics = []

        print(f"Starting {self.n_folds}-Fold Cross-Validation...")

        # We iterate over the unique IDs to ensure all views of an image stay in the same fold
        for fold, (train_idx_unique, val_idx_unique) in enumerate(
            skf.split(unique_ids, unique_labels)
        ):
            print(f"\n--- Fold {fold} ---")

            # Get the actual IDs for this fold
            train_ids = set(unique_ids[train_idx_unique])
            val_ids = set(unique_ids[val_idx_unique])

            # Create boolean masks for the full densified dataset
            # np.isin is efficient for this
            train_mask = np.isin(ids, list(train_ids))
            val_mask = np.isin(ids, list(val_ids))

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # Construct the Pipeline
            # 1. DualStreamPreprocessor: PCA on visual, Quantile on tabular, Global Scaling
            # 2. LDA: Linear Discriminant Analysis with Ledoit-Wolf shrinkage
            pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        DualStreamPreprocessor(
                            pca_variance=Config.PCA_VARIANCE,
                            dino_dim=dino_dim,
                            conv_dim=conv_dim,
                            tab_dim=tab_dim,
                        ),
                    ),
                    (
                        "classifier",
                        LinearDiscriminantAnalysis(
                            solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
                        ),
                    ),
                ]
            )

            # Train
            print("Fitting model...")
            pipeline.fit(X_train, y_train)

            # Evaluate
            print("Evaluating...")
            # Predict probabilities
            y_pred_proba = pipeline.predict_proba(X_val)
            y_pred = pipeline.predict(X_val)

            # Calculate Metrics
            # Passing pipeline.classes_ ensures correct mapping for log_loss
            loss = log_loss(y_val, y_pred_proba, labels=pipeline.classes_)
            acc = accuracy_score(y_val, y_pred)

            print(f"Fold {fold} Results:")
            print(f"  Log Loss: {loss}")
            print(f"  Accuracy: {acc}")

            fold_metrics.append({"fold": fold, "log_loss": loss, "accuracy": acc})

            # Save the pipeline
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            joblib.dump(pipeline, model_path)
            print(f"Model saved to {model_path}")

        # 5. Summary
        avg_loss = np.mean([m["log_loss"] for m in fold_metrics])
        avg_acc = np.mean([m["accuracy"] for m in fold_metrics])

        print("\n=== Training Summary ===")
        print(f"Average Log Loss: {avg_loss}")
        print(f"Average Accuracy: {avg_acc}")

        # Save classes for inference
        # We save the classes from the last fold's model. Since stratification covers all classes,
        # this is safe.
        classes_path = os.path.join(self.models_dir, "classes.pkl")
        joblib.dump(pipeline.classes_, classes_path)
        print(f"Class labels saved to {classes_path}")


def train_ensemble(load_cached_data=True):
    """
    Wrapper function to instantiate Trainer and run the training process.
    """
    trainer = Trainer()
    trainer.train_ensemble(load_cached_data=load_cached_data)
