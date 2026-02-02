import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_manager import DensifiedDataLoader
from library.model_pipeline import HierarchicalLDA


class KFoldTrainer:
    """
    Orchestrates the Stratified K-Fold Cross-Validation training and inference
    using Hierarchical Discriminant Stacking with Hyper-Densified Orthogonal Centroids.
    """

    def __init__(self):
        """
        Initialize the trainer with data loader and storage for class names.
        """
        self.loader = DensifiedDataLoader()
        self.class_names = None

    def train_kfold_ensemble(self, load_cached_data=True):
        """
        Executes the Stratified K-Fold training loop.

        Loads hyper-densified data for training and canonical data for validation.
        Trains a HierarchicalLDA model for each fold, evaluates it, and saves the pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load features from disk cache.
        """
        seed_everything()
        print("Loading training data...")

        # 1. Load Data
        # Train Data: Hyper-Densified (9 centroids per image)
        # Used for fitting the models to maximize sample size and stability.
        train_densified = self.loader.generate_train_data(
            load_cached_data=load_cached_data
        )

        # Validation Data: Canonical (1 centroid per image)
        # We generate canonical features for the training set to use as validation data.
        # This matches the inference topology.
        train_canonical = self.loader.generate_inference_data(
            dataset_name="train_canonical",
            csv_path=Config.TRAIN_CSV,
            load_cached_data=load_cached_data,
        )

        # Extract unique IDs and Labels for Stratified Split
        # train_canonical has exactly one row per image, perfect for defining splits.
        unique_ids = train_canonical["ids"]
        unique_y = train_canonical["y"]

        # 2. Initialize K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []

        print(f"Starting Stratified K-Fold CV (K={Config.N_FOLDS})...")

        # 3. Training Loop
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(unique_y)), unique_y)
        ):
            print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            # Identify IDs for this fold
            train_ids_fold = unique_ids[train_idx]
            val_ids_fold = unique_ids[val_idx]

            # --- Prepare Training Set (Densified) ---
            # Select all 9 centroids for each image in the training split
            train_mask = np.isin(train_densified["ids"], train_ids_fold)

            X_dino_train = train_densified["dino"][train_mask]
            X_conv_train = train_densified["convnext"][train_mask]
            X_tab_train = train_densified["tabular"][train_mask]
            y_train = train_densified["y"][train_mask]

            # --- Prepare Validation Set (Canonical) ---
            # Select the single canonical centroid for each image in the validation split
            val_mask = np.isin(train_canonical["ids"], val_ids_fold)

            X_dino_val = train_canonical["dino"][val_mask]
            X_conv_val = train_canonical["convnext"][val_mask]
            X_tab_val = train_canonical["tabular"][val_mask]
            y_val = train_canonical["y"][val_mask]

            # --- Fit Model ---
            model = HierarchicalLDA()
            model.fit(X_dino_train, X_conv_train, X_tab_train, y_train)

            # Store class names from the first model (assuming consistency across folds)
            if self.class_names is None:
                self.class_names = model.classes_

            # --- Evaluate ---
            # Predict probabilities on validation set
            probs_val = model.predict_proba(X_dino_val, X_conv_val, X_tab_val)

            # Compute Log Loss
            # We pass model.classes_ to ensure correct label mapping even if a class is missing in val
            score = log_loss(y_val, probs_val, labels=model.classes_)
            fold_scores.append(score)

            print(f"Fold {fold + 1} Log Loss: {score:.15f}")

            # --- Save Model ---
            model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # 4. Summary
        print("\n==============================")
        print(f"Average CV Log Loss: {np.mean(fold_scores):.15f}")
        print(f"Std CV Log Loss:     {np.std(fold_scores):.15f}")
        print("==============================")

    def generate_submission(self, load_cached_data=True):
        """
        Generates predictions for the test set using the ensemble of trained models.
        Averages the probabilities from all K folds and saves the submission file.

        Args:
            load_cached_data (bool): If True, attempts to load features from disk cache.
        """
        seed_everything()
        print("\nGenerating submission for test set...")

        # 1. Load Test Data (Canonical)
        test_data = self.loader.generate_inference_data(
            dataset_name="test",
            csv_path=Config.TEST_CSV,
            load_cached_data=load_cached_data,
        )

        X_dino_test = test_data["dino"]
        X_conv_test = test_data["convnext"]
        X_tab_test = test_data["tabular"]
        test_ids = test_data["ids"]

        # 2. Ensemble Inference
        avg_probs = None
        models_found = 0

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold}.pkl")

            if not os.path.exists(model_path):
                print(f"Warning: Model for fold {fold} not found. Skipping.")
                continue

            # Load model
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Predict
            probs = model.predict_proba(X_dino_test, X_conv_test, X_tab_test)

            # Accumulate
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs += probs

            models_found += 1

        if models_found == 0:
            raise RuntimeError("No trained models found. Cannot generate submission.")

        # 3. Average Probabilities
        avg_probs /= models_found

        # 4. Save Submission
        if self.class_names is None:
            # Fallback if class names weren't set during training (e.g. if training was skipped)
            # Try to retrieve from the last loaded model
            self.class_names = model.classes_

        save_submission(test_ids, avg_probs, self.class_names)
