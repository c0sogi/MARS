import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import setup_logging
from library.data_manager import DataManager
from library.model_factory import ModelFactory

# Initialize logger
logger = setup_logging()


class Workflow:
    """
    Orchestrates the training, validation, and submission generation for the
    Hyper-Densified Independent-Component LDA strategy.
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.model_factory = ModelFactory()

    def run_cross_validation(self):
        """
        Executes 10-Fold Stratified Cross-Validation.
        Trains on Hyper-Densified data, validates on Canonical data.
        Saves models for each fold.
        """
        logger.info("Starting Cross-Validation Workflow...")

        # ==========================================
        # 1. Load Data
        # ==========================================
        # Load Densified Training Set (9x samples per image) for Model Fitting
        d_dino, d_conv, d_tab, d_ids, d_labels = (
            self.data_manager.create_densified_training_set()
        )
        X_densified = np.hstack([d_dino, d_conv, d_tab])

        # Load Canonical Training Set (1x sample per image) for Fold Validation
        c_dino, c_conv, c_tab, c_ids, c_labels = (
            self.data_manager.create_canonical_inference_set("train")
        )
        X_canonical = np.hstack([c_dino, c_conv, c_tab])

        # Load Canonical Holdout Set (val.csv) for Final Ensemble Check
        v_dino, v_conv, v_tab, v_ids, v_labels = (
            self.data_manager.create_canonical_inference_set("val")
        )
        X_holdout = np.hstack([v_dino, v_conv, v_tab])

        # ==========================================
        # 2. Stratified K-Fold Loop
        # ==========================================
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []

        # Split based on Canonical IDs (Unique Images)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_canonical, c_labels)):
            logger.info(f"Processing Fold {fold + 1}/{Config.N_FOLDS}")

            # --- A. Prepare Fold Data ---
            # Identify the unique IDs for training and validation in this fold
            train_ids_fold = c_ids[train_idx]

            # Filter Densified Data for Training
            # We select all 9 centroids for every image in the training split
            mask_train = np.isin(d_ids, train_ids_fold)
            X_train_fold = X_densified[mask_train]
            y_train_fold = d_labels[mask_train]

            # Filter Canonical Data for Validation
            # Validation is always done on the single canonical centroid
            X_val_fold = X_canonical[val_idx]
            y_val_fold = c_labels[val_idx]

            # --- B. Train Model ---
            model = self.model_factory.build_lda_pipeline()
            model.fit(X_train_fold, y_train_fold)

            # --- C. Evaluate ---
            probs = model.predict_proba(X_val_fold)
            score = log_loss(y_val_fold, probs, labels=model.classes_)
            fold_scores.append(score)

            logger.info(f"Fold {fold + 1} Log Loss: {score:.15f}")

            # --- D. Save Model ---
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
            joblib.dump(model, model_path)

        # ==========================================
        # 3. Summary & Holdout Evaluation
        # ==========================================
        avg_score = np.mean(fold_scores)
        logger.info(f"Average CV Log Loss: {avg_score:.15f}")

        logger.info("Evaluating Ensemble on Holdout Set (val.csv)...")
        ensemble_probs = self._predict_ensemble(X_holdout)

        # Retrieve classes from the first model to ensure correct label ordering
        model0 = joblib.load(os.path.join(Config.WORKING_DIR, "model_fold_0.joblib"))

        # Cite debug_lesson_1: Handle Disjoint Label Sets When Subsampling High-Cardinality Data
        # Filter validation samples to only those classes present in the training set
        mask_valid = np.isin(v_labels, model0.classes_)

        if np.any(mask_valid):
            holdout_score = log_loss(
                v_labels[mask_valid],
                ensemble_probs[mask_valid],
                labels=model0.classes_,
            )
            logger.info(f"Holdout Set Log Loss: {holdout_score:.15f}")
        else:
            logger.warning(
                "Skipping holdout evaluation: No overlapping classes between training and validation sets (likely due to DEBUG subsampling)."
            )

    def generate_submission(self):
        """
        Generates predictions for the test set using the trained ensemble.
        Saves the result to submission.csv.
        """
        logger.info("Generating Submission...")

        # 1. Load Test Data (Canonical View)
        t_dino, t_conv, t_tab, t_ids, _ = (
            self.data_manager.create_canonical_inference_set("test")
        )
        X_test = np.hstack([t_dino, t_conv, t_tab])

        # 2. Predict with Ensemble
        avg_probs = self._predict_ensemble(X_test)

        # 3. Retrieve Class Names
        model0 = joblib.load(os.path.join(Config.WORKING_DIR, "model_fold_0.joblib"))
        classes = model0.classes_

        # 4. Format Submission
        df_sub = pd.DataFrame(avg_probs, columns=classes)
        df_sub.insert(0, "id", t_ids)

        # 5. Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    def _predict_ensemble(self, X):
        """
        Helper function to run inference using all saved fold models.
        Returns the arithmetic mean of the probability vectors.
        """
        probs_sum = None
        n_models = 0

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
            if not os.path.exists(model_path):
                logger.warning(f"Model for fold {fold} not found. Skipping.")
                continue

            model = joblib.load(model_path)
            probs = model.predict_proba(X)

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs
            n_models += 1

        if n_models == 0:
            raise RuntimeError("No models found for ensemble prediction.")

        return probs_sum / n_models
