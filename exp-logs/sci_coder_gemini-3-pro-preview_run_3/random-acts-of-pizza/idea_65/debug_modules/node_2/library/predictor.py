import os
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from library.config import (
    MODEL_DIR,
    SUBMISSION_PATH,
    MODEL_KEYS,
    VOLATILE_MODELS,
    STABLE_MODELS,
    N_FOLDS,
    ID_COL,
    TARGET_COL,
)
from library.utils import setup_logger

logger = setup_logger("predictor")


class Predictor:
    """
    Manages the inference phase using the artifacts from the trainer.
    Implements the Hybrid Inference Protocol:
    - Volatile Learners: CV-Bagging (Average of K-Fold models).
    - Stable Learners: Full-Retraining (Single model on full data).
    - Stacking: Meta-Learner aggregation.
    """

    def __init__(self):
        self.model_dir = MODEL_DIR
        self.submission_path = SUBMISSION_PATH
        self.n_folds = N_FOLDS

    def _prepare_input(self, features_dict, model_key):
        """
        Constructs the specific feature matrix for a given model based on its branch.
        Replicates the logic from Trainer._prepare_input to ensure consistency.

        Args:
            features_dict (dict): Dictionary containing 'X_lexical', 'X_community', etc.
            model_key (str): The identifier of the model to determine feature mix.

        Returns:
            scipy.sparse.csr_matrix or np.ndarray: The constructed feature matrix.
        """
        X_lexical = features_dict["X_lexical"]
        X_community = features_dict["X_community"]
        X_semantic = features_dict["X_semantic"]
        X_metadata = features_dict["X_metadata"]

        if "lexical" in model_key:
            # Branch 1: Sparse Lexical + Metadata
            return sp.hstack([X_lexical, X_metadata], format="csr")

        elif "community" in model_key:
            # Branch 2: Sparse Behavioral + Metadata
            return sp.hstack([X_community, X_metadata], format="csr")

        elif "semantic" in model_key:
            # Branch 3: Dense Semantic + Metadata
            return np.hstack([X_semantic, X_metadata])

        elif "metadata" in model_key or "temporal" in model_key:
            # Branch 4: Metadata Only
            return X_metadata

        else:
            raise ValueError(
                f"Could not determine feature mapping for model: {model_key}"
            )

    def _load_model(self, filename):
        """
        Helper to load a model artifact from the model directory.
        """
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return joblib.load(path)

    def predict_ensemble(self, test_features, test_ids):
        """
        Executes the Hybrid Inference Protocol to generate final predictions.

        Args:
            test_features (dict): Dictionary of processed test features.
            test_ids (pd.Series or list): Request IDs for the test set.

        Returns:
            pd.DataFrame: The submission dataframe.
        """
        logger.info("Starting Ensemble Inference...")

        # Determine number of samples
        # We use X_metadata as it is dense and always present
        n_samples = test_features["X_metadata"].shape[0]

        # Initialize Level 1 Prediction Matrix
        # Columns must match MODEL_KEYS order used in Meta-Learner training
        l1_preds = pd.DataFrame(index=np.arange(n_samples), columns=MODEL_KEYS)

        # 1. Generate Base Learner Predictions
        for model_key in MODEL_KEYS:
            logger.info(f"Predicting with {model_key}...")

            # Prepare features for this specific model
            X_test = self._prepare_input(test_features, model_key)

            if model_key in VOLATILE_MODELS:
                # Volatile Protocol: CV-Bagging
                # Load all fold models, predict, and average
                fold_preds = np.zeros((n_samples, self.n_folds))

                for fold in range(self.n_folds):
                    filename = f"{model_key}_fold_{fold}.joblib"
                    try:
                        model = self._load_model(filename)
                        fold_preds[:, fold] = model.predict_proba(X_test)[:, 1]
                    except FileNotFoundError:
                        logger.error(
                            f"Missing fold model {filename}. Ensure training completed successfully."
                        )
                        raise

                # Average predictions across folds
                l1_preds[model_key] = fold_preds.mean(axis=1)

            elif model_key in STABLE_MODELS:
                # Stable Protocol: Single Full-Data Model
                filename = f"{model_key}.joblib"
                try:
                    model = self._load_model(filename)
                    l1_preds[model_key] = model.predict_proba(X_test)[:, 1]
                except FileNotFoundError:
                    logger.error(
                        f"Missing stable model {filename}. Ensure training completed successfully."
                    )
                    raise

            else:
                logger.warning(
                    f"Model {model_key} is not categorized as Volatile or Stable. Skipping."
                )

        # 2. Generate Meta-Learner Predictions
        logger.info("Generating Final Predictions with Meta-Learner...")
        try:
            meta_learner = self._load_model("meta_learner.joblib")
            # The meta-learner expects the input matrix columns to correspond to MODEL_KEYS
            final_probs = meta_learner.predict_proba(l1_preds.values)[:, 1]
        except FileNotFoundError:
            logger.error(
                "Meta-learner model not found. Ensure training completed successfully."
            )
            raise

        # 3. Create and Save Submission
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_probs})

        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        submission.to_csv(self.submission_path, index=False)

        logger.info(f"Submission saved to {self.submission_path}")

        return submission
