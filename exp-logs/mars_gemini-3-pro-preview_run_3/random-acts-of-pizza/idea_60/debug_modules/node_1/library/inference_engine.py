import os
import numpy as np
import pandas as pd
import scipy.sparse
import joblib

from library.config import Config
from library.utils import load_artifact


class EnsemblePredictor:
    """
    Manages the inference lifecycle of the Granular Hept-View Stacking Ensemble.
    Implements the Robust Hybrid Inference Protocol:
    1. Loads 5-Fold models for Volatile Learners (XGB/LGBM) and averages predictions.
    2. Loads Full-Retrain models for Stable Learners (RF/Linear).
    3. Stacks predictions via the Meta-Learner.
    """

    def __init__(self, data_dict: dict):
        """
        Args:
            data_dict (dict): Dictionary containing processed feature matrices.
                              Must contain keys like 'X_meta_test', 'X_lex_test', etc.
        """
        self.data = data_dict
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")

        # Configuration must match Training Engine exactly
        self.learner_config = {
            "lexical_bagger": {
                "type": "stable",
                "branch": "lexical",
            },
            "community_bagger": {
                "type": "stable",
                "branch": "behavioral",
            },
            "semantic_booster": {
                "type": "volatile",
                "branch": "semantic",
            },
            "semantic_gradient": {
                "type": "volatile",
                "branch": "semantic",
            },
            "semantic_bagger": {
                "type": "stable",
                "branch": "semantic",
            },
            "metadata_anchor": {
                "type": "stable",
                "branch": "metadata",
            },
            "temporal_booster": {
                "type": "volatile",
                "branch": "metadata",
            },
        }

    def _get_test_features(self, branch: str):
        """
        Constructs the feature matrix for the test set for a specific branch.
        Concatenates the specific view with the Global Metadata.

        Args:
            branch (str): The modality branch ('lexical', 'behavioral', 'semantic', 'metadata').

        Returns:
            np.ndarray or scipy.sparse.csr_matrix: The combined feature matrix.
        """
        # Get Global Metadata
        X_meta = self.data["X_meta_test"]

        # If branch is metadata, that's all we need
        if branch == "metadata":
            return X_meta

        # Map branch to specific view key
        view_key_map = {
            "lexical": "X_lex_test",
            "behavioral": "X_beh_test",
            "semantic": "X_sem_test",
        }

        if branch not in view_key_map:
            raise ValueError(f"Unknown branch: {branch}")

        view_key = view_key_map[branch]
        X_view = self.data[view_key]

        # Concatenate (Stacking Sparse + Dense if necessary)
        # We use the metadata (dense) as context for all sparse views
        if scipy.sparse.issparse(X_view):
            return scipy.sparse.hstack([X_view, X_meta], format="csr")
        else:
            return np.hstack([X_view, X_meta])

    def _load_model(self, name: str):
        """Loads a serialized model from the models directory."""
        path = os.path.join(self.models_dir, f"{name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return joblib.load(path)

    def predict(self) -> np.ndarray:
        """
        Executes the inference pipeline.

        Returns:
            np.ndarray: Final probability predictions (Level 2 output).
        """
        print("\n--- Starting Hybrid Inference ---")

        # Initialize Level 1 Predictions DataFrame
        n_samples = self.data["X_meta_test"].shape[0]
        level1_preds = pd.DataFrame(
            index=range(n_samples), columns=self.learner_config.keys()
        )

        # Generate Level 1 Predictions
        for name, config in self.learner_config.items():
            print(f"Predicting with {name} ({config['type']})...", end=" ")

            # Construct Features
            X_test = self._get_test_features(config["branch"])

            if config["type"] == "volatile":
                # Hybrid Inference: Average of 5 fold models (CV-Bagging)
                fold_preds = []
                for fold in range(Config.N_FOLDS):
                    model_name = f"{name}_fold_{fold}"
                    try:
                        model = self._load_model(model_name)
                        # Predict proba for class 1
                        preds = model.predict_proba(X_test)[:, 1]
                        fold_preds.append(preds)
                    except FileNotFoundError:
                        print(f"\nWarning: {model_name} not found. Skipping fold.")

                if not fold_preds:
                    raise RuntimeError(f"No models found for volatile learner {name}")

                # Average predictions
                level1_preds[name] = np.mean(fold_preds, axis=0)
                print(f"Averaged {len(fold_preds)} folds.")

            else:  # stable
                # Hybrid Inference: Use single full model
                model_name = f"{name}_full"
                model = self._load_model(model_name)
                level1_preds[name] = model.predict_proba(X_test)[:, 1]
                print("Used full model.")

        # Generate Level 2 Predictions (Stacking)
        print("Predicting with Meta-Learner...", end=" ")
        meta_learner = self._load_model("meta_learner")
        final_probs = meta_learner.predict_proba(level1_preds.values)[:, 1]
        print("Done.")

        return final_probs

    def generate_submission(self):
        """
        Orchestrates prediction and saves the submission file.
        """
        # 1. Get Predictions
        final_probs = self.predict()

        # 2. Load Test IDs
        # We load the processed test parquet to get IDs aligned with the feature matrix
        try:
            test_meta_df = load_artifact(Config.TEST_PATH)
        except FileNotFoundError:
            # Fallback if metadata isn't directly available, though Config guarantees it
            print("Warning: Could not load test metadata for IDs. Checking cache...")
            test_meta_df = load_artifact(
                os.path.join(Config.WORKING_DIR, "test_processed.parquet")
            )

        # 3. Create Submission DataFrame
        submission_df = pd.DataFrame(
            {Config.ID_COL: test_meta_df[Config.ID_COL], Config.TARGET_COL: final_probs}
        )

        # 4. Save
        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)

        print(f"\nSubmission saved to {submission_path}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Head:\n{submission_df.head()}")
