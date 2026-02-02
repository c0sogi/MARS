import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from library.config import Config
from library.utils import get_logger, timer
from library.feature_engineering import FeaturePipeline
from library.model_definitions import get_base_learners


class BaggingInference:
    """
    Implements the inference phase for the Restored-History Hex-View Stacking Ensemble.
    Uses CV-Bagging (averaging predictions across 5 fold-models) for base learners
    and a Meta-Learner for final calibration.
    """

    def __init__(self):
        self.logger = get_logger("BaggingInference")
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.pipeline = FeaturePipeline()
        self.base_learners_factories = get_base_learners()
        self.model_names = list(self.base_learners_factories.keys())

    def _get_model_features(self, X_dict, model_name):
        """
        Retrieves and concatenates the specific feature views required for a given model.
        Replicates the logic from CVEnsembleTrainer to ensure consistency during inference.

        Args:
            X_dict (dict): Dictionary of feature matrices (test).
            model_name (str): Name of the model to determine feature composition.

        Returns:
            np.ndarray or scipy.sparse.csr_matrix: The combined feature matrix.
        """
        # 1. Lexical Bagger: Sparse Lexical + Dense Contextual
        if model_name == "lexical_bagger":
            return sp.hstack([X_dict["lexical"], sp.csr_matrix(X_dict["contextual"])])

        # 2. Community Bagger: Sparse Behavioral + Dense Contextual
        elif model_name == "community_bagger":
            return sp.hstack(
                [X_dict["behavioral"], sp.csr_matrix(X_dict["contextual"])]
            )

        # 3. Semantic Models: Dense Semantic + Dense Contextual
        elif model_name in ["semantic_booster", "semantic_bagger"]:
            return np.hstack([X_dict["semantic"], X_dict["contextual"]])

        # 4. Contextual Models: Dense Contextual Only
        elif model_name in ["metadata_anchor", "temporal_booster"]:
            return X_dict["contextual"]

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set using the ensemble of saved models.

        Args:
            load_cached_data (bool): Whether to load features from cache. Defaults to True.
        """
        with timer("Inference Pipeline"):
            # 1. Retrieve Test Data
            self.logger.info("Loading test data features...")
            # The pipeline returns (X_train, y_train, X_test, test_ids)
            # We only need the test components here.
            _, _, X_test_dict, test_ids = self.pipeline.get_data(
                load_cached_data=load_cached_data
            )

            # 2. Generate Bagged Predictions for Base Learners
            self.logger.info("Generating bagged predictions for base learners...")

            # DataFrame to hold Level 1 predictions (Meta-Features)
            # Shape: (n_test_samples, n_base_learners)
            test_meta_features = pd.DataFrame(
                index=range(len(test_ids)), columns=self.model_names
            )

            for model_name in self.model_names:
                self.logger.info(f"Processing base learner: {model_name}")

                # Construct the specific feature set for this model
                X_test = self._get_model_features(X_test_dict, model_name)

                fold_preds = []
                # Iterate through all 5 folds to perform bagging
                for fold in range(Config.N_FOLDS):
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{fold}.joblib"
                    )

                    if not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"Model artifact not found: {model_path}. "
                            "Ensure training is complete."
                        )

                    model = joblib.load(model_path)

                    # Generate probabilities
                    if hasattr(model, "predict_proba"):
                        # Binary classification, take probability of positive class
                        p_test = model.predict_proba(X_test)[:, 1]
                    else:
                        # Fallback (though all defined models should support proba)
                        p_test = model.predict(X_test)

                    fold_preds.append(p_test)

                # Average across folds (Bagging)
                avg_pred = np.mean(fold_preds, axis=0)
                test_meta_features[model_name] = avg_pred

            # 3. Meta-Learner Prediction
            self.logger.info("Generating final predictions with Meta-Learner...")
            meta_model_path = os.path.join(self.models_dir, "meta_learner.joblib")

            if not os.path.exists(meta_model_path):
                raise FileNotFoundError(
                    f"Meta-learner model not found: {meta_model_path}"
                )

            meta_model = joblib.load(meta_model_path)

            # Ensure the order of columns matches the training order
            X_meta_test = test_meta_features[self.model_names].values
            final_preds = meta_model.predict_proba(X_meta_test)[:, 1]

            # 4. Save Submission
            submission = pd.DataFrame(
                {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds}
            )

            # Ensure submission directory exists
            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
            self.logger.info(f"Submission shape: {submission.shape}")
            self.logger.info(f"Head:\n{submission.head()}")
