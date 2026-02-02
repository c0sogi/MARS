import os
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from library.config import Config
from library.feature_engineering import FeaturePipeline


class InferenceEngine:
    """
    Handles the inference phase for the Hex-View Stacking Ensemble.

    This engine:
    1. Loads pre-computed test features via the FeaturePipeline.
    2. Reconstructs the specific input matrices (modalities) required by each base learner.
    3. Performs CV-Bagging Inference:
       - For each base learner type, loads all 5 saved fold-models.
       - Generates predictions for the test set from each fold-model.
       - Averages these predictions (bagging) to ensure robust estimates.
    4. Stacks the bagged Level 1 predictions.
    5. Loads the trained Meta-Learner and generates the final probability.
    6. Saves the submission file.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the Inference Engine.

        Args:
            load_cached_data (bool): Whether to load features from cache.
                                     Defaults to True as features should be generated during training.
        """
        self.feature_pipeline = FeaturePipeline(load_cached_data=load_cached_data)
        self.models_dir = Config.MODEL_DIR
        self.submission_path = Config.SUBMISSION_PATH

        # Define the list of base learners and their required feature modalities
        # Must match the order and structure used in TrainingEngine
        self.learners = [
            ("lexical_bagger", "lexical_meta"),
            ("community_bagger", "community_meta"),
            ("semantic_booster", "semantic_meta"),
            ("semantic_bagger", "semantic_meta"),
            ("temporal_booster", "meta_only"),
            ("metadata_anchor", "meta_only"),
        ]

    def _prepare_test_inputs(self, data):
        """
        Constructs the modality-specific input matrices for the test set.
        Mirroring the logic in TrainingEngine._prepare_data.

        Args:
            data (dict): Dictionary containing all feature arrays from FeaturePipeline.

        Returns:
            dict: Map of input_type -> X_test matrix
        """
        inputs = {}

        # Extract base components
        X_test_lexical = data["X_test_lexical"]
        X_test_community = data["X_test_community"]
        X_test_semantic = data["X_test_semantic"]
        X_test_meta = data["X_test_meta"]

        # 1. Lexical + Meta (Sparse)
        # Meta is dense, convert to sparse for hstack
        inputs["lexical_meta"] = sparse.hstack(
            [X_test_lexical, sparse.csr_matrix(X_test_meta)]
        )

        # 2. Community + Meta (Sparse)
        inputs["community_meta"] = sparse.hstack(
            [X_test_community, sparse.csr_matrix(X_test_meta)]
        )

        # 3. Semantic + Meta (Dense)
        inputs["semantic_meta"] = np.hstack([X_test_semantic, X_test_meta])

        # 4. Meta Only (Dense)
        inputs["meta_only"] = X_test_meta

        return inputs

    def run_inference(self, n_folds=5):
        """
        Executes the inference pipeline.

        Args:
            n_folds (int): Number of folds used during training.
                           Determines how many models to load per learner.
        """
        print("Starting Inference Pipeline...")

        # 1. Load Features
        print("Loading test features...")
        data = self.feature_pipeline.get_all_features()
        test_ids = data["test_ids"]

        # 2. Prepare Inputs
        inputs_map = self._prepare_test_inputs(data)
        n_test = len(test_ids)
        n_learners = len(self.learners)

        # Matrix to hold the bagged predictions for Level 2
        # Shape: [n_test_samples, n_base_learners]
        l1_test_preds = np.zeros((n_test, n_learners))

        # 3. Level 1 Inference (Bagging)
        print(f"Generating predictions using {n_folds}-fold CV Bagging...")

        for learner_idx, (learner_name, input_type) in enumerate(self.learners):
            print(f"Processing {learner_name}...", end=" ")

            # Get specific input data
            X_test = inputs_map[input_type]

            # Accumulator for bagging
            fold_preds_sum = np.zeros(n_test)

            # Load and predict with each fold model
            for fold in range(n_folds):
                model_filename = f"{learner_name}_fold_{fold}.joblib"
                model_path = os.path.join(self.models_dir, model_filename)

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                model = joblib.load(model_path)

                # Predict probability of positive class
                preds = model.predict_proba(X_test)[:, 1]
                fold_preds_sum += preds

            # Average predictions
            avg_preds = fold_preds_sum / n_folds
            l1_test_preds[:, learner_idx] = avg_preds
            print("Done.")

        # 4. Level 2 Inference (Stacking)
        print("Loading Meta-Learner...")
        meta_model_path = os.path.join(self.models_dir, "meta_learner.joblib")

        if not os.path.exists(meta_model_path):
            raise FileNotFoundError(f"Meta-learner model not found: {meta_model_path}")

        meta_learner = joblib.load(meta_model_path)

        print("Generating final probabilities...")
        final_probs = meta_learner.predict_proba(l1_test_preds)[:, 1]

        # 5. Save Submission
        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_probs}
        )

        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        submission.to_csv(self.submission_path, index=False)

        print(f"Inference complete. Submission saved to {self.submission_path}")
        print("Sample predictions:")
        print(submission.head())
