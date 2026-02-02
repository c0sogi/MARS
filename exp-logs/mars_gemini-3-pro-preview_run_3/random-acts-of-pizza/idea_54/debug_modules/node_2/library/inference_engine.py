import os
import joblib
import numpy as np
import pandas as pd
from library.config import CACHE_DIR, SUBMISSION_DIR, N_FOLDS
from library.feature_engineering import FeatureFactory
from library.model_factory import get_base_learners


class HybridPredictor:
    """
    Implements the Hybrid Inference Protocol for the Oct-View Stacking Ensemble.
    Generates predictions for the test set using the trained models.
    """

    def __init__(self, feature_factory: FeatureFactory):
        self.feature_factory = feature_factory
        self.models_dir = os.path.join(CACHE_DIR, "models")

        # Mapping models to their required feature subsets
        # Must match the mapping used in HybridTrainer
        self.feature_map = {
            "lexical_bagger": ["lexical", "metadata"],
            "lexical_anchor": ["lexical", "metadata"],
            "community_bagger": ["behavioral", "metadata"],
            "semantic_booster": ["semantic", "metadata"],
            "semantic_gradient": ["semantic", "metadata"],
            "semantic_bagger": ["semantic", "metadata"],
            "metadata_anchor": ["metadata"],
            "temporal_booster": ["metadata"],
        }

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Generates final probabilities for the test set.

        Args:
            df_test (pd.DataFrame): The test dataframe.

        Returns:
            np.ndarray: Final probability scores (Class 1).
        """
        print("\n=== Starting Inference ===")

        # 1. Generate Features
        # FeatureFactory handles caching internally
        print("Generating test features...")
        feature_cache = self.feature_factory.transform(df_test, "test", load_cache=True)

        # 2. Level 1 Inference
        # We use get_base_learners() to ensure we iterate in the exact same order
        # as the training process, guaranteeing the columns align for the meta-learner.
        base_learners_ref = get_base_learners()
        level_1_preds = pd.DataFrame(index=df_test.index)

        print("Generating Level 1 predictions...")
        for model_name in base_learners_ref.keys():
            # Prepare data
            required_keys = self.feature_map[model_name]
            X_test = self.feature_factory.combine_features(feature_cache, required_keys)

            # Determine Inference Strategy based on file existence
            # Stable learners have a single "_full" model.
            # Volatile learners have multiple "_fold_{i}" models.
            full_model_path = os.path.join(self.models_dir, f"{model_name}_full.joblib")

            if os.path.exists(full_model_path):
                # --- Stable Learner Strategy ---
                # Load single fully-retrained model
                # print(f"  Predicting with {model_name} (Stable/Full)...")
                model = joblib.load(full_model_path)
                preds = model.predict_proba(X_test)[:, 1]
            else:
                # --- Volatile Learner Strategy ---
                # Load all fold models and average (CV-Bagging)
                # print(f"  Predicting with {model_name} (Volatile/Bagged)...")
                fold_preds = []
                for i in range(N_FOLDS):
                    fold_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{i}.joblib"
                    )
                    if not os.path.exists(fold_path):
                        raise FileNotFoundError(f"Model file missing: {fold_path}")

                    model = joblib.load(fold_path)
                    fold_preds.append(model.predict_proba(X_test)[:, 1])

                # Average predictions
                preds = np.mean(fold_preds, axis=0)

            level_1_preds[model_name] = preds

        # 3. Level 2 Inference (Meta-Learner)
        print("Generating Level 2 predictions (Meta-Learner)...")
        meta_learner_path = os.path.join(self.models_dir, "meta_learner.joblib")
        if not os.path.exists(meta_learner_path):
            raise FileNotFoundError(f"Meta-learner model missing: {meta_learner_path}")

        meta_learner = joblib.load(meta_learner_path)

        # Ensure input shape matches
        X_meta = level_1_preds.values
        final_probs = meta_learner.predict_proba(X_meta)[:, 1]

        return final_probs

    def generate_submission(self, df_test: pd.DataFrame):
        """
        Generates predictions and saves the submission file.
        """
        # Get predictions
        probs = self.predict(df_test)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"request_id": df_test["request_id"], "requester_received_pizza": probs}
        )

        # Save
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission.to_csv(save_path, index=False)
        print(f"\nSubmission saved to {save_path}")
        print(submission.head())
