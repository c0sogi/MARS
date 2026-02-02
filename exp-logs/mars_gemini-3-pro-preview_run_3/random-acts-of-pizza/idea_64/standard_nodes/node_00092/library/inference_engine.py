import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse

from library.config import Config
from library.utils import Timer, set_seed
from library.data_factory import DataFactory
from library.feature_engine import FeatureGenerator
from library.model_zoo import ModelZoo


class HybridPredictor:
    """
    Manages the inference process for the High-Fidelity Hept-View Stacking Ensemble.
    Implements the Consistent Hybrid Inference Protocol:
    - Volatile Learners: CV-Bagging (Average of N_FOLDS models).
    - Stable Learners: Single model inference (Retrained on full Union Dataset).
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.submission_dir = Config.SUBMISSION_DIR

        # Ensure directories exist (though they should already from training)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Set Global Seed
        set_seed(Config.SEED)

    def _prepare_feature_set(self, feature_key, X_feature, X_meta):
        """
        Concatenates specific modality features with the global metadata vector.
        Replicates the logic from TrainingEngine to ensure feature consistency.

        Args:
            feature_key (str): 'lexical', 'behavioral', 'semantic', or 'metadata'.
            X_feature: The primary feature matrix (sparse or dense).
            X_meta: The dense metadata matrix.

        Returns:
            Combined feature matrix (CSR sparse or Numpy dense).
        """
        if feature_key in ["lexical", "behavioral"]:
            # Sparse Primary + Dense Metadata -> Sparse CSR
            if not scipy.sparse.issparse(X_meta):
                X_meta_sparse = scipy.sparse.csr_matrix(X_meta)
            else:
                X_meta_sparse = X_meta

            # Use hstack for efficient concatenation
            return scipy.sparse.hstack([X_feature, X_meta_sparse], format="csr")

        elif feature_key == "semantic":
            # Dense Primary + Dense Metadata -> Dense Numpy
            return np.hstack([X_feature, X_meta])

        elif feature_key == "metadata":
            # Metadata Only -> Dense Numpy
            return X_meta

        else:
            raise ValueError(f"Unknown feature key: {feature_key}")

    def predict(self):
        """
        Executes the full inference pipeline:
        1. Load Test Data & Generate Features.
        2. Level 1 Inference (Hybrid Volatile/Stable logic).
        3. Level 2 Meta-Learning Inference.
        4. Save Submission.
        """
        with Timer("Full Inference Pipeline"):

            # =========================================================
            # 1. Data Loading & Feature Generation
            # =========================================================
            # We load the union dataset to ensure we have the test_df processed identically
            _, test_df = DataFactory.load_union_dataset(load_cached_data=True)
            request_ids = test_df[Config.ID_COL].values

            # Generate Features
            # We pass train_df as None or dummy if possible, but FeatureGenerator requires both.
            # However, since we are loading cached data, we can pass the loaded test_df.
            # To strictly follow the API, we reload the union train set just to initialize the generator,
            # even though we only need test features.
            train_df_dummy, _ = DataFactory.load_union_dataset(load_cached_data=True)
            fg = FeatureGenerator(train_df_dummy, test_df)

            # Load all feature modalities (expecting them to be cached from training)
            _, X_test_lex = fg.get_lexical_features(load_cached_data=True)
            _, X_test_beh = fg.get_behavioral_features(load_cached_data=True)
            _, X_test_sem = fg.get_semantic_features(load_cached_data=True)
            _, X_test_meta = fg.get_metadata_features(load_cached_data=True)

            # =========================================================
            # 2. Feature Assembly
            # =========================================================
            print("Assembling concatenated feature sets for inference...")

            # Map raw features to keys
            raw_features_test = {
                "lexical": X_test_lex,
                "behavioral": X_test_beh,
                "semantic": X_test_sem,
                "metadata": X_test_meta,
            }

            inputs_test = {}

            for f_key, X_te_raw in raw_features_test.items():
                if f_key == "metadata":
                    inputs_test[f_key] = X_te_raw
                else:
                    inputs_test[f_key] = self._prepare_feature_set(
                        f_key, X_te_raw, X_test_meta
                    )

            # =========================================================
            # 3. Level 1 Ensemble Inference
            # =========================================================
            models_conf = ModelZoo.get_models_dict()
            level1_preds = pd.DataFrame()

            print("\n--- Generating Level 1 Predictions ---")

            for model_name, conf in models_conf.items():
                feature_key = conf["feature_set"]
                X_test = inputs_test[feature_key]

                # Initialize prediction accumulator
                final_pred = np.zeros(X_test.shape[0])

                if conf["type"] == "volatile":
                    # Volatile: Load N_FOLDS models and average predictions (CV-Bagging)
                    print(
                        f"  {model_name} [Volatile]: Averaging {Config.N_FOLDS} folds..."
                    )

                    for fold in range(Config.N_FOLDS):
                        model_path = os.path.join(
                            self.working_dir, f"{model_name}_fold_{fold}.joblib"
                        )

                        if not os.path.exists(model_path):
                            raise FileNotFoundError(
                                f"Model file not found: {model_path}"
                            )

                        model = joblib.load(model_path)
                        fold_pred = model.predict_proba(X_test)[:, 1]
                        final_pred += fold_pred

                    # Average
                    final_pred /= Config.N_FOLDS

                else:
                    # Stable: Load single model retrained on full dataset
                    print(f"  {model_name} [Stable]: Loading full model...")

                    model_path = os.path.join(
                        self.working_dir, f"{model_name}_full.joblib"
                    )

                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model file not found: {model_path}")

                    model = joblib.load(model_path)
                    final_pred = model.predict_proba(X_test)[:, 1]

                # Store in DataFrame
                level1_preds[model_name] = final_pred

            # =========================================================
            # 4. Level 2 Meta-Learner Inference
            # =========================================================
            print("\n--- Generating Level 2 Predictions ---")

            meta_model_path = os.path.join(self.working_dir, "meta_learner.joblib")
            if not os.path.exists(meta_model_path):
                raise FileNotFoundError(
                    f"Meta-learner model not found: {meta_model_path}"
                )

            meta_learner = joblib.load(meta_model_path)

            # Prepare input for meta-learner (Order of columns must match training)
            # The dict iteration order is generally preserved in Python 3.7+, but relying on
            # the DataFrame column order created during training is safer.
            # Ideally, we assume ModelZoo.get_models_dict() returns consistent order.
            X_meta_test = level1_preds.values

            final_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

            # =========================================================
            # 5. Submission Generation
            # =========================================================
            print("Saving submission file...")

            submission = pd.DataFrame(
                {"request_id": request_ids, "requester_received_pizza": final_probs}
            )

            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
            print(f"Submission shape: {submission.shape}")
            print("First 5 predictions:")
            print(submission.head())
