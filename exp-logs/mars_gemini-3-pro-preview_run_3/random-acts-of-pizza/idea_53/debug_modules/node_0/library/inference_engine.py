import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse

from library.config import (
    MODEL_DIR,
    N_FOLDS,
    MODEL_TYPES,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
)
from library.model_factory import get_base_models
from library.data_loader import load_dataset
from library.feature_extraction import FeaturePipeline


class HybridPredictor:
    """
    Inference engine for the Oct-View Stacking Ensemble.
    Replicates the feature stacking logic of the trainer and applies the
    Hybrid Inference Protocol (Single model for Stable, CV-Bagging for Volatile).
    """

    def __init__(self, n_folds=N_FOLDS):
        self.n_folds = n_folds

        # Define the feature mapping exactly as in the Trainer
        self.feature_map = {
            "lexical_bagger": ["lexical", "meta"],
            "lexical_anchor": ["lexical", "meta"],
            "community_bagger": ["community", "meta"],
            "semantic_booster": ["semantic", "meta"],
            "semantic_gradient": ["semantic", "meta"],
            "semantic_bagger": ["semantic", "meta"],
            "metadata_anchor": ["meta"],
            "temporal_booster": ["meta"],
        }

        # We retrieve the model names from the factory to ensure
        # the column order for the meta-learner matches training exactly.
        # We don't need the model instances, just the keys in order.
        self.model_names = list(get_base_models().keys())

    def _get_model_features(
        self, model_name, X_lexical, X_community, X_semantic, X_meta
    ):
        """
        Constructs the specific feature matrix for a given model by stacking components.
        Mirrors logic in library.training_engine.HybridTrainer.
        """
        components = self.feature_map.get(model_name)
        if not components:
            raise ValueError(f"Unknown model name: {model_name}")

        features_to_stack = []
        is_sparse = False

        for comp in components:
            if comp == "lexical":
                features_to_stack.append(X_lexical)
                is_sparse = True
            elif comp == "community":
                features_to_stack.append(X_community)
                is_sparse = True
            elif comp == "semantic":
                features_to_stack.append(X_semantic)
            elif comp == "meta":
                features_to_stack.append(X_meta)

        if len(features_to_stack) == 1:
            return features_to_stack[0]

        # Stack appropriately
        if is_sparse:
            return scipy.sparse.hstack(features_to_stack).tocsr()
        else:
            return np.hstack(features_to_stack)

    def predict(self, df_test, feature_pipeline):
        """
        Generates final predictions for the test set.

        Args:
            df_test: processed test DataFrame.
            feature_pipeline: fitted FeaturePipeline instance.

        Returns:
            pd.DataFrame: DataFrame with 'request_id' and 'requester_received_pizza'.
        """
        print("Transforming test features...")
        # Transform features using the pipeline (uses cache if available)
        X_lexical, X_community, X_semantic, X_meta = feature_pipeline.transform(
            df_test, split="test"
        )

        # Container for Level 1 predictions
        # Rows: Samples, Cols: Base Models
        l1_preds = pd.DataFrame(index=df_test.index)

        print("Generating Level 1 predictions...")
        for model_name in self.model_names:
            model_type = MODEL_TYPES.get(model_name, "stable")

            # Construct features for this specific model
            X_model = self._get_model_features(
                model_name, X_lexical, X_community, X_semantic, X_meta
            )

            if model_type == "stable":
                # Stable models: Load single fully-retrained model
                model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                model = joblib.load(model_path)
                preds = model.predict_proba(X_model)[:, 1]

            else:
                # Volatile models: CV-Bagging (Average of N_FOLDS models)
                fold_preds = []
                for fold in range(self.n_folds):
                    model_path = os.path.join(
                        MODEL_DIR, f"{model_name}_fold_{fold}.joblib"
                    )
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"Fold model file not found: {model_path}"
                        )

                    model = joblib.load(model_path)
                    p = model.predict_proba(X_model)[:, 1]
                    fold_preds.append(p)

                # Average predictions across folds
                preds = np.mean(fold_preds, axis=0)

            l1_preds[model_name] = preds

        print("Generating Level 2 predictions (Meta-Learner)...")
        # Load Meta-Learner
        meta_path = os.path.join(MODEL_DIR, "meta_learner.joblib")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta-learner not found: {meta_path}")

        meta_learner = joblib.load(meta_path)

        # Predict final probabilities
        # Ensure input is the DataFrame to match training OOF structure
        final_probs = meta_learner.predict_proba(l1_preds)[:, 1]

        # Construct submission DataFrame
        submission_df = pd.DataFrame({ID_COL: df_test[ID_COL], TARGET_COL: final_probs})

        return submission_df


def generate_submission(load_cached_data=True):
    """
    Main entry point to generate the submission file.
    """
    # 1. Load Test Data
    print("Loading test dataset...")
    df_test = load_dataset("test", load_cached_data=load_cached_data)

    # 2. Initialize Feature Pipeline
    # We must instantiate it to load the vectorizers/scalers fitted during training.
    # Note: The pipeline logic in library.feature_extraction handles loading fitted state
    # implicitly via caching or we assume the training script has run and populated the cache/state.
    # However, standard sklearn objects (vectorizers) need to be fitted.
    # Since we are in inference mode, we assume the pipeline's 'fit_transform' was called on train
    # during the training phase. Ideally, the FeaturePipeline would pickle its internal transformers.
    # Given the provided code in library/feature_extraction.py, the vectorizers are re-fit on
    # the dataframe passed to fit_transform.
    # CRITICAL: To ensure consistency, we must load the TRAINING data first to fit the pipeline
    # exactly as it was done during training, OR rely on the fact that the provided FeaturePipeline
    # implementation in the prompt re-fits on whatever is passed to fit_transform.
    #
    # Looking at library/feature_extraction.py:
    # fit_transform fits on the provided df.
    # transform uses the fitted attributes.
    #
    # Therefore, to correctly transform test data, we must first "prime" the pipeline
    # by fitting it on the training data (or loading a pickled pipeline, but we don't have that).
    # We will load train data to fit the pipeline.

    print("Initializing and fitting feature pipeline on training data...")
    df_train = load_dataset("train", load_cached_data=load_cached_data)
    feature_pipeline = FeaturePipeline(load_cached_data=load_cached_data)

    # Fit on train (this ensures vectorizers match the trained models)
    # This also triggers cache loading/creation for train features
    feature_pipeline.fit_transform(df_train)

    # 3. Run Inference
    predictor = HybridPredictor()
    submission_df = predictor.predict(df_test, feature_pipeline)

    # 4. Save Submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
