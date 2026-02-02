import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.metrics import roc_auc_score
import warnings

# Import provided library modules
from library.config import Config
from library.feature_engineering import FeaturePipeline
from library.training_engine import TrainingEngine
from library.inference_engine import InferenceEngine


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    warnings.filterwarnings("ignore")
    np.random.seed(Config.SEED)

    print("Initializing Fast Baseline Run...")

    # Patch Config for speed (Fast Baseline) and GPU acceleration
    # Reducing estimators to ensure completion within strict time limits
    Config.PARAMS_LEXICAL_BAGGER["n_estimators"] = 100
    Config.PARAMS_COMMUNITY_BAGGER["n_estimators"] = 100
    Config.PARAMS_SEMANTIC_BAGGER["n_estimators"] = 100

    # Enable GPU for Boosting models if available, and reduce estimators
    Config.PARAMS_SEMANTIC_BOOSTER.update({"n_estimators": 200, "device": "cuda"})
    Config.PARAMS_TEMPORAL_BOOSTER.update({"n_estimators": 200, "device": "gpu"})

    Config.PARAMS_METADATA_ANCHOR["max_iter"] = 500

    # -------------------------------------------------------------------------
    # 2. Monkey-Patch TrainingEngine for Strict Hold-Out Validation
    # -------------------------------------------------------------------------
    # We replace _prepare_data to prevent merging Train and Val.
    # This ensures Val remains a true hold-out set for the required metric calculation.

    def patched_prepare_data(self):
        print("Loading and preparing feature sets (Patched: Train Only)...")
        data = self.feature_pipeline.get_all_features()

        # Use ONLY Train data for the CV process
        y_train = data["y_train"]

        # Helper to stack features for Train (Sparse/Dense)
        def stack_train(feat_key, meta_key, is_sparse=False):
            feat = data[feat_key]
            meta = data[meta_key]
            if is_sparse:
                return sparse.hstack([feat, sparse.csr_matrix(meta)])
            else:
                return np.hstack([feat, meta])

        # Helper for Test (required by TrainingEngine to generate provisional submission)
        def stack_test(feat_key, meta_key, is_sparse=False):
            feat = data[feat_key]
            meta = data[meta_key]
            if is_sparse:
                return sparse.hstack([feat, sparse.csr_matrix(meta)])
            else:
                return np.hstack([feat, meta])

        inputs = {}
        # Construct inputs strictly for Train and Test (ignoring Val here)
        inputs["lexical_meta"] = (
            stack_train("X_train_lexical", "X_train_meta", True),
            stack_test("X_test_lexical", "X_test_meta", True),
        )

        inputs["community_meta"] = (
            stack_train("X_train_community", "X_train_meta", True),
            stack_test("X_test_community", "X_test_meta", True),
        )

        inputs["semantic_meta"] = (
            stack_train("X_train_semantic", "X_train_meta", False),
            stack_test("X_test_semantic", "X_test_meta", False),
        )

        inputs["meta_only"] = (data["X_train_meta"], data["X_test_meta"])

        return inputs, y_train, data["test_ids"]

    # Apply the patch
    TrainingEngine._prepare_data = patched_prepare_data

    # -------------------------------------------------------------------------
    # 3. Training Phase
    # -------------------------------------------------------------------------
    print("\n=== Starting Training Phase ===")
    # load_cached_data=True uses the pre-computed features in ./working
    trainer = TrainingEngine(load_cached_data=True)
    trainer.run(n_folds=5)

    # -------------------------------------------------------------------------
    # 4. Hold-Out Validation Inference
    # -------------------------------------------------------------------------
    print("\n=== Performing Hold-Out Validation ===")

    # Reload features to get access to the Validation set
    feature_pipeline = FeaturePipeline(load_cached_data=True)
    data = feature_pipeline.get_all_features()
    y_val = data["y_val"]

    # Construct Validation Inputs (must match Training structure)
    val_inputs = {}
    val_inputs["lexical_meta"] = sparse.hstack(
        [data["X_val_lexical"], sparse.csr_matrix(data["X_val_meta"])]
    )
    val_inputs["community_meta"] = sparse.hstack(
        [data["X_val_community"], sparse.csr_matrix(data["X_val_meta"])]
    )
    val_inputs["semantic_meta"] = np.hstack(
        [data["X_val_semantic"], data["X_val_meta"]]
    )
    val_inputs["meta_only"] = data["X_val_meta"]

    # Perform Inference using the trained ensemble
    learners = trainer.learners  # List of (name, input_type)
    n_folds = 5
    n_val = len(y_val)
    l1_val_preds = np.zeros((n_val, len(learners)))

    print("Predicting on Validation Set...")
    for i, (name, input_type) in enumerate(learners):
        X_v = val_inputs[input_type]
        fold_preds = np.zeros(n_val)

        # Bagging: Average predictions from all 5 fold models
        for f in range(n_folds):
            model_path = os.path.join(Config.MODEL_DIR, f"{name}_fold_{f}.joblib")
            model = joblib.load(model_path)
            fold_preds += model.predict_proba(X_v)[:, 1]

        l1_val_preds[:, i] = fold_preds / n_folds

    # Meta-Learner Prediction
    meta_model = joblib.load(os.path.join(Config.MODEL_DIR, "meta_learner.joblib"))
    final_val_probs = meta_model.predict_proba(l1_val_preds)[:, 1]

    # Compute and Print Metric
    val_auc = roc_auc_score(y_val, final_val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - final_val_probs)

    # Correlate error with dense metadata features
    correlations = []
    # X_val_meta columns correspond to Config.DENSE_FEATURES
    for idx, col_name in enumerate(Config.DENSE_FEATURES):
        feat_vals = data["X_val_meta"][:, idx]
        # Handle potential constant features to avoid NaN correlation
        if np.std(feat_vals) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        else:
            corr = 0.0
        correlations.append((col_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Features Correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({val_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        # Use the provided InferenceEngine for the final test run
        inferencer = InferenceEngine(load_cached_data=True)
        inferencer.run_inference(n_folds=5)
    else:
        print(
            f"\nValidation Metric ({val_auc}) <= Threshold ({THRESHOLD}). Submission Skipped."
        )
        # Ensure no accidental submission file exists if we failed the check
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)
            print("Removed invalid submission file.")


if __name__ == "__main__":
    main()
