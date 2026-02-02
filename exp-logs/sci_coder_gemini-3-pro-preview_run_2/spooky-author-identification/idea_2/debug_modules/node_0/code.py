import os
import sys
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.data_loader import load_data
from library.features import HybridFeatureGenerator
from library.stacking_manager import StackingManager
from library.utils import seed_everything


def main():
    print("=== Starting Author Identification Pipeline Demo ===")

    # 1. Configuration & Setup
    # Override Config for speed in this demonstration
    print("Configuring for fast demonstration...")
    Config.N_FOLDS = 2  # Reduce folds
    Config.SVD_N_COMPONENTS = 20  # Reduce dimensions
    Config.XGB_PARAMS["n_estimators"] = 5  # Fewer trees
    Config.LR_PARAMS["max_iter"] = 20  # Fewer iterations
    Config.WORD_NGRAM_RANGE = (1, 1)  # Simpler n-grams
    Config.CHAR_NGRAM_RANGE = (1, 2)

    # Set seeds
    seed_everything(Config.SEED)

    # 2. Data Loading
    # We use debug=True to load a small subset of data
    print("\n[Step 1] Loading Data...")
    train_df, val_df, test_df = load_data(debug=True, n_debug_samples=200)

    # Validation
    assert len(train_df) == 200, "Train DataFrame should have 200 samples in debug mode"
    assert len(val_df) == 200, "Val DataFrame should have 200 samples in debug mode"
    assert len(test_df) == 200, "Test DataFrame should have 200 samples in debug mode"
    print(f"Loaded {len(train_df)} training samples.")

    # 3. Feature Engineering
    print("\n[Step 2] Generating Features...")
    feature_gen = HybridFeatureGenerator()

    # Process features (force recompute by setting load_cached_data=False for demo purposes,
    # or rely on the fact that config hash changed so it won't find old cache)
    features = feature_gen.process(
        train_df, val_df, test_df, load_cached_data=False, debug=True
    )

    # Unpack features
    X_train_sparse = features["train_sparse"]
    X_val_sparse = features["val_sparse"]
    X_test_sparse = features["test_sparse"]
    X_train_dense = features["train_dense"]
    X_test_dense = features["test_dense"]
    y_train = features["y_train"]
    y_val = features["y_val"]
    label_classes = features["label_classes"]

    # Validation
    assert X_train_sparse.shape[0] == len(train_df)
    assert X_train_dense.shape[0] == len(train_df)
    assert X_train_dense.shape[1] == Config.SVD_N_COMPONENTS
    assert len(y_train) == len(train_df)
    assert len(label_classes) == 3
    print(f"Sparse Feature Shape: {X_train_sparse.shape}")
    print(f"Dense Feature Shape: {X_train_dense.shape}")

    # 4. Stacking Ensemble
    print("\n[Step 3] Stacking Ensemble Manager...")
    manager = StackingManager()

    # 4a. Generate OOF Predictions (Layer 1)
    print("Generating Out-Of-Fold Predictions...")
    oof_preds = manager.get_oof_predictions(
        X_train_sparse, X_train_dense, y_train, load_cached_data=False, debug=True
    )

    # Validation
    assert "lr" in oof_preds and "mnb" in oof_preds and "xgb" in oof_preds
    assert oof_preds["lr"].shape == (len(train_df), 3)
    # Check that probabilities sum roughly to 1 (allowing for float precision)
    row_sums = oof_preds["lr"].sum(axis=1)
    assert np.allclose(row_sums, 1.0), "OOF probabilities should sum to 1"

    # 4b. Train Meta-Learner (Layer 2)
    print("Training Meta-Learner...")
    meta_learner = manager.train_meta_learner(oof_preds, y_train)

    # Validation
    assert hasattr(meta_learner, "coef_"), "Meta-learner should be fitted"

    # 4c. Refit Base Models on Full Training Data
    # Note: In a real scenario, we might concatenate train+val, but here we follow the provided logic
    # which seems to imply refitting on the training set used for OOF (or a combined set if provided).
    # The library method refit_base_models takes X_sparse, X_dense, y.
    print("Refitting Base Models...")
    base_models = manager.refit_base_models(X_train_sparse, X_train_dense, y_train)

    # Validation
    assert len(base_models) == 3

    # 5. Final Prediction
    print("\n[Step 4] Generating Final Predictions...")
    final_probs = manager.predict_ensemble(
        base_models, meta_learner, X_test_sparse, X_test_dense
    )

    # Validation
    assert final_probs.shape == (len(test_df), 3)
    assert np.all((final_probs >= 0) & (final_probs <= 1))
    print(f"Predictions generated for {len(final_probs)} test samples.")

    # 6. Submission
    print("\n[Step 5] Saving Submission...")
    manager.save_submission(test_df["id"], final_probs, label_classes)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert submission_df.shape == (len(test_df), 4)  # id + 3 classes
    assert list(submission_df.columns) == ["id", "EAP", "HPL", "MWS"]

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
