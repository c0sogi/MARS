import os
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import ID_COL, WORKING_DIR
from library.utils import set_seed, save_submission
from library.data_loader import load_datasets
from library.features import FeatureEngineer
from library.training_pipeline import train_ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Configuration and Setup
    set_seed(42)

    # 2. Data Loading (Debug Mode)
    # We use debug=True to load only 100 samples for fast verification.
    # load_from_cache=False ensures we demonstrate the raw loading logic.
    print("\n[Step 1] Loading Datasets...")
    X_train, y_train, X_val, y_val, X_test = load_datasets(
        load_from_cache=False, debug=True
    )

    # Verification
    print(f"  Train shape: {X_train.shape}, Target shape: {y_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    assert len(X_train) == len(y_train), "Training features and target length mismatch."
    assert not X_train.empty, "Training data is empty."
    assert ID_COL in X_test.columns, f"ID column '{ID_COL}' missing from test set."

    # 3. Feature Engineering
    print("\n[Step 2] Feature Engineering...")
    # Initialize engineer (disable cache to force computation for demo)
    fe = FeatureEngineer(load_from_cache=False, debug=True)

    # Fit and Transform Training Data
    print("  Processing Training Data...")
    X_train_feats = fe.fit_transform(X_train, split_name="train")

    # Transform Validation and Test Data
    print("  Processing Validation and Test Data...")
    X_val_feats = fe.transform(X_val, split_name="val")
    X_test_feats = fe.transform(X_test, split_name="test")

    # Verification of Feature Dictionary Structure
    expected_keys = {"lexical", "semantic", "community"}
    assert expected_keys.issubset(
        X_train_feats.keys()
    ), "Missing feature views in output."

    # Verification of Dimensions
    n_train = len(X_train)
    assert (
        X_train_feats["lexical"].shape[0] == n_train
    ), "Lexical feature row count mismatch."
    assert (
        X_train_feats["semantic"].shape[0] == n_train
    ), "Semantic feature row count mismatch."
    assert (
        X_train_feats["community"].shape[0] == n_train
    ), "Community feature row count mismatch."

    print("  Feature engineering successful.")

    # 4. Model Training (Ensemble Stacking)
    print("\n[Step 3] Training Ensemble...")
    # This function handles:
    # - 5-Fold CV for Base Learners (Lexical, Semantic, Community)
    # - OOF Prediction Generation
    # - Meta-Learner Training
    # - Final Retraining on full training set
    models = train_ensemble(X_train_feats, y_train)

    # Verification
    assert "meta" in models, "Meta-learner model missing from returned dictionary."
    assert "lexical" in models, "Lexical base learner missing."
    assert "semantic" in models, "Semantic base learner missing."
    assert "community" in models, "Community base learner missing."
    print("  Ensemble training successful.")

    # 5. Inference
    print("\n[Step 4] Running Inference on Test Set...")

    # Generate Level 1 Predictions (Base Learners)
    p_lex = models["lexical"].predict_proba(X_test_feats)
    p_sem = models["semantic"].predict_proba(X_test_feats)
    p_com = models["community"].predict_proba(X_test_feats)

    # Stack predictions for Level 2 (Meta Learner)
    # Shape: (n_samples, 3)
    L1_preds = np.column_stack([p_lex, p_sem, p_com])

    # Generate Final Predictions
    final_preds = models["meta"].predict_proba(L1_preds)

    # Verification
    assert len(final_preds) == len(
        X_test
    ), "Prediction count does not match test set size."
    assert np.all(
        (final_preds >= 0) & (final_preds <= 1)
    ), "Probabilities out of [0, 1] range."
    print(f"  Generated {len(final_preds)} predictions.")

    # 6. Submission
    print("\n[Step 5] Saving Submission...")
    submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")
    request_ids = X_test[ID_COL].values

    save_submission(request_ids, final_preds, output_path=submission_path)

    # Verification
    assert os.path.exists(submission_path), "Submission file was not created."
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (len(X_test), 2), "Submission file has incorrect shape."
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect submission columns."

    print(f"  Submission saved to {submission_path}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
