import os
import sys
import numpy as np
import pandas as pd
import random
import joblib
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.data_utils import load_data
from library.weighting import compute_sample_weights
from library.features import build_feature_matrix
from library.nbsvm import NBSVMClassifier
from library.evaluation import compute_final_metric


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_data_loading(nrows=5000):
    """Demonstrates loading data subsets and verifies structure."""
    print(f"\n[1] Loading Data Subsets (nrows={nrows})...")

    # Load subsets of data
    train_df = load_data("train", load_cached_data=False, nrows=nrows)
    val_df = load_data("val", load_cached_data=False, nrows=nrows)
    test_df = load_data("test", load_cached_data=False, nrows=nrows)

    # Verify shapes
    print(f"    Train shape: {train_df.shape}")
    print(f"    Val shape:   {val_df.shape}")
    print(f"    Test shape:  {test_df.shape}")

    assert len(train_df) == nrows, "Train subset length mismatch"
    assert (
        Config.BINARY_TARGET_COL in train_df.columns
    ), "Binary target column missing in Train"
    assert Config.TEXT_COL in train_df.columns, "Text column missing in Train"

    # Verify binary target logic
    # target >= 0.5 should be 1, else 0
    sample_check = train_df.iloc[0]
    expected_binary = 1 if sample_check[Config.TARGET_COL] >= 0.5 else 0
    assert (
        sample_check[Config.BINARY_TARGET_COL] == expected_binary
    ), "Binary target calculation incorrect"

    print("    Data loading and structure verification passed.")
    return train_df, val_df, test_df


def demo_weighting(train_df):
    """Demonstrates bias-centric sample weighting."""
    print("\n[2] Computing Sample Weights...")

    # Compute weights
    # We force load_cached_data=False to ensure we compute fresh weights for this subset
    weights = compute_sample_weights(train_df, load_cached_data=False)

    # Basic shape check
    assert len(weights) == len(train_df), "Weights length mismatch"

    # Verify logic: Rows with identity mentions should have higher weights
    # We find indices where any identity column is >= 0.5
    identity_cols = [c for c in Config.IDENTITY_COLUMNS if c in train_df.columns]
    if not identity_cols:
        print(
            "    Warning: No identity columns found in subset. Skipping detailed weight logic check."
        )
        return weights

    identities = train_df[identity_cols].fillna(0.0)
    has_identity = (identities >= 0.5).any(axis=1)

    # Check a positive case (if exists in subset)
    if has_identity.sum() > 0:
        idx_identity = has_identity.idxmax()  # First index with identity
        # Depending on implementation, idxmax returns label index, we need positional if array is numpy
        # compute_sample_weights returns a numpy array aligned with df rows.
        # We need integer location.
        pos_identity = train_df.index.get_loc(idx_identity)

        w_identity = weights[pos_identity]
        assert (
            w_identity == Config.BIAS_WEIGHT_MULTIPLIER
        ), f"Expected weight {Config.BIAS_WEIGHT_MULTIPLIER} for identity example, got {w_identity}"
        print("    Verified higher weight for identity example.")
    else:
        print("    No identity examples in this subset to verify high weights.")

    # Check a negative case
    if (~has_identity).sum() > 0:
        idx_no_ident = (~has_identity).idxmax()
        pos_no_ident = train_df.index.get_loc(idx_no_ident)
        w_base = weights[pos_no_ident]
        assert (
            w_base == Config.BASE_WEIGHT
        ), f"Expected weight {Config.BASE_WEIGHT} for standard example, got {w_base}"
        print("    Verified base weight for standard example.")

    print("    Weighting logic verification passed.")
    return weights


def demo_feature_extraction(train_df, val_df, test_df):
    """Demonstrates TF-IDF feature extraction."""
    print("\n[3] Building Feature Matrices...")

    # Force re-computation to fit the subset vocabulary
    # Note: In a real run, we might want to use the full vocab, but for this demo
    # we want to ensure the code runs end-to-end without error on the subset.
    X_train, X_val, X_test = build_feature_matrix(
        train_df, val_df, test_df, load_cached_data=False
    )

    print(f"    X_train shape: {X_train.shape}")

    # Verify dimensions
    assert X_train.shape[0] == len(train_df), "X_train row count mismatch"
    assert X_val.shape[0] == len(val_df), "X_val row count mismatch"
    assert X_test.shape[0] == len(test_df), "X_test row count mismatch"

    # Verify consistency (same number of features)
    assert (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ), "Feature dimension mismatch across splits"

    print("    Feature extraction verification passed.")
    return X_train, X_val, X_test


def demo_model_training(X_train, y_train, weights, X_val):
    """Demonstrates NBSVM model training and prediction."""
    print("\n[4] Training NBSVM Model...")

    model = NBSVMClassifier(C=1.0, random_state=42)

    # Fit model
    model.fit(X_train, y_train, sample_weight=weights)
    print("    Model fitted successfully.")

    # Predict
    val_probs = model.predict_proba(X_val)

    # Verify output
    assert val_probs.shape == (X_val.shape[0], 2), "Prediction shape mismatch"
    assert np.all(
        (val_probs >= 0) & (val_probs <= 1)
    ), "Probabilities out of [0, 1] range"

    # Extract positive class probabilities
    val_preds = val_probs[:, 1]

    print("    Prediction verification passed.")
    return model, val_preds


def demo_evaluation(val_df, val_preds):
    """Demonstrates calculation of competition metrics."""
    print("\n[5] Evaluating Model Performance...")

    y_true = val_df[Config.BINARY_TARGET_COL].values

    # Compute final metric dictionary
    metrics = compute_final_metric(y_true, val_preds, val_df)

    # Print results
    print("-" * 40)
    print(f"    Final Weighted Score: {metrics['score']:.4f}")
    print(f"    Overall AUC:          {metrics['overall_auc']:.4f}")
    print(f"    Subgroup AUC Mean:    {metrics['subgroup_auc_mean']:.4f}")
    print(f"    BPSN AUC Mean:        {metrics['bpsn_auc_mean']:.4f}")
    print(f"    BNSP AUC Mean:        {metrics['bnsp_auc_mean']:.4f}")
    print("-" * 40)

    # Verify metric structure
    assert "score" in metrics
    assert isinstance(metrics["per_identity_metrics"], pd.DataFrame)

    # Check if we have valid numbers (not NaN) for overall AUC
    # (Bias metrics might be NaN if subset is too small to have specific subgroups)
    assert not np.isnan(metrics["overall_auc"]), "Overall AUC is NaN"

    print("    Evaluation logic verification passed.")


def demo_submission(model, X_test, test_df):
    """Demonstrates generating a submission file."""
    print("\n[6] Generating Submission...")

    test_probs = model.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame({"id": test_df["id"], "prediction": test_probs})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"    Submission saved to: {Config.SUBMISSION_PATH}")

    # Verify file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify content
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(saved_df) == len(test_df), "Submission row count mismatch"
    assert (
        "id" in saved_df.columns and "prediction" in saved_df.columns
    ), "Submission columns missing"

    print("    Submission verification passed.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    Config.setup()

    # 2. Load Data (Subset for speed)
    # Using 10,000 rows for training to ensure we get some identity examples
    train_df, val_df, test_df = demo_data_loading(nrows=10000)

    # 3. Compute Weights
    weights = demo_weighting(train_df)

    # 4. Build Features
    X_train, X_val, X_test = demo_feature_extraction(train_df, val_df, test_df)

    # 5. Train Model
    y_train = train_df[Config.BINARY_TARGET_COL].values
    model, val_preds = demo_model_training(X_train, y_train, weights, X_val)

    # 6. Evaluate
    demo_evaluation(val_df, val_preds)

    # 7. Generate Submission
    demo_submission(model, X_test, test_df)

    print("\nAll demonstration steps completed successfully.")
