import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
import library.config
from library.config import get_config, WORKING_DIR, SUBMISSION_PATH, SEED
from library.feature_engineering import generate_dataset
from library.models import train_with_mining_curriculum
from library.metrics import optimize_threshold


def run_demo():
    print("=== Starting NFL Contact Detection Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configure for Speed (Debug Mode)
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment...")

    # Get debug configurations (reduced rounds, estimators, etc.)
    l_scout, l_expert, x_expert, t_config = get_config(debug=True)

    # Update the global configuration dictionaries in the library in-place.
    # This ensures that when library.models uses these dicts, it sees the debug values.
    library.config.LGBM_SCOUT_PARAMS.update(l_scout)
    library.config.LGBM_EXPERT_PARAMS.update(l_expert)
    library.config.XGB_EXPERT_PARAMS.update(x_expert)
    library.config.TRAIN_CONFIG.update(t_config)

    print("      Debug configuration applied.")
    print(f"      Scout Rounds: {library.config.TRAIN_CONFIG['scout_rounds']}")
    print(f"      Expert Rounds: {library.config.TRAIN_CONFIG['expert_rounds']}")

    # -------------------------------------------------------------------------
    # 2. Generate Datasets
    # -------------------------------------------------------------------------
    print("\n[2/6] Generating Features...")

    # Generate Train (Sampled via debug=True)
    # We set load_cached_data=False to demonstrate the pipeline from scratch.
    print("      Processing Training Data...")
    X_train, y_train, train_ids = generate_dataset(
        split="train", load_cached_data=False, debug=True
    )
    print(f"      -> Train Shape: {X_train.shape}")

    # Generate Validation (Sampled via debug=True)
    print("      Processing Validation Data...")
    X_val, y_val, val_ids = generate_dataset(
        split="val", load_cached_data=False, debug=True
    )
    print(f"      -> Val Shape: {X_val.shape}")

    # Generate Test (Full set required for submission)
    print("      Processing Test Data...")
    X_test, _, test_ids = generate_dataset(split="test", load_cached_data=False)
    print(f"      -> Test Shape: {X_test.shape}")

    # -------------------------------------------------------------------------
    # 3. Model Training (Curriculum Learning)
    # -------------------------------------------------------------------------
    print("\n[3/6] Training Models...")

    # This function handles:
    # 1. Scout Model Training (on balanced subset)
    # 2. Hard Negative Mining (on full train)
    # 3. Expert Model Training (on Positives + Hard Negatives)
    # 4. Ensemble Creation
    ensemble = train_with_mining_curriculum(
        X_train, y_train, X_val, y_val, load_cached_data=False
    )

    # -------------------------------------------------------------------------
    # 4. Inference
    # -------------------------------------------------------------------------
    print("\n[4/6] Running Inference...")

    # Predict on Test Set
    test_preds = ensemble.predict_proba(X_test)

    # Load the optimized threshold found during validation
    thresh_path = os.path.join(WORKING_DIR, "best_threshold.npy")
    if os.path.exists(thresh_path):
        best_thresh = np.load(thresh_path)[0]
    else:
        best_thresh = 0.5

    print(f"      Applied Threshold: {best_thresh:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5/6] Creating Submission File...")

    # Create prediction dataframe
    df_preds = pd.DataFrame(
        {"contact_id": test_ids, "contact": (test_preds >= best_thresh).astype(int)}
    )

    # Load sample submission template to ensure correct order/completeness
    sample_sub_path = "./input/sample_submission.csv"
    df_sample = pd.read_csv(sample_sub_path)

    # Merge predictions onto sample submission
    # We use left join to keep sample submission structure
    df_final = df_sample[["contact_id"]].merge(df_preds, on="contact_id", how="left")

    # Fill missing values (if any) with 0 (No Contact)
    df_final["contact"] = df_final["contact"].fillna(0).astype(int)

    # Save
    df_final.to_csv(SUBMISSION_PATH, index=False)
    print(f"      Saved to: {SUBMISSION_PATH}")

    # -------------------------------------------------------------------------
    # 6. Verification
    # -------------------------------------------------------------------------
    print("\n[6/6] Verifying Results...")

    # Check file existence
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    # Check shape
    if len(df_final) != len(df_sample):
        raise AssertionError(
            f"Row count mismatch. Expected {len(df_sample)}, got {len(df_final)}"
        )

    # Check columns
    expected_cols = ["contact_id", "contact"]
    if not all(col in df_final.columns for col in expected_cols):
        raise AssertionError(f"Missing columns. Expected {expected_cols}")

    # Verify metric utility with dummy data
    print("      Verifying metric calculation logic...")
    dummy_y = np.array([0, 1, 0, 1, 0])
    dummy_p = np.array([0.1, 0.9, 0.2, 0.8, 0.4])
    t, mcc = optimize_threshold(dummy_y, dummy_p)
    if mcc < 0.5:  # Should be 1.0
        raise AssertionError("Metric calculation check failed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
