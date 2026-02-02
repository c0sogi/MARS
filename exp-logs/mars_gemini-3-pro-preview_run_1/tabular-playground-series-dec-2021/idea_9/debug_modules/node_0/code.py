import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import the provided library modules
from library import config, data, features, model, ensemble


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Set random seeds for reproducibility
    np.random.seed(config.SEED)

    # Override config parameters for speed in this demo
    config.N_FOLDS = 2
    config.NUM_BOOST_ROUND = 10
    config.EARLY_STOPPING_ROUNDS = 5

    # Define temporary paths for demo outputs
    demo_submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    print(f"Config: Folds={config.N_FOLDS}, BoostRounds={config.NUM_BOOST_ROUND}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading and Preprocessing Data...")

    # Load Train Data
    # Note: In a real run, load_cached_data=True would use saved parquets.
    # We force False here to demonstrate the raw processing logic or True if available.
    # We'll stick to default to use cache if present, else process.
    df_train_full = data.load_and_preprocess("train")

    # Load Test Data
    df_test_full = data.load_and_preprocess("test")

    # Subsample data for the demonstration to ensure quick execution
    # We take 2000 training samples and 500 test samples
    df_train_demo = df_train_full.sample(n=2000, random_state=config.SEED).reset_index(
        drop=True
    )
    df_test_demo = df_test_full.sample(n=500, random_state=config.SEED).reset_index(
        drop=True
    )

    print(f"Full Train Shape: {df_train_full.shape}")
    print(f"Demo Train Shape: {df_train_demo.shape}")
    print(f"Demo Test Shape: {df_test_demo.shape}")

    # Verify Feature Engineering
    # Check if geometry features were added (e.g., 'Euclidean_Distance_To_Hydrology')
    expected_feat = "Euclidean_Distance_To_Hydrology"
    if expected_feat not in df_train_demo.columns:
        raise AssertionError(f"Feature Engineering failed: {expected_feat} not found.")

    # Check if dense categorical indices were added
    if "Soil_Type_Index" not in df_train_demo.columns:
        raise AssertionError("Feature Engineering failed: Soil_Type_Index not found.")

    # -------------------------------------------------------------------------
    # 3. Data Splitting
    # -------------------------------------------------------------------------
    print("\n[Step 3] Splitting Features and Target...")

    X_train, y_train = data.get_X_y(df_train_demo)
    X_test, y_test = data.get_X_y(df_test_demo)

    # Validate Split
    if config.ID_COL in X_train.columns:
        raise AssertionError("ID column was not dropped from X_train.")
    if config.TARGET_COL in X_train.columns:
        raise AssertionError("Target column was not dropped from X_train.")
    if y_train is None:
        raise AssertionError("Target y_train is None.")
    if y_test is not None:
        raise AssertionError(
            "y_test should be None for test set (unless provided in file)."
        )

    # Keep Test IDs for submission
    test_ids = df_test_demo[config.ID_COL]

    # -------------------------------------------------------------------------
    # 4. Model Training (Single Fold / Direct Usage)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Demonstrating XGBTrainer (Single Split)...")

    # Create a simple holdout split for demonstration
    split_idx = int(len(X_train) * 0.8)
    X_tr_split = X_train.iloc[:split_idx]
    y_tr_split = y_train.iloc[:split_idx]
    X_val_split = X_train.iloc[split_idx:]
    y_val_split = y_train.iloc[split_idx:]

    trainer = model.XGBTrainer()

    # Validate parameter override logic in XGBTrainer init
    if trainer.params["objective"] != "multi:softprob":
        raise AssertionError("XGBTrainer did not force objective to 'multi:softprob'.")

    trainer.train(
        X_tr_split, y_tr_split, X_val=X_val_split, y_val=y_val_split, verbose_eval=False
    )

    # Predict
    preds_prob = trainer.predict(X_val_split)

    # Validate Predictions
    if preds_prob.shape != (len(X_val_split), config.NUM_CLASSES):
        raise AssertionError(
            f"Prediction shape mismatch. Expected {(len(X_val_split), config.NUM_CLASSES)}, got {preds_prob.shape}"
        )

    # Check if probabilities sum to ~1
    row_sums = preds_prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise AssertionError("Predicted probabilities do not sum to 1.")

    print("Single model training and prediction successful.")

    # -------------------------------------------------------------------------
    # 5. Ensemble Pipeline (CV & Pseudo-Labeling)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Demonstrating EnsemblePipeline (CV)...")

    pipeline = ensemble.EnsemblePipeline()

    # Run Cross-Validation
    oof_probs, test_probs_avg = pipeline.run_cv_training(X_train, y_train, X_test)

    # Validate CV Outputs
    if len(oof_probs) != len(X_train):
        raise AssertionError("OOF predictions length mismatch.")
    if len(test_probs_avg) != len(X_test):
        raise AssertionError("Test predictions length mismatch.")

    print("CV Training completed.")

    # -------------------------------------------------------------------------
    # 6. Augmentation / Pseudo-Labeling
    # -------------------------------------------------------------------------
    print("\n[Step 6] Demonstrating Pseudo-Label Augmentation...")

    # To ensure we trigger augmentation, we artificially boost confidence of test predictions
    # Make the first 10 predictions 100% confident for class 0
    mock_test_probs = test_probs_avg.copy()
    mock_test_probs[:10, :] = 0.0
    mock_test_probs[:10, 0] = 1.0  # High confidence for class 0

    # Generate augmented dataset
    X_aug, y_aug = pipeline.generate_augmented_train_set(
        X_train, y_train, X_test, mock_test_probs
    )

    # Validate Augmentation
    # We expect at least 10 samples to be added
    expected_min_len = len(X_train) + 10
    if len(X_aug) < expected_min_len:
        print(f"Original size: {len(X_train)}, Augmented size: {len(X_aug)}")
        # Note: It might be exactly equal if threshold is very high and we didn't mock enough,
        # but with manual mock it should pass.
        raise AssertionError("Augmentation failed to add samples.")

    print(
        f"Augmentation successful. Increased training set from {len(X_train)} to {len(X_aug)}."
    )

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Submission...")

    pipeline.save_submission(test_ids, test_probs_avg, output_path=demo_submission_path)

    if not os.path.exists(demo_submission_path):
        raise AssertionError("Submission file was not created.")

    # Verify file content format
    sub_df = pd.read_csv(demo_submission_path)
    if list(sub_df.columns) != [config.ID_COL, config.TARGET_COL]:
        raise AssertionError(f"Submission columns incorrect. Got {sub_df.columns}")
    if len(sub_df) != len(df_test_demo):
        raise AssertionError("Submission row count mismatch.")

    # Verify inverse mapping (targets should be original class IDs, e.g., 1, 2, 3...)
    # Our mapping maps 1->0. So output should be >= 1.
    if sub_df[config.TARGET_COL].min() < 1:
        raise AssertionError(
            "Submission contains invalid class labels (likely internal 0-indexed indices)."
        )

    print(f"Submission generated at {demo_submission_path}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
