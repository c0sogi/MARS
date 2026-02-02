import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset
from library.features import HybridFeaturePipeline
from library.model import BaggedLREnsemble


def main():
    print("=== Starting Random Acts of Pizza Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Small subset for speed

    # Reduce Grid Search complexity
    Config.LOGREG_C_GRID = [0.1, 1.0]

    # Reduce Ensemble complexity
    Config.BAGGING_PARAMS["n_estimators"] = 2

    # Reduce Feature dimensionality for speed
    Config.TFIDF_TEXT_PARAMS["max_features"] = 50
    Config.TFIDF_SUBREDDIT_PARAMS["max_features"] = 20

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Setup logger
    logger = setup_logger("demo_script")
    print("Configuration complete. DEBUG mode enabled.")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Step 2] Loading Datasets...")

    # Force reload from source to demonstrate loading logic
    df_train, df_val, df_test = load_dataset(load_cached_data=False)

    # Verification
    print(f"Train Shape: {df_train.shape}")
    print(f"Val Shape:   {df_val.shape}")
    print(f"Test Shape:  {df_test.shape}")

    assert (
        len(df_train) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} training samples in debug mode, got {len(df_train)}"
    assert (
        Config.TARGET_COL in df_train.columns
    ), "Target column missing from training data"
    assert (
        Config.TARGET_COL not in df_test.columns
    ), "Target column should not be in test data"

    print("Data loading verification passed.")

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 3] Executing Feature Engineering Pipeline...")

    # Instantiate the pipeline
    pipeline = HybridFeaturePipeline()

    # Fit on training data
    print("Fitting pipeline...")
    pipeline.fit(df_train)
    assert pipeline.is_fitted, "Pipeline should be marked as fitted after fit()"

    # Transform all splits
    print("Transforming datasets...")
    X_train = pipeline.transform(df_train)
    X_val = pipeline.transform(df_val)
    X_test = pipeline.transform(df_test)

    # Extract targets
    y_train = df_train[Config.TARGET_COL].values.astype(int)
    y_val = df_val[Config.TARGET_COL].values.astype(int)
    test_ids = df_test[Config.ID_COL].values

    # Verification
    print(f"Feature Matrix Shape (Train): {X_train.shape}")

    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ), "Feature dimensions mismatch across splits"
    assert not np.isnan(X_train).any(), "Feature matrix contains NaNs"
    assert len(y_train) == len(X_train), "Mismatch between features and labels"

    print("Feature engineering verification passed.")

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 4] Training Bagged Logistic Regression Ensemble...")

    # Instantiate model
    model = BaggedLREnsemble()

    # Optimize and Fit
    # This runs internal CV to find best C, then fits BaggingClassifier
    model.optimize_and_fit(X_train, y_train)

    # Verification
    assert model.model is not None, "Internal model should be initialized after fitting"
    assert model.best_c in Config.LOGREG_C_GRID, "Selected C should be from the grid"

    print(f"Model trained successfully. Best C: {model.best_c}")

    # ---------------------------------------------------------
    # 5. Evaluation & Submission
    # ---------------------------------------------------------
    print("\n[Step 5] Evaluating and Generating Submission...")

    # Validation Prediction
    val_probs = model.predict_proba(X_val)

    # Calculate Metric
    try:
        val_auc = roc_auc_score(y_val, val_probs)
        print(f"Validation ROC AUC: {val_auc:.4f}")
    except ValueError:
        # This might happen if the small debug sample only has one class
        print(
            "Skipping AUC calculation (likely due to single-class sample in DEBUG mode)."
        )
        val_auc = 0.5

    assert val_probs.shape[0] == len(y_val), "Prediction shape mismatch"
    assert np.all(
        (val_probs >= 0) & (val_probs <= 1)
    ), "Probabilities must be in [0, 1]"

    # Test Prediction
    test_probs = model.predict_proba(X_test)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: test_probs}
    )

    # Verify Submission Format
    print("Submission Head:")
    print(submission_df.head())

    assert len(submission_df) == len(df_test), "Submission row count mismatch"
    assert list(submission_df.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns mismatch"

    # Save to a demo path
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_submission_path, index=False)

    print(f"Submission saved to {demo_submission_path}")
    print("\n=== Pipeline Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
