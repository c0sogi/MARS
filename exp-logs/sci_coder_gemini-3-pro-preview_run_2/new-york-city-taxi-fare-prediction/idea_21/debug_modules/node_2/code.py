import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

# Import provided library modules
import library.config as config
import library.data_factory as data_factory
import library.feature_pipeline as feature_pipeline
import library.model_engine as model_engine


def main():
    print("Starting NYC Taxi Fare Prediction Demo...")

    # ==========================================
    # 1. Configuration & Patching for Speed
    # ==========================================
    print("\n[Config] Patching parameters for fast demonstration...")

    # Reduce Learner Set size to 50,000 samples (down from 5M) to speed up OOF generation and training
    # We must patch the variable in data_factory where it is imported/used
    original_subset_size = data_factory.LEARNER_SUBSET_SIZE
    data_factory.LEARNER_SUBSET_SIZE = 50_000
    print(
        f"  - LEARNER_SUBSET_SIZE: {original_subset_size} -> {data_factory.LEARNER_SUBSET_SIZE}"
    )

    # Reduce XGBoost estimators and relax learning rate for quick convergence
    # XGB_PARAMS is a dict in config, so we can modify it in place
    config.XGB_PARAMS["n_estimators"] = 100
    config.XGB_PARAMS["learning_rate"] = 0.1
    print(f"  - XGB_PARAMS['n_estimators']: -> {config.XGB_PARAMS['n_estimators']}")

    # Reduce Early Stopping Rounds
    # Must patch in model_engine where it is imported
    model_engine.EARLY_STOPPING_ROUNDS = 10
    print(f"  - EARLY_STOPPING_ROUNDS: -> {model_engine.EARLY_STOPPING_ROUNDS}")

    # Set Seed for Reproducibility
    np.random.seed(config.SEED)

    # ==========================================
    # 2. Feature Engineering Pipeline
    # ==========================================
    print("\n[Pipeline] Running Feature Engineering (scratch)...")
    # We force load_cached_data=False to demonstrate the logic execution.
    # This will:
    # 1. Load Train (Wisdom) and compute Global Stats
    # 2. Load and Subsample Learner Set
    # 3. Compute OOF Fingerprints for Learner
    # 4. Process Val and Test sets

    try:
        X_train, y_train, X_val, y_val, X_test, test_keys = (
            feature_pipeline.build_features(load_cached_data=False)
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

    # ==========================================
    # 3. Validation of Data
    # ==========================================
    print("\n[Validation] Verifying Feature Sets...")

    # Check Shapes
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # Assert Subset Size
    if len(X_train) != data_factory.LEARNER_SUBSET_SIZE:
        raise AssertionError(
            f"X_train size mismatch. Expected {data_factory.LEARNER_SUBSET_SIZE}, got {len(X_train)}"
        )

    # Assert Alignment
    if len(X_train) != len(y_train):
        raise AssertionError("X_train and y_train length mismatch")

    # Assert Feature Consistency
    if X_train.shape[1] != X_test.shape[1]:
        raise AssertionError("Feature count mismatch between Train and Test")

    # Check for critical engineered features
    required_features = [
        "dist_haversine",
        "L5_mean",
        "L6_std",
        "rate_gh5_hour",
        "hour_sin",
    ]
    for feat in required_features:
        if feat not in X_train.columns:
            raise AssertionError(f"Missing required feature: {feat}")

    # Check Data Types (should be numeric for XGBoost)
    if not pd.api.types.is_numeric_dtype(X_train["dist_haversine"]):
        raise AssertionError("dist_haversine is not numeric")

    print("  Data validation passed.")

    # ==========================================
    # 4. Model Training
    # ==========================================
    print("\n[Model] Training XGBoost Regressor...")

    # Train model (scratch)
    model = model_engine.train_regressor(
        X_train, y_train, X_val, y_val, load_cached_model=False
    )

    # Validate Model Object
    if not isinstance(model, xgb.XGBRegressor):
        raise AssertionError("Returned model is not an XGBRegressor")

    # Check if fitted (booster should be available)
    if not hasattr(model, "get_booster"):
        raise AssertionError("Model does not appear to be fitted")

    print(f"  Model trained. Best Score (RMSE): {model.best_score}")

    # ==========================================
    # 5. Prediction & Submission
    # ==========================================
    print("\n[Prediction] Generating Predictions on Test Set...")

    preds = model_engine.predict_fare(model, X_test)

    # Validate Predictions
    if len(preds) != len(X_test):
        raise AssertionError("Prediction count mismatch")

    # Check for negative fares (should be floored at 2.50)
    if np.any(preds < 2.50):
        raise AssertionError("Found predictions below minimum fare ($2.50)")

    print(f"  Predictions generated. Mean Fare: ${preds.mean():.2f}")

    # Generate Submission
    output_csv = "./working/demo_submission.csv"
    print(f"\n[Submission] Saving to {output_csv}...")

    model_engine.generate_submission(preds, test_keys, output_path=output_csv)

    # Verify File Existence
    if not os.path.exists(output_csv):
        raise FileNotFoundError("Submission file was not created")

    # Verify File Content
    sub_df = pd.read_csv(output_csv)
    if list(sub_df.columns) != ["key", "fare_amount"]:
        raise AssertionError("Submission columns are incorrect")

    if len(sub_df) != len(test_keys):
        raise AssertionError("Submission row count mismatch")

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
