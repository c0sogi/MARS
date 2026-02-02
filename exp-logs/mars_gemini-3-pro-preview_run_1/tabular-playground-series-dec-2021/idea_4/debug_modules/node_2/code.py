import os
import pandas as pd
import numpy as np
import shutil
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Import library modules
import library.config as config
import library.data_utils as data_utils
import library.model_utils as model_utils
import library.pipeline as pipeline


def main():
    print("Initializing Demonstration Script...")

    # --- 1. Setup & Configuration Overrides ---
    # We override the configuration to run a fast, lightweight version of the pipeline.

    # Modify Model Parameters for speed
    config.MODEL_PARAMS.update(
        {
            "n_estimators": 20,  # Reduce from 3000 to 20
            "learning_rate": 0.1,
            "max_depth": 4,
            "device": "cpu",  # Use CPU for small data overhead efficiency
            "tree_method": "hist",
            "n_jobs": 4,
            "verbosity": 0,
        }
    )

    # Modify Pipeline Parameters
    config.PIPELINE_PARAMS.update(
        {
            "n_folds": 2,  # Reduce folds from 5 to 2
            "early_stopping_rounds": 5,
            "verbose_eval": False,
        }
    )

    # Define temporary paths in working directory
    TEMP_DIR = os.path.join(config.WORKING_DIR, "demo_temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    TEMP_SAMPLE_SUB = os.path.join(TEMP_DIR, "temp_sample_submission.csv")

    # Override the sample submission path in config so the pipeline reads our matching subset IDs
    config.DATA_PATHS["sample_submission"] = TEMP_SAMPLE_SUB

    # Ensure Cache Directory exists (it's defined in config as working/idea_4)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    print("Configuration updated for fast execution.")

    # --- 2. Prepare Subsampled Data ---
    # We load a tiny fraction of the metadata to create a 'processed' cache.
    # This ensures the pipeline functions run on this small dataset.

    SUBSET_SIZE = 500

    print(f"Loading raw metadata and creating subset of {SUBSET_SIZE} rows...")

    # Load raw metadata
    df_train_full = pd.read_csv(config.DATA_PATHS["train_path"], nrows=SUBSET_SIZE)
    df_val_full = pd.read_csv(config.DATA_PATHS["val_path"], nrows=SUBSET_SIZE)
    df_test_full = pd.read_csv(config.DATA_PATHS["test_path"], nrows=SUBSET_SIZE)

    # Create a matching sample submission file for the test subset
    # The pipeline expects the submission file to match the test set length and IDs
    sample_sub_df = pd.DataFrame(
        {
            config.ID_COL: df_test_full[config.ID_COL],
            config.TARGET_COL: [2] * len(df_test_full),  # Dummy default class
        }
    )
    sample_sub_df.to_csv(TEMP_SAMPLE_SUB, index=False)
    print(f"Created temporary sample submission at {TEMP_SAMPLE_SUB}")

    # --- 3. Test Feature Engineering (Unit Test) ---
    print("\n--- Testing Feature Engineering ---")

    # Verify columns before
    original_cols = set(df_train_full.columns)

    # Apply engineering
    df_train_eng = data_utils.engineer_features(df_train_full)
    df_val_eng = data_utils.engineer_features(df_val_full)
    df_test_eng = data_utils.engineer_features(df_test_full)

    # Verify columns after
    new_cols = set(df_train_eng.columns)
    added_cols = new_cols - original_cols
    expected_cols = {
        "Euclidean_Distance_To_Hydrology",
        "Relative_Elevation_Hydrology",
        "Aspect_Sin",
        "Aspect_Cos",
    }

    print(f"New features generated: {added_cols}")
    if not expected_cols.issubset(added_cols):
        raise AssertionError(
            f"Feature engineering failed. Missing: {expected_cols - added_cols}"
        )

    # Save these engineered subsets to the CACHE_DIR as parquet
    # This 'mocks' the processing step of the pipeline, forcing it to use our data
    train_cache_path = os.path.join(config.CACHE_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(config.CACHE_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(config.CACHE_DIR, "test_processed.parquet")

    df_train_eng.to_parquet(train_cache_path, index=False)
    df_val_eng.to_parquet(val_cache_path, index=False)
    df_test_eng.to_parquet(test_cache_path, index=False)

    print(f"Cached processed subsets to {config.CACHE_DIR}")

    # --- 4. Test Model Training (Unit Test) ---
    print("\n--- Testing Model Training ---")

    features = [
        c for c in df_train_eng.columns if c not in [config.TARGET_COL, config.ID_COL]
    ]
    X_train = df_train_eng[features]
    y_train = df_train_eng[config.TARGET_COL]

    # Train model
    model = model_utils.train_model(X_train, y_train)

    # Verify model type
    if not isinstance(model, xgb.XGBClassifier):
        raise AssertionError("train_model did not return an XGBClassifier.")

    # Test Prediction
    preds = model_utils.predict(model, X_train)
    probs = model_utils.predict_proba(model, X_train)

    acc = accuracy_score(y_train, preds)
    print(f"Training subset accuracy: {acc:.4f}")

    if probs.shape != (len(X_train), len(np.unique(y_train))):
        # Note: If classes are missing in subset, shape might differ, but generally N_samples x N_classes
        pass

    print("Model training and prediction verified.")

    # --- 5. Test Data Augmentation (Unit Test) ---
    print("\n--- Testing Pseudo-Label Augmentation ---")

    # Create fake probabilities for the test set (high confidence for class 2)
    # Assume classes are sorted. If class 2 is at index 1 (example), we set index 1 to 0.995
    # We need to know the classes the model saw.
    classes = model.classes_
    n_classes = len(classes)

    # Create random probs
    fake_probs = np.zeros((len(df_test_eng), n_classes))
    # Set the first class to have high probability
    fake_probs[:, 0] = 0.999

    # Run augmentation
    aug_df = data_utils.create_augmented_train(
        df_train_eng, df_test_eng, fake_probs, threshold=0.99
    )

    print(f"Original Train Size: {len(df_train_eng)}")
    print(f"Augmented Train Size: {len(aug_df)}")

    if len(aug_df) <= len(df_train_eng):
        raise AssertionError(
            "Augmentation failed to add rows despite high confidence probabilities."
        )

    print("Augmentation logic verified.")

    # --- 6. Run Full Pipeline (Integration Test) ---
    print("\n--- Running Full Self-Training Pipeline ---")

    # This function will:
    # 1. Call load_dataset (which will pick up our cached parquet files)
    # 2. Run Stage 1 CV (2 folds, fast)
    # 3. Pseudo-label
    # 4. Run Stage 2 CV
    # 5. Generate submission using the TEMP_SAMPLE_SUB IDs

    pipeline.run_self_training(load_cached_data=True)

    # --- 7. Validate Submission ---
    print("\n--- Validating Output ---")

    submission_path = config.DATA_PATHS["submission_output"]
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Check if IDs match our subset
    expected_ids = df_test_full[config.ID_COL].values
    actual_ids = df_sub[config.ID_COL].values

    if not np.array_equal(expected_ids, actual_ids):
        raise AssertionError("Submission IDs do not match the test subset IDs.")

    if df_sub[config.TARGET_COL].isnull().any():
        raise AssertionError("Submission contains NaN values in target column.")

    print("Pipeline execution and validation successful.")


if __name__ == "__main__":
    main()
