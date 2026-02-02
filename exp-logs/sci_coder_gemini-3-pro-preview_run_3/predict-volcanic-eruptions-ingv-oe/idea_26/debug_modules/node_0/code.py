import os
import pandas as pd
import numpy as np
import shutil
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Import the provided library modules
import library.config as config
import library.feature_engineering as fe
import library.data_processor as dp
import library.model_trainer as mt


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo:
    1. Creates a specific working directory.
    2. Generates mini-metadata files (subsets) to speed up feature extraction.
    3. Overrides configuration parameters for speed.
    """
    print("Setting up demo environment...")

    # 1. Define and create demo working directory
    demo_working_dir = os.path.join(config.WORKING_DIR, "demo_execution")
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config working directory
    config.WORKING_DIR = demo_working_dir

    # 2. Create Mini Metadata (Subset of data)
    # Load original metadata
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample small subsets (e.g., 20 samples for train, 10 for val, 10 for test)
    # This ensures the demo runs in seconds/minutes rather than hours
    mini_train = train_meta.head(20).copy()
    mini_val = val_meta.head(10).copy()
    mini_test = test_meta.head(10).copy()

    # Save mini metadata
    mini_train_path = os.path.join(demo_working_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_working_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_working_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Override config paths to point to mini metadata
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path

    print(
        f"Created mini-datasets: Train={len(mini_train)}, Val={len(mini_val)}, Test={len(mini_test)}"
    )

    # 3. Override Model Hyperparameters for Speed
    # Reduce estimators and increase learning rate slightly for quick convergence check
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["learning_rate"] = 0.1
    config.LGBM_PARAMS["min_child_samples"] = 2  # Reduce constraint for small data
    config.EARLY_STOPPING_ROUNDS = 5
    config.N_FOLDS = 2  # Reduce folds

    print("Configuration overrides applied.")


def demo_feature_engineering():
    """
    Demonstrates loading and processing data using data_processor and feature_engineering.
    """
    print("\n=== Demo: Feature Engineering ===")

    # Load Training Data (Features + Target)
    # Note: load_cached_data=False forces re-computation for demonstration purposes
    X_train, y_train = dp.load_train_data(load_cached_data=False)

    # Validation
    assert not X_train.empty, "X_train should not be empty"
    assert len(X_train) == 20, f"Expected 20 training samples, got {len(X_train)}"
    assert len(y_train) == 20, f"Expected 20 target values, got {len(y_train)}"
    assert (
        not X_train.isnull().values.any()
    ), "Features should not contain NaNs after processing"

    print(f"Successfully generated training features. Shape: {X_train.shape}")

    # Load Validation Data
    X_val, y_val = dp.load_val_data(load_cached_data=False)
    assert len(X_val) == 10, f"Expected 10 validation samples, got {len(X_val)}"
    print(f"Successfully generated validation features. Shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val


def demo_model_training(X_train, y_train):
    """
    Demonstrates training the ensemble model using model_trainer.
    """
    print("\n=== Demo: Model Training ===")

    # Train K-Fold Ensemble
    # This uses the overridden config.N_FOLDS (2) and config.LGBM_PARAMS
    models = mt.train_kfold_ensemble(X_train, y_train, save_models=True)

    # Validation
    assert isinstance(models, list), "train_kfold_ensemble should return a list"
    assert (
        len(models) == config.N_FOLDS
    ), f"Expected {config.N_FOLDS} models, got {len(models)}"

    # Check if model files were saved
    for i in range(config.N_FOLDS):
        model_path = os.path.join(config.WORKING_DIR, f"lgbm_model_fold_{i}.txt")
        assert os.path.exists(model_path), f"Model file {model_path} was not saved."

    print(f"Successfully trained {len(models)} models.")
    return models


def demo_inference_and_submission(models):
    """
    Demonstrates prediction on test set and generating submission file.
    """
    print("\n=== Demo: Inference and Submission ===")

    # Load Test Data (Features + IDs)
    X_test, test_ids = dp.load_test_data(load_cached_data=False)

    # Validation
    assert len(X_test) == 10, f"Expected 10 test samples, got {len(X_test)}"

    # Predict
    predictions = mt.predict_with_ensemble(models, X_test)

    # Validation
    assert len(predictions) == len(X_test), "Prediction count mismatch"
    assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"

    print(f"Generated predictions. Mean predicted time: {np.mean(predictions):.2f}")

    # Save Submission
    submission_path = os.path.join(config.WORKING_DIR, "submission_demo.csv")
    mt.save_submission(test_ids, predictions, submission_path)

    # Verify file creation
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify content format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns"
    assert len(sub_df) == 10, "Incorrect number of rows in submission"

    print(f"Submission saved to {submission_path}")


def main():
    # Set global seed for reproducibility
    np.random.seed(config.SEED)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Feature Engineering
        X_train, y_train, X_val, y_val = demo_feature_engineering()

        # 3. Model Training
        # We combine mini_train and mini_val for the ensemble training demonstration
        # In a real scenario, we might keep them separate or use the full train set.
        # Here we concatenate to have enough data for the folds if needed,
        # though the function handles splitting internally.
        # We'll just use X_train/y_train as the "development" set for CV.
        models = demo_model_training(X_train, y_train)

        # 4. Inference & Submission
        demo_inference_and_submission(models)

        print("\nAll demo steps completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
