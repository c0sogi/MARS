import os
import sys
import pandas as pd
import numpy as np
import warnings

# Add the current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import provided library modules
from library import config
from library import data_loader
from library import feature_engineering
from library import model_trainer
from library import inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Overriding configuration for fast demonstration...")

    # Enable debug mode to sample a small subset of data
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 5000  # Small sample for quick execution

    # Reduce Cross-Validation folds
    config.N_FOLDS = 2

    # Reduce XGBoost complexity for speed
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["n_jobs"] = 4

    # Disable cache to force execution of logic
    config.USE_CACHE = False

    print(f"Debug Mode: {config.DEBUG}")
    print(f"Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print(f"Folds: {config.N_FOLDS}")
    print(f"Estimators: {config.XGB_PARAMS['n_estimators']}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Loading Data...")

    # Load data (this will trigger sampling due to DEBUG=True)
    df_train, df_test = data_loader.load_dataset(load_cached_data=False)

    # Validation
    print(f"Train Shape: {df_train.shape}")
    print(f"Test Shape: {df_test.shape}")

    assert (
        len(df_train) <= config.DEBUG_SAMPLE_SIZE
    ), "Train set size exceeds debug limit"
    assert len(df_test) <= config.DEBUG_SAMPLE_SIZE, "Test set size exceeds debug limit"
    assert config.TARGET_COL in df_train.columns, "Target column missing in train set"
    assert config.ID_COL in df_train.columns, "ID column missing in train set"

    print("Data loading verified.")

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[3] Running Feature Engineering Pipeline...")

    # Process data (fit on train, transform test)
    # We pass load_cached_data=False to ensure the pipeline logic runs
    df_train_proc, df_test_proc = feature_engineering.process_data(
        df_train, df_test, load_cached_data=False
    )

    # Validation
    print(f"Processed Train Shape: {df_train_proc.shape}")

    # Check for specific engineered features
    expected_features = [
        "Euclidean_Distance_To_Hydrology",
        "Soil_Type_Index",  # Result of reverse one-hot
        "PCA_1",  # Result of PCA
    ]

    for feat in expected_features:
        if feat not in df_train_proc.columns:
            raise AssertionError(
                f"Expected feature '{feat}' not found in processed data."
            )

    # Check that original one-hot columns are removed (e.g., Soil_Type1)
    if "Soil_Type1" in df_train_proc.columns:
        raise AssertionError("One-hot column 'Soil_Type1' should have been removed.")

    print("Feature engineering verified.")

    # ---------------------------------------------------------
    # 4. Model Training (Ensemble)
    # ---------------------------------------------------------
    print("\n[4] Training Ensemble Model...")

    trainer = model_trainer.EnsembleTrainer()

    # Run CV
    models, cv_score = trainer.run_stratified_cv(df_train_proc)

    # Validation
    assert (
        len(models) == config.N_FOLDS
    ), f"Expected {config.N_FOLDS} models, got {len(models)}"
    assert isinstance(cv_score, float), "CV score should be a float"
    assert 0 <= cv_score <= 1, "CV score should be between 0 and 1"

    print(f"Training completed. CV Accuracy: {cv_score:.4f}")

    # ---------------------------------------------------------
    # 5. Inference and Submission
    # ---------------------------------------------------------
    print("\n[5] Running Inference...")

    # Initialize Inference Engine with trained models and encoder
    engine = inference.InferenceEngine(
        models=models, label_encoder=trainer.le, feature_names=trainer.feature_names
    )

    # Generate Probabilities
    probs = engine.predict_ensemble(df_test_proc)

    # Validation of probabilities
    num_classes = len(trainer.le.classes_)
    assert probs.shape == (
        len(df_test_proc),
        num_classes,
    ), f"Probability shape mismatch. Expected ({len(df_test_proc)}, {num_classes}), got {probs.shape}"

    # Generate Submission File
    submission_df = engine.save_submission(df_test_proc, probs)

    # Validation of submission
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."
    assert len(submission_df) == len(df_test_proc), "Submission row count mismatch."
    assert config.ID_COL in submission_df.columns, "Submission missing ID column."
    assert (
        config.TARGET_COL in submission_df.columns
    ), "Submission missing Target column."

    # Check a few values to ensure they are valid classes
    predicted_classes = submission_df[config.TARGET_COL].unique()
    known_classes = trainer.le.classes_  # These are the original labels (e.g., 1, 2, 7)

    # It's possible the test set predicts a subset of classes, so we check if predictions are subset of known
    if not set(predicted_classes).issubset(set(known_classes)):
        raise AssertionError(
            f"Predictions contain unknown classes: {set(predicted_classes) - set(known_classes)}"
        )

    print(f"Inference verified. Submission saved to {config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
