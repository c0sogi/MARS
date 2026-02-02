import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Import from the provided library
from library.config import Config, set_seed
from library.utils import setup_logger, load_object
from library.data_loader import load_and_process_data
from library.feature_extraction import FeaturePreprocessor
from library.model_factory import create_pipeline
from library.trainer import run_training
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    print("=== Starting Demonstration of Pizza Request Prediction Pipeline ===")

    # 1. Setup and Configuration Optimization
    # We modify Config parameters to ensure the demo runs very fast.
    print("\n[1] Configuring environment for rapid demonstration...")
    set_seed(42)

    # Monkey-patch Config for speed
    Config.N_FOLDS = 2  # Reduce folds from 5 to 2
    Config.N_BAGGING_ESTIMATORS = 2  # Reduce ensemble size
    Config.LR_C_RANGE = [1.0]  # Reduce GridSearch space to a single value
    Config.DEBUG_SAMPLES = 50  # Use a small subset of data

    # Clean up working directory for a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("Configuration optimized: Folds=2, Bagging=2, Samples=50.")

    # 2. Data Loading Demonstration
    print("\n[2] Testing Data Loader...")
    # We force load_cached_data=False to verify the processing logic
    df_train = load_and_process_data(split="train", load_cached_data=False, debug=True)

    # Validation
    assert isinstance(df_train, pd.DataFrame), "Data loader should return a DataFrame"
    assert (
        len(df_train) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} samples"
    assert "request_id" in df_train.columns, "Missing request_id column"
    assert "text_combined" in df_train.columns, "Missing text_combined column"
    assert "label" in df_train.columns, "Missing label column for training data"
    print("Data Loader verification passed.")

    # 3. Feature Extraction Demonstration
    print("\n[3] Testing Feature Extraction...")
    preprocessor = FeaturePreprocessor()

    # This will compute embeddings and cache them
    data_train = preprocessor.get_data(split="train", load_cached=False, debug=True)

    X_train = data_train["X"]
    y_train = data_train["y"]
    feature_slices = data_train["feature_slices"]

    # Validation
    assert isinstance(X_train, np.ndarray), "Features should be a numpy array"
    assert len(X_train) == Config.DEBUG_SAMPLES, "Feature count mismatch"
    assert len(y_train) == Config.DEBUG_SAMPLES, "Label count mismatch"
    assert "primary" in feature_slices, "Missing primary view slice"
    assert "aux" in feature_slices, "Missing auxiliary view slice"
    assert "meta" in feature_slices, "Missing metadata view slice"

    # Check dimensions
    # Primary (MiniLM-L6) is 384d, Aux (MPNet) is 768d, Meta is 10d
    # Note: Aux view is reduced to 50d (Config.AUX_PCA_COMPONENTS) inside the pipeline,
    # but the raw feature matrix contains the full embeddings (768d).
    expected_dim = 384 + 768 + 10
    assert (
        X_train.shape[1] == expected_dim
    ), f"Expected {expected_dim} features, got {X_train.shape[1]}"
    print("Feature Extraction verification passed.")

    # 4. Model Factory Demonstration
    print("\n[4] Testing Model Pipeline Creation...")
    pipeline = create_pipeline(
        feature_slices=feature_slices,
        pca_components=5,  # Small PCA for demo
        n_bagging_estimators=2,
    )

    assert isinstance(
        pipeline, Pipeline
    ), "Factory should return a scikit-learn Pipeline"

    # Test fitting on the small subset
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict_proba(X_train)[:, 1]

    assert len(preds) == len(y_train), "Prediction length mismatch"
    assert preds.min() >= 0.0 and preds.max() <= 1.0, "Probabilities out of range"
    print("Model Factory verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Testing Full Training Loop (Cross-Validation)...")
    # This runs the stratified K-Fold training and saves models
    model_paths = run_training(debug=True)

    # Validation
    assert isinstance(model_paths, list), "run_training should return a list of paths"
    assert len(model_paths) == Config.N_FOLDS, f"Expected {Config.N_FOLDS} models"
    for path in model_paths:
        assert os.path.exists(path), f"Model file not found: {path}"
        # Verify we can load it back
        loaded_model = load_object(path)
        assert hasattr(
            loaded_model, "predict_proba"
        ), "Saved object is not a valid classifier"

    print("Training Loop verification passed.")

    # 6. Inference Demonstration
    print("\n[6] Testing Inference and Submission Generation...")
    # Generate submission using the trained models
    # We force load_cached_data=False to ensure test features are computed
    generate_submission(model_paths=model_paths, load_cached_data=False, debug=True)

    # Validation
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    assert "request_id" in df_sub.columns, "Submission missing request_id"
    assert (
        "requester_received_pizza" in df_sub.columns
    ), "Submission missing probability column"
    assert (
        len(df_sub) == Config.DEBUG_SAMPLES
    ), "Submission row count mismatch (Debug Mode)"

    print("Inference verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
