import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from provided library files
from library.config import Config
from library.feature_engineering import FeatureProcessor
from library.model import TaxiFareModel
from library.utils import calculate_rmse, format_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_gpu():
    """Detects and prints available GPU resources."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device_name}")
        # Note: The provided PhysicsInformedLinearModel is based on Sklearn (CPU).
        # While we detect the GPU here, the specific model class provided
        # runs on CPU. We proceed with the provided architecture.
    else:
        print("No GPU detected. Running on CPU.")


def run_failure_analysis(X_val, y_val, y_pred, feature_names):
    """
    Analyzes model performance by correlating error magnitude with features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(y_val - y_pred)

    # Compute correlations between features and error
    # X_val is a numpy array, we iterate through columns
    correlations = {}
    for idx, feature_name in enumerate(feature_names):
        # Calculate Pearson correlation
        if np.std(X_val[:, idx]) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(X_val[:, idx], errors)[0, 1]
        correlations[feature_name] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Input Features:")
    for name, corr in sorted_corrs:
        print(f"  {name}: {corr:.6f}")


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    Config.setup()
    check_gpu()

    print("Loading datasets...")
    # Load raw data
    train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # 2. Feature Engineering
    print("Processing features...")
    fp = FeatureProcessor()

    # Fit and transform training data
    # load_cached_data=True allows skipping computation if ./working/idea_1/X_train.npy exists
    X_train, y_train = fp.fit_transform(train_df, load_cached_data=True)

    # Transform validation and test data (using the scaler fitted on train)
    X_val = fp.transform(val_df, load_cached_data=True, cache_name="val")
    # Extract target for validation
    y_val = val_df["fare_amount"].values.astype(np.float32)

    X_test = fp.transform(test_df, load_cached_data=True, cache_name="test")

    # 3. Model Training
    print("Initializing and training model...")
    model = TaxiFareModel()

    # Train the model
    # We pass validation data to allow the internal early stopping logic to function
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # 4. Validation & Metrics
    print("Performing final validation...")
    val_preds = model.predict(X_val)
    final_rmse = calculate_rmse(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # 5. Failure Analysis
    run_failure_analysis(X_val, y_val, val_preds, fp.feature_cols)

    # 6. Submission
    print("Generating submission...")
    test_preds = model.predict(X_test)

    # Save submission
    format_submission(test_df["key"], test_preds)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
