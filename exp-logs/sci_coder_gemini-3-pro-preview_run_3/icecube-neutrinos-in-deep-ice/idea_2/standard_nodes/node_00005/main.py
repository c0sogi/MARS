import numpy as np
import pandas as pd
import os
import sys
import random

# Import provided library components
from library.config import SEED, FEATURE_NAMES, SUBMISSION_PATH
from library.feature_engineering import load_datasets
from library.model import VectorRegressor
from library.utils import cartesian_to_spherical


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # We limit the dataset size to ensure the baseline runs quickly (within ~2 hours).
    # 500,000 events is a sufficient sample for a robust baseline on this task.
    DEBUG_SIZE = 500000
    print(f"Loading datasets (limit={DEBUG_SIZE})...")

    # load_datasets returns: ((X_train, y_train), (X_val, y_val), (X_test, ids_test))
    # y_train/y_val are (N, 3) arrays of unit vector components (x, y, z)
    (X_train, y_train), (X_val, y_val), (X_test, ids_test) = load_datasets(
        load_cached_data=True, debug_size=DEBUG_SIZE
    )

    # 3. Train Model
    print("Initializing and training VectorRegressor...")
    model = VectorRegressor()
    model.fit(X_train, y_train, X_val, y_val)

    # 4. Validation & Metric Calculation
    print("Performing validation inference...")

    # Predict unit vectors for validation set
    # shape: (N_val, 3)
    val_pred_vectors = model.predict(X_val)

    # Calculate Angular Error
    # Formula: angle = arccos(clip(dot(true, pred), -1, 1))
    # y_val contains true unit vectors
    dot_products = np.sum(y_val * val_pred_vectors, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products)

    mean_angular_error = np.mean(angular_errors)

    # Required Output Format
    print(f"Final Validation Metric: {mean_angular_error}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Correlate error magnitude with input features
    df_val_features = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    df_val_features["error_magnitude"] = angular_errors

    # Calculate correlations
    correlations = df_val_features.corr()["error_magnitude"].drop("error_magnitude")
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 features correlated with angular error:")
    print(top_correlations)

    # 6. Test Inference & Submission
    print("\nGenerating test predictions...")
    test_pred_vectors = model.predict(X_test)

    # Convert Cartesian predictions back to Spherical coordinates (Azimuth, Zenith)
    test_azimuth, test_zenith = cartesian_to_spherical(
        test_pred_vectors[:, 0], test_pred_vectors[:, 1], test_pred_vectors[:, 2]
    )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"event_id": ids_test, "azimuth": test_azimuth, "zenith": test_zenith}
    )

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
