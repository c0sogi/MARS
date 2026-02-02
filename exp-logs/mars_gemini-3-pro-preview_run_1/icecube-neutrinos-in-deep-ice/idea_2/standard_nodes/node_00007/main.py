import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import SUBMISSION_PATH, SEED
from library.data_loader import load_train_dataset, load_val_dataset, load_test_dataset
from library.model import DirectionalLGBM
from library.utils import angular_dist_score, cartesian_to_spherical


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Set random seed
    set_seed(SEED)

    print("Starting baseline pipeline...")

    # ==========================================
    # 1. Load Training Data
    # ==========================================
    # We limit the number of rows to ensure the baseline runs quickly (approx 2 hours limit)
    # 300,000 samples is sufficient for a robust LightGBM baseline
    print("Loading training data (subset)...")
    X_train, y_train = load_train_dataset(load_cached_data=True, debug_n_rows=300000)
    print(f"Training data loaded. Shape: {X_train.shape}")

    # ==========================================
    # 2. Load Validation Data
    # ==========================================
    # We limit validation to 60,000 samples for speed
    print("Loading validation data (subset)...")
    X_val, y_val = load_val_dataset(load_cached_data=True, debug_n_rows=60000)
    print(f"Validation data loaded. Shape: {X_val.shape}")

    # ==========================================
    # 3. Train Model
    # ==========================================
    print("Initializing DirectionalLGBM model...")
    model = DirectionalLGBM()

    print("Training model with early stopping...")
    # The model class handles the separate training of x, y, z regressors
    model.fit(X_train, y_train, X_val, y_val)

    print("Saving trained model...")
    model.save()

    # ==========================================
    # 4. Validation Evaluation
    # ==========================================
    print("Evaluating on validation set...")
    # Predict unit direction vectors (N, 3)
    val_vectors = model.predict(X_val)

    # Convert predicted vectors to spherical coordinates for metric calculation
    pred_azimuth, pred_zenith = cartesian_to_spherical(
        val_vectors[:, 0], val_vectors[:, 1], val_vectors[:, 2]
    )

    # Construct DataFrame for prediction
    y_pred_val = pd.DataFrame({"azimuth": pred_azimuth, "zenith": pred_zenith})

    # Convert true Cartesian targets back to spherical coordinates
    # y_val contains 'x', 'y', 'z' components
    true_azimuth, true_zenith = cartesian_to_spherical(
        y_val["x"], y_val["y"], y_val["z"]
    )

    y_true_val = pd.DataFrame({"azimuth": true_azimuth, "zenith": true_zenith})

    # Calculate Mean Angular Error
    score = angular_dist_score(y_true_val, y_pred_val)
    print(f"Final Validation Metric: {score}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("Performing failure analysis...")

    # Calculate the angular error for each individual event
    # Dot product of true and predicted unit vectors
    dot_product = (
        (val_vectors[:, 0] * y_val["x"])
        + (val_vectors[:, 1] * y_val["y"])
        + (val_vectors[:, 2] * y_val["z"])
    )

    # Clip to valid range [-1, 1] to avoid numerical errors with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Error is the angle between vectors
    errors = np.arccos(dot_product)

    # Create a temporary DataFrame to analyze correlations
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate correlation of features with the error magnitude
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    print(
        "Feature correlations with error magnitude (positive means higher feature value -> higher error):"
    )
    print(correlations.sort_values(ascending=False))

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("Loading full test data and generating predictions...")

    # Load the ENTIRE test set for submission
    # debug_n_rows=None ensures all test events are processed
    X_test, test_ids = load_test_dataset(load_cached_data=True, debug_n_rows=None)
    print(f"Test data loaded. Shape: {X_test.shape}")

    print("Predicting on test set...")
    test_vectors = model.predict(X_test)

    # Convert predictions to spherical coordinates
    test_azimuth, test_zenith = cartesian_to_spherical(
        test_vectors[:, 0], test_vectors[:, 1], test_vectors[:, 2]
    )

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"event_id": test_ids, "azimuth": test_azimuth, "zenith": test_zenith}
    )

    # Ensure columns are in the correct order
    submission = submission[["event_id", "azimuth", "zenith"]]

    # Save to CSV
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Submission generated successfully.")


if __name__ == "__main__":
    main()
