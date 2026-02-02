import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

# Import provided library components
from library.config import (
    set_seed,
    SUBMISSION_FILE_PATH,
    ID_COL,
    METADATA_DIR,
    GEOMETRIC_FEATURES,
)
from library.utils import compute_config_hash
from library.image_processing import extract_visual_features_for_row
from library.data_loader import DataManager
from library.model import HighPrecisionOAS


def main():
    # 1. Setup and Reproducibility
    print("Initializing demonstration...")
    set_seed(42)

    # 2. Demonstrate Low-Level Image Processing
    # We will manually extract features for one image to verify the logic works.
    print("\n--- Demonstrating Image Processing (Unit Test) ---")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df_sample = pd.read_csv(train_meta_path).head(1)
        row = df_sample.iloc[0]

        print(f"Extracting features for Image ID: {row[ID_COL]}")
        features = extract_visual_features_for_row(row)

        # Verification
        assert isinstance(features, dict), "Output must be a dictionary"
        assert all(
            k in features for k in GEOMETRIC_FEATURES
        ), "Missing geometric features"
        print("Feature extraction successful. Sample features:")
        print(f"  Area: {features['Area']:.2f}")
        print(f"  Solidity: {features['Solidity']:.4f}")
    else:
        print("Warning: Metadata file not found, skipping unit test.")

    # 3. Demonstrate Data Loading Pipeline
    # The DataManager handles loading, merging, caching, and preprocessing (Scaling/Yeo-Johnson)
    print("\n--- Demonstrating Data Pipeline ---")
    data_manager = DataManager(load_cached_data=True)

    # Load data (this will trigger processing if cache is missing)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data_manager.load_data()

    print(f"Data Loaded Successfully:")
    print(f"  Training Set:   {X_train.shape}")
    print(f"  Validation Set: {X_val.shape}")
    print(f"  Test Set:       {X_test.shape}")
    print(f"  Num Classes:    {len(classes)}")

    # Assertions to ensure data integrity
    assert X_train.dtype == np.float64, "Data must be float64 for high precision model"
    assert not np.isnan(X_train).any(), "Training data contains NaNs"
    assert (
        len(test_ids) == X_test.shape[0]
    ), "Mismatch between test IDs and test features"

    # 4. Demonstrate Model Training
    print("\n--- Demonstrating Model Training (HighPrecisionOAS) ---")
    model = HighPrecisionOAS()

    print("Fitting model...")
    model.fit(X_train, y_train)

    # Check if model parameters are populated
    assert model.means_ is not None, "Model means not computed"
    assert model.covariance_ is not None, "Model covariance not computed"

    # 5. Validation
    print("\n--- Validating Model ---")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # We pass labels explicitly to handle cases where a batch might miss a class
    metric_score = log_loss(y_val, val_probs, labels=range(len(classes)))
    print(f"Validation Log Loss: {metric_score:.5f}")

    # Verify probabilities
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0"
    assert (
        val_probs.min() >= 0 and val_probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    # 6. Generate Submission
    print("\n--- Generating Submission ---")
    test_probs = model.predict_proba(X_test)

    # Format submission DataFrame
    # Columns must be the class names
    submission_df = pd.DataFrame(test_probs, columns=classes)

    # Insert ID column at the beginning
    submission_df.insert(0, ID_COL, test_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_FILE_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_FILE_PATH}")

    # Final check on the saved file
    saved_df = pd.read_csv(SUBMISSION_FILE_PATH)
    print(f"Saved submission shape: {saved_df.shape}")
    assert saved_df.shape[0] == len(test_ids), "Submission row count mismatch"
    assert saved_df.shape[1] == len(classes) + 1, "Submission column count mismatch"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
