import pandas as pd
import numpy as np
import sys
import os

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_rmse, save_submission
from library.feature_extractor import extract_features
from library.regressor import RidgeRegressor


def perform_failure_analysis(val_meta, val_y, val_preds):
    """
    Analyzes the correlation between prediction errors and metadata features.

    Args:
        val_meta (np.array): Validation metadata features.
        val_y (np.array): Ground truth targets.
        val_preds (np.array): Model predictions.
    """
    # Calculate absolute error
    errors = np.abs(val_y - val_preds)

    # Define feature names corresponding to the order in Data Loader
    # Based on PawpularityDataset class in library/data_loader.py
    feature_names = [
        "Focus",  # Mapped from 'Subject Focus' or 'Focus'
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # Check if dimensions match
    if val_meta.shape[1] != len(feature_names):
        print(
            f"Warning: Metadata feature count ({val_meta.shape[1]}) does not match expected names ({len(feature_names)})."
        )
        # Generate generic names if mismatch to prevent crash
        feature_names = [f"Feature_{i}" for i in range(val_meta.shape[1])]

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(val_meta, columns=feature_names)
    df_analysis["Error_Magnitude"] = errors

    # Calculate correlations
    correlations = df_analysis.corr()["Error_Magnitude"].drop("Error_Magnitude")

    print("\nCorrelation between Absolute Error and Metadata Features:")
    print(correlations.sort_values(ascending=False))


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility across all libraries
    set_seed(Config.SEED)

    print("==========================================")
    print("      Pawpularity Prediction Baseline     ")
    print("==========================================")

    # 2. Feature Extraction
    # This step loads images, runs MobileNetV2 (on GPU if available),
    # and returns embeddings + metadata.
    # It handles caching automatically via the library implementation.
    print("\n[Step 1/4] Extracting Features...")
    data = extract_features(load_cached_data=True)

    # Unpack data
    train_img, train_meta, train_y = data["train"]
    val_img, val_meta, val_y = data["val"]
    test_img, test_meta, test_ids = data["test"]

    print(f"Train set shape: Images {train_img.shape}, Meta {train_meta.shape}")
    print(f"Val set shape:   Images {val_img.shape}, Meta {val_meta.shape}")
    print(f"Test set shape:  Images {test_img.shape}, Meta {test_meta.shape}")

    # 3. Model Training
    print("\n[Step 2/4] Training Ridge Regressor...")
    # Initialize model with config parameters
    regressor = RidgeRegressor(alpha=Config.RIDGE_ALPHA, random_state=Config.SEED)

    # Fit on training data (concatenates image and meta features internally)
    regressor.fit(train_img, train_meta, train_y)

    # 4. Validation and Analysis
    print("\n[Step 3/4] Validating and Analyzing Failures...")

    # Generate validation predictions
    # Note: regressor.predict automatically clips predictions to [1, 100]
    val_preds = regressor.predict(val_img, val_meta)

    # Compute and print metric
    val_rmse = compute_rmse(val_y, val_preds)
    # REQUIRED FORMAT: Final Validation Metric: <value>
    print(f"Final Validation Metric: {val_rmse}")

    # Perform failure analysis
    perform_failure_analysis(val_meta, val_y, val_preds)

    # 5. Submission
    print("\n[Step 4/4] Generating Submission...")

    # Generate test predictions
    test_preds = regressor.predict(test_img, test_meta)

    # Save to CSV using the provided utility
    save_submission(test_ids, test_preds, Config.SUBMISSION_PATH)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
