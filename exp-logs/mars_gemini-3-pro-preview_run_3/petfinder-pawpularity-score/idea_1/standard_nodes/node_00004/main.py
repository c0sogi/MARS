import pandas as pd
import numpy as np
import sys
import os

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_rmse, save_submission
from library.data_loader import get_dataloaders
from library.regressor import FineTuningRegressor


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

    # 2. Data Loading
    print("\n[Step 1/4] Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # 3. Model Training
    print("\n[Step 2/4] Fine-tuning MobileNetV2...")
    regressor = FineTuningRegressor()
    regressor.fit(train_loader, val_loader)

    # 4. Validation and Analysis
    print("\n[Step 3/4] Validating and Analyzing Failures...")

    # Generate validation predictions
    val_preds = regressor.predict(val_loader)

    # Extract Ground Truth and Metadata for analysis
    # We need to iterate the loader to get them in the same order as predictions
    val_y_list = []
    val_meta_list = []

    # Note: val_loader has shuffle=False, so order is preserved
    for _, meta, targets, _ in val_loader:
        val_y_list.append(targets.numpy())
        val_meta_list.append(meta.numpy())

    val_y = np.concatenate(val_y_list)
    val_meta = np.concatenate(val_meta_list)

    # Compute and print metric
    val_rmse = compute_rmse(val_y, val_preds)
    # REQUIRED FORMAT: Final Validation Metric: <value>
    print(f"Final Validation Metric: {val_rmse}")

    # Perform failure analysis
    perform_failure_analysis(val_meta, val_y, val_preds)

    # 5. Submission
    print("\n[Step 4/4] Generating Submission...")

    # Generate test predictions
    test_preds = regressor.predict(test_loader)

    # Extract IDs
    test_ids_list = []
    for _, _, _, ids in test_loader:
        test_ids_list.extend(ids)

    save_submission(test_ids_list, test_preds, Config.SUBMISSION_PATH)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
