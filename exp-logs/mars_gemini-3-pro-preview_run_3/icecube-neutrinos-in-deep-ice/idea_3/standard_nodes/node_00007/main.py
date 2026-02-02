import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import TRAIN_META_PATH, VAL_META_PATH, SEED, FEATURE_NAMES
from library.trainer import Trainer
from library.utils import (
    angular_dist_score,
    cartesian_to_spherical,
    spherical_to_cartesian,
)


def main():
    # ---------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------
    print("Initializing Pipeline...")
    np.random.seed(SEED)

    # Initialize the Trainer (handles feature extraction and model wrapping)
    trainer = Trainer()

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    # We limit the sample size to ensure the script completes within the 2-hour limit.
    # The feature extraction process involves Python loops which are time-consuming
    # for the full 95M train / 23M val datasets.
    TRAIN_SAMPLE_SIZE = 500000
    VAL_SAMPLE_SIZE = 500000

    print(f"Loading Training Data (Subset: {TRAIN_SAMPLE_SIZE})...")
    X_train, y_train, _ = trainer.load_dataset(
        TRAIN_META_PATH,
        mode="train",
        sample_size=TRAIN_SAMPLE_SIZE,
        load_cached_data=True,
    )

    print(f"Loading Validation Data (Subset: {VAL_SAMPLE_SIZE})...")
    X_val, y_val, val_ids = trainer.load_dataset(
        VAL_META_PATH, mode="train", sample_size=VAL_SAMPLE_SIZE, load_cached_data=True
    )

    # ---------------------------------------------------------
    # 3. Training
    # ---------------------------------------------------------
    print("Training Model...")
    # Train the directional LightGBM model (3 independent regressors for x, y, z)
    trainer.model.fit(X_train, y_train, X_val, y_val)

    # Save the trained model artifacts
    trainer.model.save(trainer.model_path)

    # ---------------------------------------------------------
    # 4. Evaluation
    # ---------------------------------------------------------
    print("Evaluating on Validation Set...")
    pred_azimuth, pred_zenith = trainer.model.predict(X_val)

    # Convert ground truth Cartesian targets back to Spherical for metric calculation
    # y_val columns are [target_x, target_y, target_z]
    true_azimuth, true_zenith = cartesian_to_spherical(
        y_val[:, 0], y_val[:, 1], y_val[:, 2]
    )

    # Compute Metric
    metric = angular_dist_score(true_azimuth, true_zenith, pred_azimuth, pred_zenith)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {metric}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate angular error for each event
    # We compute the dot product between true and predicted unit vectors
    ux, uy, uz = spherical_to_cartesian(true_azimuth, true_zenith)
    vx, vy, vz = spherical_to_cartesian(pred_azimuth, pred_zenith)

    # Dot product clamped to [-1, 1]
    dot_prod = np.clip(ux * vx + uy * vy + uz * vz, -1.0, 1.0)
    errors = np.arccos(dot_prod)

    # Create a DataFrame to analyze correlations
    # We use a subset for correlation analysis if the validation set is large,
    # though 500k is manageable.
    df_analysis = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    df_analysis["error_magnitude"] = errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )

    print("\n--- Failure Analysis Results ---")
    print("Top 5 Features Positively Correlated with Error (High value -> High Error):")
    print(correlations.head(5))

    print(
        "\nTop 5 Features Negatively Correlated with Error (High value -> Low Error):"
    )
    print(correlations.tail(5))

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 1.5329564173900305

    if metric < THRESHOLD:
        print(f"\nValidation metric {metric} is better than threshold {THRESHOLD}.")
        print("Generating submission for Test Set...")
        # Generates predictions on the full test set and saves to ./submission/submission.csv
        trainer.generate_submission(load_cached_data=True)
    else:
        print(f"\nValidation metric {metric} did not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
