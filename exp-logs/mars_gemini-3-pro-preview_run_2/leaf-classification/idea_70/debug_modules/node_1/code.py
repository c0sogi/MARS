import os
import shutil
import numpy as np
import pandas as pd
import warnings
import sys

# Import from provided libraries
from library.utils import set_seed
from library.features import DatasetLoader
import library.preprocessing
import importlib

importlib.reload(library.preprocessing)
from library.preprocessing import Preprocessor
from library.models import GreedyEnsembleSelector
from library.pipeline import run_smpge_pipeline

# Configuration for demonstration
DEMO_CACHE_DIR = "./working/idea_70"
MAX_ENSEMBLE_SIZE = 2  # Reduced for speed
TOLERANCE = 1e-4


def main():
    # 1. Setup
    print("Initializing Demonstration...")
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Set seed for reproducibility
    set_seed(42)

    # Clean up specific cache directory to demonstrate fresh data processing
    if os.path.exists(DEMO_CACHE_DIR):
        print(f"Cleaning cache directory: {DEMO_CACHE_DIR}")
        shutil.rmtree(DEMO_CACHE_DIR)

    # -------------------------------------------------------------------------
    # 2. Demonstrate Preprocessing (Data Loading + Feature Extraction + Transformation)
    # -------------------------------------------------------------------------
    print("\n[Demo] Running Preprocessor...")
    preprocessor = Preprocessor()

    # load_cached_data=False forces the DatasetLoader to read images and extract features
    # This demonstrates library.features.DatasetLoader and library.features.ImageProcessor
    data = preprocessor.get_data(load_cached_data=False)

    # Validation of Data Dictionary
    required_keys = [
        "X_train_global",
        "X_val_global",
        "X_test_global",
        "X_train_stratified",
        "y_train",
        "y_val",
        "test_ids",
    ]
    for key in required_keys:
        if key not in data:
            raise AssertionError(f"Missing key in preprocessed data: {key}")

    # Validate Shapes
    # Global view should have 192 features (64 margin + 64 shape + 64 texture)
    n_samples_train = data["y_train"].shape[0]
    assert data["X_train_global"].shape == (
        n_samples_train,
        192,
    ), f"Expected X_train_global shape ({n_samples_train}, 192), got {data['X_train_global'].shape}"

    print("  - Data loaded and transformed successfully.")
    print(f"  - Training Samples: {n_samples_train}")
    print(f"  - Feature Count (Global View): {data['X_train_global'].shape[1]}")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Model Training (GreedyEnsembleSelector)
    # -------------------------------------------------------------------------
    print("\n[Demo] Training GreedyEnsembleSelector...")

    # Initialize selector with small ensemble size for speed
    selector = GreedyEnsembleSelector(
        max_ensemble_size=MAX_ENSEMBLE_SIZE, tolerance=TOLERANCE
    )

    # Fit on the preprocessed data
    selector.fit(data)

    # Validate that experts were selected
    if not selector.selected_experts:
        raise AssertionError("Selector failed to select any experts.")

    print(f"  - Selected {len(selector.selected_experts)} experts.")

    # Generate predictions
    print("  - Generating predictions...")
    preds = selector.predict(data)

    # Validate Predictions
    n_test_samples = data["test_ids"].shape[0]
    n_classes = len(np.unique(data["y_train"]))

    assert preds.shape == (
        n_test_samples,
        n_classes,
    ), f"Prediction shape mismatch. Expected ({n_test_samples}, {n_classes}), got {preds.shape}"

    # Check row sums (should be approx 1.0)
    row_sums = preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0"

    print("  - Prediction logic verified.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[Demo] Running Full SMPGE Pipeline...")

    # We use load_cached_data=True here to reuse the data we just processed in Step 2.
    # This demonstrates the caching mechanism in library.preprocessing.
    submission_df = run_smpge_pipeline(
        load_cached_data=True, max_ensemble_size=MAX_ENSEMBLE_SIZE, tolerance=TOLERANCE
    )

    # -------------------------------------------------------------------------
    # 5. Final Validation
    # -------------------------------------------------------------------------
    print("\n[Demo] Validating Submission...")

    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_check = pd.read_csv(submission_path)

    # Check ID column
    if "id" not in df_check.columns:
        raise AssertionError("Submission missing 'id' column.")

    # Check number of rows
    assert len(df_check) == 99, f"Expected 99 rows in submission, got {len(df_check)}"

    # Check number of columns (1 ID + 99 Classes = 100)
    assert (
        len(df_check.columns) == 100
    ), f"Expected 100 columns, got {len(df_check.columns)}"

    print("  - Submission file passed validation checks.")
    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    main()
