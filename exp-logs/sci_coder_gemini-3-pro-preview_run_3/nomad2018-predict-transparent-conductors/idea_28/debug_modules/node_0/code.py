import os
import sys
import numpy as np
import pandas as pd
import logging

# Import library components
from library.config import (
    TRAIN_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    TARGET_COLS,
)
from library.structure_utils import load_xyz, get_neighbor_list, get_cell_parameters
from library.features import extract_features_from_atoms
from library.data_pipeline import process_dataset
from library.model import DualTargetRegressor
from library.utils import calculate_rmsle, save_submission, setup_logger

# Set random seeds for reproducibility
np.random.seed(42)


def main():
    # Setup logging to console
    logger = setup_logger("demo_script")
    logger.info("Starting demonstration script...")

    # ---------------------------------------------------------
    # 1. Demonstrate Structure Loading and Utils
    # ---------------------------------------------------------
    logger.info("--- Step 1: Structure Loading & Utils ---")

    # Load training metadata to get a valid file path
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Training metadata not found at {TRAIN_METADATA_PATH}")

    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)

    # Pick the first example
    sample_row = df_train_meta.iloc[0]
    file_path = sample_row["file_path"]
    sample_id = sample_row["id"]

    logger.info(f"Loading structure for ID {sample_id} from {file_path}")

    # Load atoms object
    atoms = load_xyz(file_path)

    # Verify it is an ASE Atoms object
    assert hasattr(atoms, "get_positions"), "Loaded object is not an ASE Atoms object"
    logger.info(f"Successfully loaded structure with {len(atoms)} atoms.")

    # Test neighbor list calculation
    idx_i, idx_j, dists, vecs = get_neighbor_list(atoms, cutoff=3.0)
    assert len(idx_i) == len(idx_j) == len(dists), "Neighbor list arrays mismatch"
    logger.info(f"Computed neighbor list: {len(dists)} interactions found.")

    # Test cell parameters extraction
    cell_params = get_cell_parameters(atoms)
    assert "volume" in cell_params, "Cell parameters missing volume"
    logger.info(f"Cell Volume: {cell_params['volume']:.4f} A^3")

    # ---------------------------------------------------------
    # 2. Demonstrate Feature Extraction
    # ---------------------------------------------------------
    logger.info("\n--- Step 2: Feature Extraction ---")

    # Extract features for a single atom
    features = extract_features_from_atoms(atoms)

    # Verify some expected features exist
    expected_keys = ["vol_per_atom", "density", "BVS_global_mean", "CN_global_mean"]
    for key in expected_keys:
        assert key in features, f"Expected feature {key} missing from extraction"

    logger.info(f"Extracted {len(features)} features for sample {sample_id}.")

    # ---------------------------------------------------------
    # 3. Demonstrate Data Pipeline (Small Subset)
    # ---------------------------------------------------------
    logger.info("\n--- Step 3: Data Pipeline (Subset Processing) ---")

    # Use a small subset of data for speed (e.g., 50 samples)
    subset_size = 50
    df_subset_meta = df_train_meta.head(subset_size).copy()

    logger.info(f"Processing a subset of {subset_size} samples...")
    # process_dataset returns a DataFrame with features and 'id'
    df_features = process_dataset(df_subset_meta)

    assert len(df_features) == subset_size, "Feature dataframe size mismatch"
    assert "id" in df_features.columns, "ID column missing from feature dataframe"

    # Merge with targets for training
    df_merged = df_features.merge(
        df_subset_meta[["id"] + TARGET_COLS], on="id", how="inner"
    )

    # Prepare X and y
    y_train = df_merged[TARGET_COLS]
    X_train = df_merged.drop(columns=["id"] + TARGET_COLS)

    logger.info(f"Training Data Shape: X={X_train.shape}, y={y_train.shape}")

    # ---------------------------------------------------------
    # 4. Demonstrate Model Training (XGBoost)
    # ---------------------------------------------------------
    logger.info("\n--- Step 4: Model Training ---")

    # Define fast hyperparameters for demonstration
    fast_params = {
        "n_estimators": 10,  # Small number of trees for speed
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "n_jobs": 1,  # Avoid overhead for small data
        "random_state": 42,
    }

    # Split into train/val for demonstration
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    # Initialize and fit the dual target regressor
    model = DualTargetRegressor(params=fast_params)
    model.fit(X_tr, y_tr, X_val, y_val)

    # Predict
    preds = model.predict(X_val)

    # Verify predictions
    assert preds.shape == y_val.shape, "Prediction shape mismatch"
    assert (preds >= 0).all().all(), "Predictions should be non-negative (energy)"

    # Calculate Metric
    score = calculate_rmsle(y_val, preds)
    logger.info(f"Model RMSLE on subset validation: {score:.4f}")

    # ---------------------------------------------------------
    # 5. Demonstrate Submission Generation
    # ---------------------------------------------------------
    logger.info("\n--- Step 5: Submission Generation ---")

    # Load test metadata (small subset)
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    df_test_meta = pd.read_csv(TEST_METADATA_PATH).head(10)
    test_ids = df_test_meta["id"].values

    # Process test features
    logger.info(f"Processing {len(df_test_meta)} test samples...")
    df_test_features = process_dataset(df_test_meta)
    X_test = df_test_features.drop(columns=["id"])

    # Predict on test
    test_preds = model.predict(X_test)

    # Define output path
    output_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # Save submission
    save_submission(
        ids=test_ids,
        formation_energy=test_preds["formation_energy_ev_natom"],
        bandgap_energy=test_preds["bandgap_energy_ev"],
        filename=output_path,
    )

    # Verify file creation
    if os.path.exists(output_path):
        logger.info(f"Submission file successfully created at {output_path}")
        # Print first few lines
        with open(output_path, "r") as f:
            head = [next(f) for _ in range(3)]
        logger.info("File head:\n" + "".join(head))
    else:
        raise RuntimeError("Submission file was not created.")

    logger.info("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
