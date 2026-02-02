import os
import sys
import shutil
import pandas as pd
import numpy as np
import ase.io

# Import from the provided library files
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    RANDOM_SEED,
    TARGET_COLS,
    set_seed,
)
from library.data_pipeline import DatasetLoader
from library.feature_engineering import process_structure
from library.geometry_descriptors import (
    calculate_bond_valences,
    calculate_weighted_bond_angles,
    calculate_structural_metrics,
)
from library.training import DualTargetRegressor, train_and_evaluate

# Ensure reproducibility
set_seed(RANDOM_SEED)


def main():
    print("Starting demonstration of library usage...")

    # ==========================================
    # 1. Data Pipeline & Feature Engineering
    # ==========================================
    print("\n--- Demonstrating Data Pipeline (Subset) ---")

    # Initialize Loader
    loader = DatasetLoader(metadata_dir=METADATA_DIR, working_dir=WORKING_DIR)

    # Load raw metadata
    full_train_metadata = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))

    # Optimization: Use only a small subset (e.g., 10 samples) for demonstration speed
    subset_size = 10
    subset_metadata = full_train_metadata.head(subset_size).copy()
    print(f"Processing subset of {subset_size} samples...")

    # Generate features for the subset using the pipeline's parallel computation method
    # We bypass _generate_or_load_features to avoid caching/loading the full dataset
    features_df = loader._compute_features_parallel(subset_metadata, n_jobs=2)

    # Merge with metadata to get targets and original columns
    combined_df = pd.merge(subset_metadata, features_df, on="id", how="left")

    # Assertions
    assert len(combined_df) == subset_size, "Combined DataFrame size mismatch"
    assert "geo_volume" in combined_df.columns, "Feature 'geo_volume' missing"
    assert "bvs_gii" in combined_df.columns, "Feature 'bvs_gii' missing"
    # Check RDF columns exist (dynamic names)
    rdf_cols = [c for c in combined_df.columns if "rdf_" in c]
    assert len(rdf_cols) > 0, "No RDF features generated"

    print("Features generated successfully.")
    print(f"Feature matrix shape: {combined_df.shape}")

    # ==========================================
    # 2. Geometry Descriptors (Low-level)
    # ==========================================
    print("\n--- Demonstrating Geometry Descriptors ---")

    # Pick one geometry file from the subset
    sample_rel_path = subset_metadata.iloc[0]["file_path"]
    sample_full_path = os.path.join(INPUT_DIR, sample_rel_path)

    print(f"Analyzing structure: {sample_rel_path}")
    atoms = ase.io.read(sample_full_path)

    # A. Bond Valences
    bvs_data = calculate_bond_valences(atoms)
    print("Bond Valence Data keys:", bvs_data.keys())
    assert "scalar_bvs" in bvs_data
    assert "vector_bvs" in bvs_data
    assert "gii" in bvs_data
    assert len(bvs_data["scalar_bvs"]) == len(atoms), "BVS array length mismatch"

    # B. Weighted Bond Angles
    angle_data = calculate_weighted_bond_angles(atoms)
    print("Bond Angle Data keys:", angle_data.keys())
    assert "M_centered_angles" in angle_data
    assert "O_centered_angles" in angle_data

    # C. Structural Metrics (ECoN, RDF)
    struct_data = calculate_structural_metrics(atoms)
    print("Structural Metrics keys:", struct_data.keys())
    assert "econ" in struct_data
    assert "rdf" in struct_data
    assert (
        "Al-O" in struct_data["rdf"]
    ), "RDF for Al-O missing (assuming Al exists or empty key created)"

    print("Geometry descriptors verified.")

    # ==========================================
    # 3. Model Training (XGBoost)
    # ==========================================
    print("\n--- Demonstrating Model Training ---")

    # Prepare data
    # Drop non-numeric columns for training (except targets)
    # Identify numeric columns
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()

    # Targets
    y = combined_df[TARGET_COLS]

    # Features: Remove targets, id, and any other non-feature numeric cols if necessary
    # The pipeline usually handles cleaning, but here we do a quick selection
    drop_cols = TARGET_COLS + ["id"]
    X = combined_df[numeric_cols].drop(
        columns=[c for c in drop_cols if c in numeric_cols]
    )

    # Handle NaNs (simple fill for demo)
    X = X.fillna(0)

    # Split into train/val (very small split for demo)
    X_train = X.iloc[:8]
    y_train = y.iloc[:8]
    X_val = X.iloc[8:]
    y_val = y.iloc[8:]

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Initialize Regressor with low estimators for speed
    # Using 'tree_method': 'hist' is standard in config, usually fast.
    fast_params = {
        "n_estimators": 5,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_jobs": 1,
        "random_state": RANDOM_SEED,
    }

    # A. Direct usage of DualTargetRegressor
    regressor = DualTargetRegressor(params=fast_params)
    regressor.fit(X_train, y_train, verbose=False)

    preds = regressor.predict(X_val)
    print("Predictions:\n", preds)

    # Assertions on predictions
    assert preds.shape == (len(X_val), 2), "Prediction shape mismatch"
    assert (preds.values >= 0).all(), "Predictions should be non-negative (energies)"
    assert list(preds.columns) == TARGET_COLS

    # B. Usage of train_and_evaluate wrapper
    print("\nRunning train_and_evaluate wrapper...")
    model, metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val, params=fast_params, n_estimators=5
    )

    print("Evaluation Metrics:", metrics)
    assert "rmsle_formation_energy_ev_natom" in metrics
    assert "rmsle_bandgap_energy_ev" in metrics

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
