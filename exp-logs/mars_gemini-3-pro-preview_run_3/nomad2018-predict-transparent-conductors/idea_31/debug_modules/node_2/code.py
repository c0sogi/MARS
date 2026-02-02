import os
import sys
import numpy as np
import pandas as pd
import ase
from ase import Atoms

# Import from the provided library files
from library.data_loader import load_metadata, read_structure
from library.descriptors_electrostatics import ElectrostaticsCalculator
from library.descriptors_geometry import GeometryCalculator
from library.feature_engineering import FeaturePipeline
from library.model_handler import EnergyPredictor

# Set random seed for reproducibility
np.random.seed(42)


def main():
    print("=== Starting Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Demonstrate Data Loading
    # -------------------------------------------------------------------------
    print("--- 1. Data Loader ---")

    # Load metadata for all splits
    train_meta = load_metadata("train")
    val_meta = load_metadata("val")
    test_meta = load_metadata("test")

    print(f"Train metadata shape: {train_meta.shape}")
    print(f"Val metadata shape:   {val_meta.shape}")
    print(f"Test metadata shape:  {test_meta.shape}")

    assert not train_meta.empty, "Train metadata should not be empty."

    # Read a single structure
    sample_id = train_meta.iloc[0]["id"]
    sample_path = train_meta.iloc[0]["file_path"]
    atoms = read_structure(sample_path)

    print(f"Loaded structure for ID {sample_id}: {atoms}")
    assert isinstance(atoms, Atoms), "read_structure should return an ASE Atoms object."
    assert len(atoms) > 0, "Structure should have atoms."

    # -------------------------------------------------------------------------
    # 2. Demonstrate Electrostatics Calculator
    # -------------------------------------------------------------------------
    print("\n--- 2. Electrostatics Calculator ---")

    electro_calc = ElectrostaticsCalculator(
        ewald_cutoff=6.0
    )  # Reduced cutoff for speed

    # Madelung Energy
    madelung = electro_calc.calculate_madelung_energy(atoms)
    print(f"Madelung Energy: {madelung:.4f}")
    assert isinstance(madelung, float), "Madelung energy should be a float."

    # Bond Valence Features
    bvs_feats = electro_calc.calculate_bvs_features(atoms)
    print(f"BVS Features keys: {list(bvs_feats.keys())[:5]} ...")
    assert (
        "GII" in bvs_feats
    ), "Global Instability Index (GII) missing from BVS features."

    # -------------------------------------------------------------------------
    # 3. Demonstrate Geometry Calculator
    # -------------------------------------------------------------------------
    print("\n--- 3. Geometry Calculator ---")

    geo_calc = GeometryCalculator(rdf_cutoff=5.0, rdf_bins=10, econ_cutoff=3.0)

    # Macroscopic Properties
    macro_props = geo_calc.get_macroscopic_props(atoms)
    print(f"Macroscopic Props: {list(macro_props.keys())}")
    assert "geo_density" in macro_props

    # ECoN
    econ_feats = geo_calc.calculate_econ(atoms)
    print(f"ECoN Features keys: {list(econ_feats.keys())[:5]} ...")

    # RDF
    rdf_feats = geo_calc.calculate_rdf(atoms)
    print(f"RDF Features keys: {list(rdf_feats.keys())[:5]} ...")

    # Full compute
    all_geo = geo_calc.compute_features(atoms)
    assert len(all_geo) > len(
        macro_props
    ), "compute_features should aggregate all geometric features."

    # -------------------------------------------------------------------------
    # 4. Demonstrate Feature Engineering Pipeline
    # -------------------------------------------------------------------------
    print("\n--- 4. Feature Engineering Pipeline ---")

    pipeline = FeaturePipeline()

    # Use a small subset for demonstration speed
    subset_size = 20
    train_subset = train_meta.head(subset_size).copy()
    val_subset = val_meta.head(subset_size).copy()
    test_subset = test_meta.head(subset_size).copy()

    print(f"Generating features for {subset_size} samples per split...")

    # Generate features (this handles caching internally, we use a unique name to avoid conflicts/long loads)
    # We disable loading cached data to force computation for the demo
    train_feats_df = pipeline.generate_features(
        train_subset, "demo_train", load_cached_data=False
    )
    val_feats_df = pipeline.generate_features(
        val_subset, "demo_val", load_cached_data=False
    )
    test_feats_df = pipeline.generate_features(
        test_subset, "demo_test", load_cached_data=False
    )

    print(f"Train features shape: {train_feats_df.shape}")

    # Verify merging
    expected_cols = [
        "id",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
        "madelung_energy",
    ]
    for col in expected_cols:
        assert (
            col in train_feats_df.columns
        ), f"Column {col} missing from merged dataframe."

    # Check for NaNs in critical columns (Madelung shouldn't be NaN for valid structures)
    if not train_feats_df["madelung_energy"].isnull().all():
        print("Features successfully computed.")
    else:
        print("Warning: All Madelung energies are NaN. Check input files.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Model Handler (Training & Prediction)
    # -------------------------------------------------------------------------
    print("\n--- 5. Model Handler ---")

    # Define fast hyperparameters for demo
    fast_xgb_params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_jobs": 1,
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    predictor = EnergyPredictor(xgb_params=fast_xgb_params)

    # Train the model
    print("Training model on subset...")
    predictor.train(train_feats_df, val_feats_df)

    # Predict on test subset
    print("Predicting on test subset...")
    results = predictor.predict(test_feats_df)

    print("Prediction results head:")
    print(results.head())

    # Validation
    assert "formation_energy_ev_natom" in results.columns
    assert "bandgap_energy_ev" in results.columns
    assert len(results) == len(test_subset)
    assert (
        results["formation_energy_ev_natom"] >= 0
    ).all(), "Formation energy predictions must be non-negative."
    assert (
        results["bandgap_energy_ev"] >= 0
    ).all(), "Bandgap predictions must be non-negative."

    # Save submission
    output_path = "./working/demo_submission.csv"
    predictor.save_submission(test_feats_df, output_path=output_path)

    assert os.path.exists(output_path), "Submission file was not created."
    print(f"\nDemo completed successfully. Output at {output_path}")


if __name__ == "__main__":
    main()
