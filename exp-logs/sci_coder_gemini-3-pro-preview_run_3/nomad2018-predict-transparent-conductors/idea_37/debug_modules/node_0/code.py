import sys
import os
import shutil
import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock

# Ensure the current directory is in the path to import libraries correctly
sys.path.append(os.getcwd())

# Import the provided library modules
from library import data_handler
from library import physics_descriptors
from library import feature_manager
from library import regression_model

# Set random seeds for reproducibility
np.random.seed(42)


def demo_data_handler():
    """
    Demonstrates loading metadata and geometry files using data_handler.
    """
    print("\n" + "=" * 40)
    print(" Demo: Data Handler")
    print("=" * 40)

    # 1. Load a small subset of metadata (limit=5 for speed)
    print("Loading train metadata (first 5 rows)...")
    df_train = data_handler.load_metadata("train", limit=5)
    print(f"Loaded train metadata shape: {df_train.shape}")

    # Assertions to verify metadata loading
    assert not df_train.empty, "Train metadata should not be empty."
    assert "file_path" in df_train.columns, "Metadata must contain 'file_path' column."
    assert "id" in df_train.columns, "Metadata must contain 'id' column."

    # 2. Load geometry (ASE Atoms objects) corresponding to the metadata
    print("Loading geometry objects...")
    atoms_list = data_handler.load_geometry(df_train)
    print(f"Loaded {len(atoms_list)} Atoms objects.")

    # Assertions to verify geometry loading
    assert len(atoms_list) == len(
        df_train
    ), "Number of Atoms objects must match metadata rows."
    assert (
        atoms_list[0].get_global_number_of_atoms() > 0
    ), "Atoms object should contain atoms."

    return atoms_list[0]


def demo_physics_descriptors(atoms):
    """
    Demonstrates calculation of physics-based descriptors for a single structure.
    """
    print("\n" + "=" * 40)
    print(" Demo: Physics Descriptors")
    print("=" * 40)

    # 1. Calculate macroscopic properties (e.g., density, volume)
    print("Calculating macroscopic properties...")
    macro = physics_descriptors.calculate_macroscopic_props(atoms)
    print(f"Macroscopic props keys: {list(macro.keys())}")

    assert "density" in macro, "Macroscopic props should include density."
    assert macro["vol_per_atom"] > 0, "Volume per atom should be positive."

    # 2. Calculate Elemental Radial Distribution Functions (RDFs)
    print("Calculating elemental RDFs...")
    rdfs = physics_descriptors.calculate_elemental_rdfs(
        atoms, r_max=4.0, n_bins=20
    )  # Reduced bins for speed
    print(f"RDF features count: {len(rdfs)}")

    # Check for expected keys
    rdf_keys = [k for k in rdfs.keys() if k.startswith("rdf_")]
    assert len(rdf_keys) > 0, "Should have generated RDF features."

    # 3. Calculate Local Environment features (Bond Valence Sums, CN)
    print("Calculating local environment features...")
    local_env = physics_descriptors.calculate_local_environment(atoms)
    print(f"Local env features count: {len(local_env)}")

    # Check for expected keys
    bvs_keys = [k for k in local_env.keys() if k.startswith("bvs_")]
    assert len(bvs_keys) > 0, "Should have generated BVS features."

    # 4. Calculate Interaction Distributions (Bond Valences, Angles)
    print("Calculating interaction distributions...")
    interactions = physics_descriptors.calculate_interaction_distributions(atoms)
    print(f"Interaction features count: {len(interactions)}")

    # 5. Full wrapper function
    print("Running full structure processing wrapper...")
    full_feats = physics_descriptors.process_single_structure(atoms)
    print(f"Total extracted features for single structure: {len(full_feats)}")
    assert len(full_feats) > 0, "Feature dictionary should not be empty."


def demo_feature_manager():
    """
    Demonstrates feature cleaning and matrix generation (using mocks for speed).
    """
    print("\n" + "=" * 40)
    print(" Demo: Feature Manager")
    print("=" * 40)

    # 1. Test clean_features
    print("Testing clean_features function...")
    df_raw = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feat_normal": [1.0, 0.5, 3.0],
            "feat_nan": [np.nan, 2.0, 3.0],
            "feat_inf": [np.inf, 1.0, 0.0],
            "feat_const": [1.0, 1.0, 1.0],  # Should be dropped
        }
    )
    print("Raw features:\n", df_raw)

    df_clean = feature_manager.clean_features(df_raw)
    print("Cleaned features:\n", df_clean)

    assert "feat_const" not in df_clean.columns, "Constant column should be dropped."
    assert not df_clean.isnull().values.any(), "NaNs should be filled."
    assert not np.isinf(df_clean.values).any(), "Infs should be replaced."
    assert "id" in df_clean.columns, "ID column should be preserved."

    # 2. Test generate_feature_matrix with mocking
    # We mock load_metadata to return a very small dataframe to avoid processing thousands of files
    print("Testing generate_feature_matrix (with mocks)...")

    original_load = data_handler.load_metadata

    def mock_load_metadata(split, limit=None, random_state=42):
        # Force a small limit regardless of input
        return original_load(split, limit=3, random_state=random_state)

    # We patch load_metadata inside feature_manager
    with patch("library.feature_manager.load_metadata", side_effect=mock_load_metadata):
        # We set load_cached_data=False to force the calculation logic to run
        # We set n_jobs=1 to avoid multiprocessing overhead for this tiny test
        df_feats = feature_manager.generate_feature_matrix(
            "train", load_cached_data=False, n_jobs=1
        )

        print(f"Generated features shape: {df_feats.shape}")
        assert len(df_feats) == 3, "Should have generated features for 3 samples."
        assert "id" in df_feats.columns, "Feature matrix must contain 'id'."
        # Check if some physics features are present (assuming calculation succeeded)
        if len(df_feats.columns) > 1:
            print("Physics descriptors successfully generated.")


def demo_regression_model():
    """
    Demonstrates the regression model wrapper and data preparation.
    """
    print("\n" + "=" * 40)
    print(" Demo: Regression Model")
    print("=" * 40)

    # 1. Test LogTransformedXGBoost
    print("Initializing LogTransformedXGBoost...")
    # Use small n_estimators for speed
    model = regression_model.LogTransformedXGBoost(n_estimators=10, max_depth=3)

    # Create synthetic data
    # X has 50 samples, 5 features
    X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
    # y must be positive for log transform
    y_train = pd.Series(np.random.rand(50) * 10 + 0.1)

    X_val = pd.DataFrame(np.random.rand(10, 5), columns=[f"f{i}" for i in range(5)])
    y_val = pd.Series(np.random.rand(10) * 10 + 0.1)

    print("Training model on synthetic data...")
    rmsle = model.train(X_train, y_train, X_val, y_val, target_name="Demo Target")
    print(f"Validation RMSLE: {rmsle:.4f}")

    print("Predicting...")
    preds = model.predict(X_val)
    print(f"Predictions: {preds[:5]}...")

    assert len(preds) == 10, "Prediction length mismatch."
    assert (preds >= 0).all(), "Predictions should be non-negative."

    # 2. Test prepare_data with mocks
    print("Testing prepare_data (with mocks)...")

    # Mock metadata
    mock_meta = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "formation_energy_ev_natom": [0.1, 0.2, 0.3],
            "bandgap_energy_ev": [1.0, 1.5, 2.0],
            "file_path": ["path/1", "path/2", "path/3"],
        }
    )

    # Mock features
    mock_feats = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "feat_A": [0.5, 0.6, 0.7],
            "feat_B": [0.1, 0.1, 0.1],  # Constant, should be dropped by clean_features
        }
    )

    # Patch load_metadata and generate_feature_matrix in regression_model
    with patch("library.regression_model.load_metadata", return_value=mock_meta):
        with patch(
            "library.regression_model.generate_feature_matrix", return_value=mock_feats
        ):

            ids, X, targets = regression_model.prepare_data(
                "train", load_cached_data=False
            )

            print("Prepared IDs:", ids.tolist())
            print("Prepared X columns:", X.columns.tolist())
            print("Prepared Targets keys:", targets.keys())

            assert len(ids) == 3
            assert "feat_A" in X.columns
            assert (
                "feat_B" not in X.columns
            ), "Constant column feat_B should have been removed."
            assert "formation_energy_ev_natom" in targets
            assert "bandgap_energy_ev" in targets


if __name__ == "__main__":
    # Create working directory for cache if it doesn't exist
    os.makedirs("./working/idea_37", exist_ok=True)

    try:
        # Run demos
        atom_obj = demo_data_handler()
        demo_physics_descriptors(atom_obj)
        demo_feature_manager()
        demo_regression_model()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        import traceback

        traceback.print_exc()
