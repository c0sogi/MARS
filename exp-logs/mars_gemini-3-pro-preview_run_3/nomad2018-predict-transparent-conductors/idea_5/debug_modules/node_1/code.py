import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
import library.config
from library.data_loader import load_metadata, load_geometries
from library.physical_descriptors import PhysicalFeaturizer
from library.gnn_processor import MatGLEmbedder
from library.model_trainer import EnergyPredictor


def run_demo():
    print("Starting demonstration of library components...")

    # 1. Set Random Seeds for Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    print("Random seeds set.")

    # 2. Monkey-patch Config for Speed
    # We reduce the number of estimators to make the training step very fast for this demo.
    print("Adjusting XGBoost parameters for speed...")
    library.config.XGB_PARAMS["n_estimators"] = 5
    library.config.XGB_PARAMS["early_stopping_rounds"] = 2
    # Ensure working directory is clean or distinct to avoid reading old cache if we want to test generation
    # The library uses 'idea_5' inside working. We will use load_cached_data=False to force re-computation.

    # 3. Demonstrate Data Loader
    print("\n--- Testing Data Loader ---")
    # Load a small subset of training metadata
    n_samples = 5
    df_meta = load_metadata(
        split="train", max_samples=n_samples, load_cached_data=False
    )

    # Assertions
    assert isinstance(df_meta, pd.DataFrame), "Metadata should be a DataFrame"
    assert len(df_meta) == n_samples, f"Expected {n_samples} rows, got {len(df_meta)}"
    assert "file_path" in df_meta.columns, "Metadata missing 'file_path' column"
    print("Metadata loaded successfully.")

    # Load geometries
    atoms_list = load_geometries(df_meta)

    # Assertions
    assert isinstance(atoms_list, list), "Geometries should be a list"
    assert (
        len(atoms_list) == n_samples
    ), f"Expected {n_samples} atoms objects, got {len(atoms_list)}"
    # Check one object
    assert hasattr(
        atoms_list[0], "get_positions"
    ), "Loaded object does not look like an ASE Atoms object"
    print("Geometries loaded successfully.")

    # 4. Demonstrate Physical Descriptors
    print("\n--- Testing Physical Featurizer ---")
    phys_featurizer = PhysicalFeaturizer()

    # Calculate analytical volume directly
    volumes = phys_featurizer.calculate_analytical_volume(df_meta)
    assert len(volumes) == n_samples
    assert np.all(volumes > 0), "Volumes must be positive"

    # Generate full features dataframe
    # We use a dummy split name to avoid overwriting real cache if needed, though load_cached_data=False prevents reading.
    df_phys = phys_featurizer.featurize(
        df_meta, atoms_list, split_name="demo_train", load_cached_data=False
    )

    # Assertions
    assert isinstance(df_phys, pd.DataFrame)
    assert len(df_phys) == n_samples
    assert "density" in df_phys.columns
    assert "analytical_volume" in df_phys.columns
    # Check for NaN
    assert not df_phys.isnull().values.any(), "Physical features contain NaNs"
    print("Physical features generated successfully.")
    print("Physical features columns:", df_phys.columns.tolist())

    # 5. Demonstrate GNN Processor (MatGL Embedder)
    print("\n--- Testing MatGL Embedder ---")
    embedder = MatGLEmbedder()

    # Generate embeddings
    # Note: This might take a moment to load the model
    df_emb = embedder.generate_chemically_resolved_embeddings(
        atoms_list, split_name="demo_train", load_cached_data=False
    )

    # Assertions
    assert isinstance(df_emb, pd.DataFrame)
    assert len(df_emb) == n_samples
    # Check expected columns. 4 elements (Al, Ga, In, O) * Embedding Dim (usually 64 for M3GNet)
    # We won't hardcode 64, but we check divisibility by 4
    n_cols = df_emb.shape[1]
    assert (
        n_cols > 0 and n_cols % 4 == 0
    ), f"Embedding columns ({n_cols}) should be divisible by 4 (Al, Ga, In, O)"
    print(f"MatGL embeddings generated successfully. Shape: {df_emb.shape}")

    # 6. Demonstrate Model Trainer (End-to-End)
    print("\n--- Testing Energy Predictor (End-to-End) ---")
    predictor = EnergyPredictor()

    # Train model on a small subset
    # We use slightly more samples for training to ensure XGBoost has enough data to split
    train_samples = 20
    print(f"Training on {train_samples} samples...")
    predictor.train_model(max_samples=train_samples, load_cached_data=False)

    # Verify models are stored
    assert "formation_energy_ev_natom" in predictor.models
    assert "bandgap_energy_ev" in predictor.models
    print("Models trained successfully.")

    # Predict on test set (small subset)
    test_samples = 5
    print(f"Predicting on {test_samples} test samples...")
    predictor.predict_and_submit(max_samples=test_samples, load_cached_data=False)

    # Verify submission file
    submission_path = library.config.SUBMISSION_FILE_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    assert len(df_sub) == test_samples, f"Submission should have {test_samples} rows"
    assert "id" in df_sub.columns
    assert "formation_energy_ev_natom" in df_sub.columns
    assert "bandgap_energy_ev" in df_sub.columns

    # Check for valid values (non-negative as enforced by predict_and_submit)
    assert (df_sub["formation_energy_ev_natom"] >= 0).all()
    assert (df_sub["bandgap_energy_ev"] >= 0).all()

    print("Submission generated and verified successfully.")
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demo()
