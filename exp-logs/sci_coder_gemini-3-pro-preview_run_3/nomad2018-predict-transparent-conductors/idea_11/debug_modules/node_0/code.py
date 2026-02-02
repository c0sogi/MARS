import os
import sys
import pandas as pd
import numpy as np
import ase
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Import library components
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
)
from library.data_utils import load_metadata, read_geometry
from library.feature_engineering import (
    PhysicalDescriptor,
    RDFDescriptor,
    MatGLEmbedder,
    FeaturePipeline,
)
from library.model_wrapper import DualEnergyPredictor
from library.workflow import train_and_evaluate, generate_submission


def test_data_utils():
    print("\n[1] Testing data_utils...")

    # Test load_metadata with sample_size
    sample_n = 5
    df_train = load_metadata("train", sample_size=sample_n)

    assert isinstance(df_train, pd.DataFrame), "load_metadata should return a DataFrame"
    assert len(df_train) == sample_n, f"Expected {sample_n} rows, got {len(df_train)}"
    assert "file_path" in df_train.columns, "Metadata must contain 'file_path'"

    # Test read_geometry
    # Pick the first file path from the loaded metadata
    rel_path = df_train.iloc[0]["file_path"]
    atoms = read_geometry(rel_path)

    assert isinstance(
        atoms, ase.Atoms
    ), "read_geometry should return an ase.Atoms object"
    assert len(atoms) > 0, "Atoms object should not be empty"

    print("    data_utils tests passed.")
    return atoms


def test_feature_engineering_components(atoms):
    print("\n[2] Testing feature engineering components...")

    # 1. PhysicalDescriptor
    phys_desc = PhysicalDescriptor()
    phys_feats = phys_desc.calculate(atoms)
    assert "phys_volume" in phys_feats, "PhysicalDescriptor missing 'phys_volume'"
    assert "phys_density" in phys_feats, "PhysicalDescriptor missing 'phys_density'"
    print(f"    Physical features calculated: {list(phys_feats.keys())}")

    # 2. RDFDescriptor
    rdf_desc = RDFDescriptor(cutoff=4.0, n_bins=10)
    rdf_feats = rdf_desc.calculate(atoms)
    # Check if we have RDF features. Keys look like 'rdf_Al_Al_0', etc.
    # We just check if dictionary is not empty and has expected prefix
    assert len(rdf_feats) > 0, "RDFDescriptor returned empty dictionary"
    assert any(
        k.startswith("rdf_") for k in rdf_feats.keys()
    ), "RDF features missing correct prefix"
    print(f"    RDF features calculated: {len(rdf_feats)} bins")

    # 3. MatGLEmbedder
    # Note: This might return zeros if model loading fails, but keys should exist
    matgl_embedder = MatGLEmbedder()
    matgl_feats = matgl_embedder.calculate(atoms)
    assert len(matgl_feats) > 0, "MatGLEmbedder returned empty dictionary"
    assert "matgl_0" in matgl_feats, "MatGL features missing expected keys"
    print(f"    MatGL features calculated: {len(matgl_feats)} dimensions")

    print("    Feature engineering component tests passed.")


def test_feature_pipeline():
    print("\n[3] Testing FeaturePipeline...")

    # Initialize pipeline
    pipeline = FeaturePipeline()

    # Process a very small split to save time
    # We force load_cached_data=False to ensure calculation logic runs
    sample_size = 5

    # Process Train
    print("    Processing train split...")
    df_train_feats = pipeline.process_split(
        "train", sample_size=sample_size, load_cached_data=False
    )

    assert len(df_train_feats) == sample_size, "Pipeline output length mismatch"
    # Check if features from different descriptors are present
    assert (
        "phys_volume" in df_train_feats.columns
    ), "Pipeline output missing physical features"
    # Check if metadata columns are preserved
    assert "id" in df_train_feats.columns, "Pipeline output missing metadata 'id'"
    assert (
        "formation_energy_ev_natom" in df_train_feats.columns
    ), "Pipeline output missing targets"

    print("    FeaturePipeline tests passed.")
    return df_train_feats


def test_model_wrapper(df_train):
    print("\n[4] Testing DualEnergyPredictor...")

    # Split the processed features into train/val for model testing
    # Since df_train is very small (5), we just use it for both for API testing
    df_val = df_train.copy()

    predictor = DualEnergyPredictor()

    # Fit
    print("    Fitting model...")
    predictor.fit(df_train, df_val)

    # Check if models are stored
    assert (
        "formation_energy_ev_natom" in predictor.models
    ), "Model for formation energy missing"
    assert "bandgap_energy_ev" in predictor.models, "Model for bandgap energy missing"

    # Evaluate
    print("    Evaluating model...")
    score = predictor.evaluate(df_val)
    assert isinstance(score, float), "Evaluate should return a float score"
    assert score >= 0, "RMSLE score should be non-negative"

    # Predict
    print("    Predicting...")
    preds = predictor.predict(df_val)
    assert "id" in preds.columns, "Predictions missing 'id'"
    assert "formation_energy_ev_natom" in preds.columns, "Predictions missing target 1"
    assert "bandgap_energy_ev" in preds.columns, "Predictions missing target 2"
    assert len(preds) == len(df_val), "Prediction length mismatch"

    print("    DualEnergyPredictor tests passed.")


def test_full_workflow():
    print("\n[5] Testing Full Workflow...")

    # Use a small sample size for the full workflow test
    debug_size = 10

    # 1. Train and Evaluate
    # This will load metadata, compute/load features, train models, and eval
    # We set load_cached_data=False to force re-computation for demonstration
    predictor = train_and_evaluate(sample_size=debug_size, load_cached_data=False)

    assert predictor is not None, "train_and_evaluate returned None"

    # 2. Generate Submission
    # This processes test data and creates the submission file
    submission_df = generate_submission(
        predictor, sample_size=debug_size, load_cached_data=False
    )

    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"
    assert len(submission_df) == debug_size, "Submission dataframe length mismatch"

    # Verify submission format
    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    print("    Full workflow tests passed.")


if __name__ == "__main__":
    print("=== Starting Demonstration Script ===")

    # 1. Test Data Utils
    sample_atoms = test_data_utils()

    # 2. Test Feature Engineering Components
    test_feature_engineering_components(sample_atoms)

    # 3. Test Feature Pipeline
    # This returns a dataframe with features we can use for model testing
    df_features = test_feature_pipeline()

    # 4. Test Model Wrapper
    test_model_wrapper(df_features)

    # 5. Test Full Workflow
    test_full_workflow()

    print("\n=== Demonstration Completed Successfully ===")
