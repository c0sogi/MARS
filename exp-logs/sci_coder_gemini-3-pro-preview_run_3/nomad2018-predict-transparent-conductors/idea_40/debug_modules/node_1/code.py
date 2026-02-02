import os
import sys
import pandas as pd
import numpy as np
import ase

# Import configuration first to patch mutable parameters
import library.config as config

# Patch XGBoost parameters for speed in this demonstration
# We reduce n_estimators significantly to ensure the script finishes quickly.
config.XGB_PARAMS["n_estimators"] = 10
config.XGB_PARAMS["n_jobs"] = 4  # Use reasonable parallelism

# Import library modules
from library.data_loader import load_metadata, load_structure
from library.descriptors import StructureFeaturizer
from library.preprocessing import get_preprocessed_data, inverse_transform_targets
from library.model import XGBoostRegressorWrapper, generate_submission


def main():
    print("=== Material Property Prediction Pipeline Demo ===\n")

    # ---------------------------------------------------------
    # 1. Data Loading Verification
    # ---------------------------------------------------------
    print("--- Step 1: Verifying Data Loading ---")
    # Load training metadata (using a small sample if debug mode were on, but here we load full)
    # We use sample_size just to show the API, though DEBUG_MODE is False by default in config.
    train_meta = load_metadata("train", sample_size=10)
    print(f"Training metadata loaded. Shape: {train_meta.shape}")

    # Load a specific geometry file to ensure ASE can read it
    sample_id = train_meta.iloc[0]["id"]
    sample_path = train_meta.iloc[0]["file_path"]
    print(f"Loading structure for ID {sample_id} from {sample_path}...")
    atoms = load_structure(sample_path)

    assert isinstance(atoms, ase.Atoms), "Loaded object is not an ASE Atoms instance."
    print(f"Structure loaded successfully: {atoms.get_chemical_formula()}")
    print("Data loading verification passed.\n")

    # ---------------------------------------------------------
    # 2. Feature Extraction Logic Verification
    # ---------------------------------------------------------
    print("--- Step 2: Verifying Feature Extraction Logic ---")
    featurizer = StructureFeaturizer()

    print("Computing features for sample structure...")
    features = featurizer.featurize(atoms)

    print(f"Generated {len(features)} features.")

    # Check for critical feature groups
    keys = features.keys()
    has_rdf = any("rdf" in k for k in keys)
    has_bvs = any("bvs" in k for k in keys)
    has_vol = "vol_per_atom" in keys

    assert has_rdf, "RDF features missing."
    assert has_bvs, "Bond Valence Sum features missing."
    assert has_vol, "Macroscopic features missing."

    print("Feature extraction logic verification passed.\n")

    # ---------------------------------------------------------
    # 3. Running Preprocessing Pipeline
    # ---------------------------------------------------------
    print("--- Step 3: Running Full Preprocessing Pipeline ---")
    print("This step computes features for all splits and cleans the data.")
    print("Note: load_cached_data=False forces re-computation.")

    # Process Train Data
    # This will also fit the DataCleaner and save its state
    print("\nProcessing Training Data...")
    df_train = get_preprocessed_data("train", load_cached_data=False)
    print(f"Train Data Shape: {df_train.shape}")

    # Verify targets are log-transformed (should be small positive values usually)
    # Formation energy is >= 0. Log1p should be >= 0.
    assert (
        (df_train[config.TARGET_COLS] >= 0).all().all()
    ), "Target values contain negatives."

    # Process Validation Data
    # This loads the DataCleaner state fitted on train
    print("\nProcessing Validation Data...")
    df_val = get_preprocessed_data("val", load_cached_data=False)
    print(f"Val Data Shape: {df_val.shape}")

    # Process Test Data
    print("\nProcessing Test Data...")
    df_test = get_preprocessed_data("test", load_cached_data=False)
    print(f"Test Data Shape: {df_test.shape}")

    # Ensure columns match (except targets)
    train_feats = set(df_train.columns) - set(config.TARGET_COLS)
    test_feats = set(df_test.columns)
    # Test might have 'id' and 'file_path', train has them too.
    # The cleaner ensures feature set consistency.
    missing_in_test = train_feats - test_feats
    assert not missing_in_test, f"Test set missing features: {missing_in_test}"

    print("Preprocessing pipeline finished successfully.\n")

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("--- Step 4: Training XGBoost Models ---")
    model_wrapper = XGBoostRegressorWrapper()

    # Train on the processed data
    # Note: Using the patched low n_estimators for speed
    model_wrapper.train(df_train, df_val)

    print("Models trained successfully.")
    for target in config.TARGET_COLS:
        assert target in model_wrapper.models, f"Model for {target} missing."
    print("\n")

    # ---------------------------------------------------------
    # 5. Generating Submission
    # ---------------------------------------------------------
    print("--- Step 5: Generating Submission ---")

    # Predict on test set
    log_preds = model_wrapper.predict(df_test)

    # Inverse transform predictions
    final_preds = pd.DataFrame()
    final_preds["id"] = df_test["id"]
    for col in config.TARGET_COLS:
        final_preds[col] = inverse_transform_targets(log_preds[col].values)

    print("Predictions generated.")
    print(final_preds.head())

    # Basic sanity checks on predictions
    if "bandgap_energy_ev" in final_preds.columns:
        assert (
            final_preds["bandgap_energy_ev"] > 0
        ).all(), "Bandgap energy should be positive."

    # Save submission (simulated by calling the library function which saves to disk)
    # We pass the model wrapper we just trained
    # Note: generate_submission inside library/model.py re-loads test data internally.
    # We rely on the cache we just created in Step 3 for speed.

    # To use the function from library.model properly, we need to ensure it uses the cache
    # we just generated.
    generate_submission(model_wrapper, load_cached_data=True)

    print("\n=== Demo Execution Complete ===")


if __name__ == "__main__":
    main()
