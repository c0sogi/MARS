import os
import pandas as pd
import numpy as np
import shutil
from sklearn.metrics import mean_squared_error

# Import from the provided library
import library.config as config
from library.features import (
    generate_features,
    get_physical_descriptors,
    get_rdf_features,
    get_distributional_local_env,
)
from library.data import prepare_train_test_data, process_geometry_data
from library.model import train_model, generate_submission, LogTransformedXGBoost
import ase.io


def run_demo():
    print("=== Starting Library Demo Script ===")

    # --- 1. Setup Demo Configuration ---
    # We override paths in the config module to use a separate working directory
    # and sample metadata files for this demonstration.

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting up demo environment in {DEMO_DIR}")

    # Define paths for sample metadata
    demo_train_meta_path = os.path.join(DEMO_DIR, "train_metadata_sample.csv")
    demo_val_meta_path = os.path.join(DEMO_DIR, "val_metadata_sample.csv")
    demo_test_meta_path = os.path.join(DEMO_DIR, "test_metadata_sample.csv")

    # Define paths for sample features (parquet)
    demo_train_feats_path = os.path.join(DEMO_DIR, "train_features.parquet")
    demo_val_feats_path = os.path.join(DEMO_DIR, "val_features.parquet")
    demo_test_feats_path = os.path.join(DEMO_DIR, "test_features.parquet")

    # Override config paths
    # Note: We do NOT modify INPUT_DIR as the geometry files remain in ./input
    config.WORKING_DIR = DEMO_DIR
    config.TRAIN_METADATA_PATH = demo_train_meta_path
    config.VAL_METADATA_PATH = demo_val_meta_path
    config.TEST_METADATA_PATH = demo_test_meta_path
    config.TRAIN_FEATURES_PATH = demo_train_feats_path
    config.VAL_FEATURES_PATH = demo_val_feats_path
    config.TEST_FEATURES_PATH = demo_test_feats_path
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # --- 2. Create Sample Data ---
    print("\n--- Creating Sample Metadata ---")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Sample a small subset (e.g., 50 train, 20 val, 20 test)
    sample_train = orig_train.head(50).copy()
    sample_val = orig_val.head(20).copy()
    sample_test = orig_test.head(20).copy()

    # Save samples
    sample_train.to_csv(demo_train_meta_path, index=False)
    sample_val.to_csv(demo_val_meta_path, index=False)
    sample_test.to_csv(demo_test_meta_path, index=False)

    print(f"Created sample train metadata: {sample_train.shape}")
    print(f"Created sample val metadata:   {sample_val.shape}")
    print(f"Created sample test metadata:  {sample_test.shape}")

    # --- 3. Test Feature Extraction Logic (Unit Level) ---
    print("\n--- Testing Feature Extraction Functions ---")

    # Pick one geometry file to test individual functions
    sample_geom_path = os.path.join(config.INPUT_DIR, sample_train.iloc[0]["file_path"])
    atoms = ase.io.read(sample_geom_path, format="aims")

    # Test Physical Descriptors
    phys_feats = get_physical_descriptors(atoms)
    print(f"Physical Features keys: {list(phys_feats.keys())}")
    assert "volume" in phys_feats
    assert "density" in phys_feats
    assert phys_feats["volume"] > 0

    # Test RDF Features
    rdf_feats = get_rdf_features(atoms)
    # Check if we have keys starting with rdf_
    rdf_keys = [k for k in rdf_feats.keys() if k.startswith("rdf_")]
    print(f"Generated {len(rdf_keys)} RDF features.")
    assert len(rdf_keys) > 0

    # Test Distributional Local Env
    dle_feats = get_distributional_local_env(atoms)
    dle_keys = [k for k in dle_feats.keys() if k.startswith("dle_")]
    print(f"Generated {len(dle_keys)} DLE features.")
    assert len(dle_keys) > 0

    # --- 4. Test Data Pipeline (Integration Level) ---
    print("\n--- Testing Data Pipeline (prepare_train_test_data) ---")

    # This function calls process_geometry_data -> generate_features internally
    # It will use the paths we monkeypatched in config
    X_train, y_train_dict, X_val, y_val_dict, X_test, test_ids = (
        prepare_train_test_data(load_cached_data=False)
    )

    # Validations
    assert (
        X_train.shape[0] == 50
    ), f"Expected 50 training samples, got {X_train.shape[0]}"
    assert X_val.shape[0] == 20, f"Expected 20 validation samples, got {X_val.shape[0]}"
    assert X_test.shape[0] == 20, f"Expected 20 test samples, got {X_test.shape[0]}"

    # Check targets
    for target in config.TARGET_COLS:
        assert target in y_train_dict
        assert len(y_train_dict[target]) == 50
        # Check if log transform was applied (values should be smaller than raw if raw > 0)
        # Formation energy is small, but bandgap is usually > 0.
        # Just checking type is Series or array
        assert isinstance(y_train_dict[target], (pd.Series, np.ndarray))

    print("Data pipeline executed successfully. Features generated and loaded.")

    # --- 5. Test Model Training ---
    print("\n--- Testing Model Training ---")

    # Define fast hyperparameters for demo
    demo_xgb_params = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_jobs": 1,
        "random_state": 42,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }

    trained_models = {}

    for target_name in config.TARGET_COLS:
        print(f"Training for target: {target_name}")
        y_train = np.expm1(
            y_train_dict[target_name]
        )  # Revert log for train_model input as it expects raw
        y_val = np.expm1(y_val_dict[target_name])  # Revert log for train_model input

        # train_model handles log transform internally
        model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            params=demo_xgb_params,
            target_name=target_name,
            early_stopping_rounds=5,
            verbose=False,
        )

        trained_models[target_name] = model

        # Verify model type
        assert isinstance(model, LogTransformedXGBoost)
        # Verify prediction capability
        sample_pred = model.predict(X_val.iloc[:5])
        assert len(sample_pred) == 5
        print(f"Sample predictions for {target_name}: {sample_pred}")

    # --- 6. Test Submission Generation ---
    print("\n--- Testing Submission Generation ---")

    submission_df = generate_submission(
        trained_models["formation_energy_ev_natom"],
        trained_models["bandgap_energy_ev"],
        X_test,
        test_ids,
        config.SUBMISSION_PATH,
    )

    # Validations
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"
    assert submission_df.shape == (
        20,
        3,
    ), f"Submission shape mismatch: {submission_df.shape}"
    assert list(submission_df.columns) == [
        "id",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
    ]
    assert not submission_df.isnull().values.any(), "Submission contains NaNs"

    print("Submission generated successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
