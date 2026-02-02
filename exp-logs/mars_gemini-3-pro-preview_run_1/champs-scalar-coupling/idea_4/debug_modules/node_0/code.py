import os
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

# Import library modules
import library.config as config
from library.utils import seed_everything
from library.geometry_engine import GeometryEngine
from library.feature_pipeline import FeaturePipeline
from library.trainer import StratifiedTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_demo_data():
    """
    Creates a small subset of the metadata files in the working directory
    to allow for rapid execution of the pipeline.
    """
    print("Creating demo datasets (subsets)...")

    # Define demo paths
    demo_train_path = os.path.join(config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(config.WORKING_DIR, "demo_test.csv")

    # Load a small sample of the original metadata
    # We select enough rows to ensure we have examples of different coupling types
    df_train = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"), nrows=1000)
    df_val = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"), nrows=200)
    df_test = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"), nrows=100)

    # Save to working directory
    df_train.to_csv(demo_train_path, index=False)
    df_val.to_csv(demo_val_path, index=False)
    df_test.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path


def patch_configuration(train_path, val_path, test_path):
    """
    Monkey-patches the config module to use the demo datasets and
    faster model hyperparameters.
    """
    print("Patching configuration for demo run...")

    # Override file paths
    config.TRAIN_METADATA_PATH = train_path
    config.VAL_METADATA_PATH = val_path
    config.TEST_METADATA_PATH = test_path

    # Override XGBoost parameters for speed
    # Reduce estimators and depth significantly for the demo
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 4
    config.XGB_PARAMS["learning_rate"] = 0.1
    config.XGB_PARAMS["n_jobs"] = 4
    # Ensure we use CPU for the demo to avoid potential GPU init overhead on small data
    # though the prompt mentions A100 is available, small data overhead might be higher.
    # We'll keep 'cuda' if available, but fallback is handled by XGBoost usually.
    # For this demo, we force CPU to be safe and consistent.
    config.XGB_PARAMS["device"] = "cpu"
    config.XGB_PARAMS["tree_method"] = "hist"

    # Reduce early stopping rounds
    config.EARLY_STOPPING_ROUNDS = 2


def test_geometry_engine(df_subset):
    """
    Demonstrates and validates the GeometryEngine.
    """
    print("\n=== Testing GeometryEngine ===")

    # Instantiate engine
    geo_engine = GeometryEngine(config.STRUCTURES_PATH)

    # Validation 1: Check if structures are loaded
    print(f"Loaded {len(geo_engine.structures_map)} molecules into memory.")
    assert len(geo_engine.structures_map) > 0, "Structures map should not be empty."

    # Validation 2: Check a specific molecule
    sample_mol = df_subset.iloc[0]["molecule_name"]
    assert (
        sample_mol in geo_engine.structures_map
    ), f"Molecule {sample_mol} not found in structures."

    coords = geo_engine.structures_map[sample_mol]["coords"]
    assert coords.shape[1] == 3, "Coordinates should have 3 dimensions (x, y, z)."

    # Validation 3: Run feature calculation on the subset
    # Note: We use a custom dataset name 'demo_geo' to avoid conflicts
    df_geo = geo_engine.get_shortest_path_features(
        df_subset, "demo_geo", load_cached_data=False
    )

    print("Geometry features shape:", df_geo.shape)
    expected_cols = ["geo_path_len", "geo_angle", "geo_dihedral"]
    for col in expected_cols:
        assert col in df_geo.columns, f"Missing column {col} in geometry features."

    print("GeometryEngine test passed.")


def test_feature_pipeline(df_subset):
    """
    Demonstrates and validates the FeaturePipeline.
    """
    print("\n=== Testing FeaturePipeline ===")

    pipeline = FeaturePipeline(config.STRUCTURES_PATH)

    # Validation 1: Generate features
    # We use 'train' as dataset_name but pointing to our demo file via patched config
    # We force load_cached_data=False to ensure logic runs
    df_features = pipeline.generate_features(df_subset, "train", load_cached_data=False)

    print("Generated features shape:", df_features.shape)

    # Check for key feature groups
    # Distance features
    assert "dist" in df_features.columns
    assert "dist_inv2" in df_features.columns

    # Neighbor features
    assert "atom_0_n_C" in df_features.columns
    assert "atom_1_n_H" in df_features.columns

    # Check for target variable presence (since this is train)
    assert "scalar_coupling_constant" in df_features.columns

    # Validation 2: Prepare data for a specific type
    # Find a type present in the subset
    available_types = df_subset["type"].unique()
    target_type = available_types[0]
    print(f"Testing preparation for coupling type: {target_type}")

    X, y = pipeline.prepare_data_for_type(df_features, target_type)

    assert not X.empty, "X should not be empty."
    assert len(y) == len(X), "Target length should match features."

    # Ensure dropped columns
    forbidden_cols = [
        "id",
        "molecule_name",
        "type",
        "atom_0",
        "atom_1",
        "scalar_coupling_constant",
    ]
    for col in forbidden_cols:
        assert col not in X.columns, f"Column {col} should have been dropped."

    print("FeaturePipeline test passed.")


def test_stratified_trainer():
    """
    Demonstrates and validates the StratifiedTrainer.
    """
    print("\n=== Testing StratifiedTrainer ===")

    trainer = StratifiedTrainer()

    # Validation 1: Training
    # This will use the patched paths and parameters
    print("Running training loop...")
    trainer.train(load_cached_data=False)

    # Check if models were saved
    model_dir = os.path.join(config.WORKING_DIR, "xgb_models")
    saved_models = os.listdir(model_dir)
    print(f"Saved models: {saved_models}")
    assert len(saved_models) > 0, "No models were saved after training."

    # Validation 2: Prediction
    print("Running prediction loop...")
    trainer.predict(load_cached_data=False)

    # Check submission file
    submission_path = config.FINAL_SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print("Submission shape:", df_sub.shape)
    assert "id" in df_sub.columns
    assert "scalar_coupling_constant" in df_sub.columns
    assert not df_sub.empty, "Submission file is empty."

    print("StratifiedTrainer test passed.")


if __name__ == "__main__":
    # 1. Set Seed
    seed_everything(42)

    # 2. Create Demo Data
    train_path, val_path, test_path = create_demo_data()

    # 3. Patch Config
    patch_configuration(train_path, val_path, test_path)

    # Load the subset dataframe for component testing
    df_demo_train = pd.read_csv(train_path)

    # 4. Run Component Tests
    test_geometry_engine(df_demo_train)
    test_feature_pipeline(df_demo_train)

    # 5. Run Integration Test (Trainer)
    test_stratified_trainer()

    print("\nAll demonstrations completed successfully.")
