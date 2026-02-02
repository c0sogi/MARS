import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import random

# Import provided library modules
import library.config as config
import library.data_utils as data_utils
from library.models_lgbm import LGBMWrapper
from library.models_nn import NNWrapper
from library.ensemble import StackingManager

# --- Constants for Demonstration ---
DEMO_DIR = "./working/demo_data"
CACHE_DIR = os.path.join(DEMO_DIR, "cache")
SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_mini_datasets():
    """
    Creates small subsets of the original data for rapid testing.
    Filters for only 2 classes to ensure StratifiedKFold works on small samples.
    """
    print("Creating mini datasets for demonstration...")
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Load original metadata (just enough rows to get samples)
    # We read a bit more to ensure we find enough samples of specific classes
    train_full = pd.read_parquet(os.path.join("./metadata", "train.parquet"))
    val_full = pd.read_parquet(os.path.join("./metadata", "val.parquet"))
    test_full = pd.read_parquet(os.path.join("./metadata", "test.parquet"))

    # Filter for top 2 classes (1 and 2) to simplify the demo problem
    # Original classes: 1, 2, 3, 4, 6, 7. We stick to 1 and 2 for binary demo.
    target_col = "Cover_Type"
    classes_to_keep = [1, 2]

    train_subset = train_full[train_full[target_col].isin(classes_to_keep)].sample(
        n=500, random_state=SEED
    )
    val_subset = val_full[val_full[target_col].isin(classes_to_keep)].sample(
        n=100, random_state=SEED
    )
    test_subset = test_full.sample(n=100, random_state=SEED)

    # Save mini datasets
    train_path = os.path.join(DEMO_DIR, "train.parquet")
    val_path = os.path.join(DEMO_DIR, "val.parquet")
    test_path = os.path.join(DEMO_DIR, "test.parquet")

    train_subset.to_parquet(train_path, index=False)
    val_subset.to_parquet(val_path, index=False)
    test_subset.to_parquet(test_path, index=False)

    print(f"Mini datasets saved to {DEMO_DIR}")
    return train_path, val_path, test_path


def patch_configuration(train_path, val_path, test_path):
    """
    Monkey-patches the config module to use demo paths and faster hyperparameters.
    """
    print("Patching library configuration...")

    # 1. Update Paths
    config.INPUT_DIR = DEMO_DIR
    config.TRAIN_PATH = train_path
    config.VAL_PATH = val_path
    config.TEST_PATH = test_path
    config.CACHE_DIR = CACHE_DIR
    config.SUBMISSION_DIR = SUBMISSION_DIR
    config.SUBMISSION_PATH = SUBMISSION_PATH

    # 2. Update Class Mappings (We reduced problem to 2 classes: 1 and 2)
    config.CLASS_MAP = {1: 0, 2: 1}
    config.INV_CLASS_MAP = {0: 1, 1: 2}
    config.NUM_CLASSES = 2

    # 3. Update Global Settings
    config.N_FOLDS = 2
    config.BASELINE_SCORE = 0.0  # Force submission generation

    # 4. Update Model Hyperparameters for Speed
    # LGBM
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["early_stopping_rounds"] = 5
    config.LGBM_PARAMS["num_class"] = config.NUM_CLASSES
    config.LGBM_PARAMS["verbose"] = -1

    # NN
    config.NN_PARAMS["epochs"] = 2
    config.NN_PARAMS["batch_size"] = 32
    config.NN_PARAMS["output_dim"] = config.NUM_CLASSES
    config.NN_PARAMS["hidden_layers"] = [64, 32]  # Smaller net

    # Meta Learner
    config.META_PARAMS["max_iter"] = 50


def test_data_utils():
    print("\n--- Testing Data Utils ---")
    # Force recompute to test logic
    data = data_utils.preprocess_data(load_cached_data=False)

    # Verify structure
    assert "tree" in data and "nn" in data

    X_train_tree, y_train, X_val_tree, y_val, X_test_tree, test_ids = data["tree"]
    X_train_nn, _, _, _, _, _ = data["nn"]

    # Verify Shapes
    print(f"Train Tree Shape: {X_train_tree.shape}")
    print(f"Train NN Shape: {X_train_nn.shape}")

    assert len(X_train_tree) == 500, "Incorrect train size"
    assert len(X_val_tree) == 100, "Incorrect val size"
    assert len(X_test_tree) == 100, "Incorrect test size"

    # Verify Interaction Features
    # Config defines interactions like 'Elevation_x_Wilderness_Area1'
    interaction_col = "Elevation_x_Wilderness_Area1"
    assert (
        interaction_col in X_train_tree.columns
    ), f"Interaction feature {interaction_col} missing"

    # Verify NN Scaling (Mean approx 0, Std approx 1 for continuous)
    # Elevation is continuous
    elev_mean = X_train_nn["Elevation"].mean()
    elev_std = X_train_nn["Elevation"].std()
    assert abs(elev_mean) < 0.1, f"Scaling failed, mean is {elev_mean}"
    assert 0.9 < elev_std < 1.1, f"Scaling failed, std is {elev_std}"

    print("Data Utils verification passed.")
    return data


def test_lgbm_model(data):
    print("\n--- Testing LightGBM Wrapper ---")
    X_train, y_train, X_val, y_val, X_test, _ = data["tree"]

    model = LGBMWrapper()  # Uses patched config
    model.fit(X_train, y_train, X_val, y_val)

    # Test Prediction
    probs = model.predict_proba(X_test)
    assert probs.shape == (100, 2), f"LGBM output shape mismatch: {probs.shape}"

    print("LightGBM verification passed.")


def test_nn_model(data):
    print("\n--- Testing Neural Network Wrapper ---")
    X_train, y_train, X_val, y_val, X_test, _ = data["nn"]

    model = NNWrapper()  # Uses patched config
    model.fit(X_train, y_train, X_val, y_val)

    # Test Prediction
    probs = model.predict_proba(X_test)
    assert probs.shape == (100, 2), f"NN output shape mismatch: {probs.shape}"

    print("Neural Network verification passed.")


def test_ensemble_pipeline():
    print("\n--- Testing Ensemble Stacking Manager ---")
    manager = StackingManager()

    # Run the full pipeline
    manager.run()

    # Verify Submission
    assert os.path.exists(SUBMISSION_PATH), "Submission file not created"

    sub_df = pd.read_csv(SUBMISSION_PATH)
    assert sub_df.shape == (100, 2), f"Submission shape mismatch: {sub_df.shape}"
    assert list(sub_df.columns) == ["Id", "Cover_Type"], "Submission columns mismatch"

    # Check if predictions are valid original classes (1 or 2)
    unique_preds = sub_df["Cover_Type"].unique()
    valid_classes = [1, 2]
    assert all(
        p in valid_classes for p in unique_preds
    ), f"Invalid class predicted: {unique_preds}"

    print("Ensemble pipeline verification passed.")


if __name__ == "__main__":
    set_seed(SEED)

    # 1. Setup Data
    train_p, val_p, test_p = create_mini_datasets()

    # 2. Patch Config
    patch_configuration(train_p, val_p, test_p)

    # 3. Test Components
    data_container = test_data_utils()
    test_lgbm_model(data_container)
    test_nn_model(data_container)

    # 4. Test Full Pipeline
    test_ensemble_pipeline()

    print("\nAll demonstrations completed successfully.")
