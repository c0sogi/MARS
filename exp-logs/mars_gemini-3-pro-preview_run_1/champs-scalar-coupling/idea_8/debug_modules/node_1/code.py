import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shutil
import warnings

# Import from the provided library
import library.config as config
from library.feature_engineering import FeatureEngineer
from library.model_pipeline import StratifiedRegressor
from library.utils import calculate_log_mae, save_submission, reduce_mem_usage

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_demo_data(working_dir, n_molecules=50):
    """
    Creates a small subset of the metadata for demonstration purposes.
    """
    print(f"\n[Demo] Creating data subsets (Top {n_molecules} molecules)...")

    # Define paths
    demo_train_path = os.path.join(working_dir, "train.csv")
    demo_val_path = os.path.join(working_dir, "val.csv")
    demo_test_path = os.path.join(working_dir, "test.csv")

    # Load original metadata
    df_train = pd.read_csv(config.TRAIN_CSV)
    df_val = pd.read_csv(config.VAL_CSV)
    df_test = pd.read_csv(config.TEST_CSV)

    # Get subset of molecules
    train_mols = df_train["molecule_name"].unique()[:n_molecules]
    val_mols = df_val["molecule_name"].unique()[:n_molecules]
    test_mols = df_test["molecule_name"].unique()[:n_molecules]

    # Filter dataframes
    sub_train = df_train[df_train["molecule_name"].isin(train_mols)].copy()
    sub_val = df_val[df_val["molecule_name"].isin(val_mols)].copy()
    sub_test = df_test[df_test["molecule_name"].isin(test_mols)].copy()

    # Save to working directory
    sub_train.to_csv(demo_train_path, index=False)
    sub_val.to_csv(demo_val_path, index=False)
    sub_test.to_csv(demo_test_path, index=False)

    print(f"  Train subset: {sub_train.shape}")
    print(f"  Val subset:   {sub_val.shape}")
    print(f"  Test subset:  {sub_test.shape}")

    # Collect all relevant molecules for structure filtering
    all_demo_mols = np.concatenate([train_mols, val_mols, test_mols])

    return demo_train_path, demo_val_path, demo_test_path, all_demo_mols


def run_demo():
    # 1. Configuration
    WORKING_DIR = "./working/demo_execution"
    os.makedirs(WORKING_DIR, exist_ok=True)
    set_seed(config.RANDOM_STATE)

    # Override Config for Speed
    print("\n[Demo] Overriding XGBoost configuration for speed...")
    config.XGB_PARAMS["training"]["n_estimators"] = 10
    config.XGB_PARAMS["training"]["early_stopping_rounds"] = 5
    config.XGB_PARAMS["training"]["verbose"] = 0
    config.XGB_PARAMS["common"]["max_depth"] = 3
    config.XGB_PARAMS["common"]["n_jobs"] = 4

    # Override Model Directory to avoid overwriting real models
    config.MODEL_DIR = os.path.join(WORKING_DIR, "xgb_models")

    # 2. Prepare Data
    train_path, val_path, test_path, demo_mols = setup_demo_data(WORKING_DIR)

    # 3. Feature Engineering Demo
    print("\n[Demo] Starting Feature Engineering Pipeline...")
    fe = FeatureEngineer(cache_dir=WORKING_DIR, verbose=True)

    # Step 3.1: Load Structures
    structures = fe.load_structures()

    # Optimization: Filter structures to only those in our demo subset
    print(f"  Filtering structures from {len(structures)} atoms...")
    structures = structures[structures["molecule_name"].isin(demo_mols)].copy()
    print(f"  ...to {len(structures)} atoms for demo.")

    # Step 3.2: Build Graph
    adjacency = fe.build_adjacency(structures)
    assert not adjacency.empty, "Adjacency matrix is empty!"
    assert "dist" in adjacency.columns, "Distance column missing in adjacency."

    # Step 3.3: Compute Node Features
    nodes = fe.compute_node_features(structures, adjacency)
    assert "degree" in nodes.columns, "Node degree feature missing."

    # Step 3.4: Process Datasets (Train/Val/Test)
    print("  Processing Train Set...")
    X_train, y_train, _ = fe.process_dataset(train_path, structures, nodes, adjacency)

    print("  Processing Validation Set...")
    X_val, y_val, _ = fe.process_dataset(val_path, structures, nodes, adjacency)

    print("  Processing Test Set...")
    X_test, _, ids_test = fe.process_dataset(
        test_path, structures, nodes, adjacency, is_test=True
    )

    # Validation of FE Output
    print("\n[Demo] Validating Feature Engineering Output...")
    assert X_train.shape[0] > 0, "X_train is empty"
    assert len(y_train) == X_train.shape[0], "Mismatch between X_train and y_train"
    assert (
        X_test.shape[1] == X_train.shape[1]
    ), "Feature mismatch between Train and Test"
    assert not X_train.isnull().any().any(), "NaN values found in X_train"
    print("  Data shapes and integrity verified.")

    # 4. Model Training Demo
    print("\n[Demo] Starting Model Training (StratifiedRegressor)...")
    regressor = StratifiedRegressor(model_dir=config.MODEL_DIR)

    # Fit models
    regressor.fit(X_train, y_train, X_val, y_val)

    # Verify models were saved
    saved_models = os.listdir(config.MODEL_DIR)
    print(f"  Saved models: {saved_models}")
    assert len(saved_models) > 0, "No models were saved!"

    # 5. Prediction & Evaluation Demo
    print("\n[Demo] Generating Predictions...")

    # Predict on Validation to check metric
    val_preds = regressor.predict(X_val)
    assert len(val_preds) == len(y_val), "Prediction length mismatch"

    # Calculate Metric
    print("  Calculating Validation Metric...")
    val_eval_df = pd.DataFrame(
        {
            "type": X_val["type"].values,
            "scalar_coupling_constant": y_val,
            "prediction": val_preds,
        }
    )
    score = calculate_log_mae(val_eval_df)
    print(f"  Demo Validation Score: {score:.4f}")

    # Predict on Test
    test_preds = regressor.predict(X_test)

    # 6. Submission Demo
    print("\n[Demo] Saving Submission...")
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    save_submission(ids_test, test_preds, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file not created."

    # Verify submission format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == [
        "id",
        "scalar_coupling_constant",
    ], "Incorrect submission columns"
    assert len(sub_df) == len(ids_test), "Submission length mismatch"

    print("\n[Demo] execution completed successfully.")


if __name__ == "__main__":
    run_demo()
