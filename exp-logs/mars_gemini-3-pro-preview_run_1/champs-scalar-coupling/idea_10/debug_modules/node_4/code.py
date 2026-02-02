import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library modules
from library import config, utils, feature_engineering, engine_gnn, engine_xgb

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting End-to-End Pipeline Demo...")

    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Create subdirectories
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    MODEL_DIR = os.path.join(DEMO_DIR, "models")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Working directory set to: {DEMO_DIR}")

    # Override Global Config Paths
    config.WORKING_DIR = DEMO_DIR
    config.CACHE_DIR = CACHE_DIR
    config.MODEL_DIR = MODEL_DIR

    # Define paths for subset data
    demo_train_path = os.path.join(DEMO_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_DIR, "test.csv")
    demo_struct_path = os.path.join(DEMO_DIR, "structures.csv")

    # ==========================================
    # 2. Data Subsetting (Speed Optimization)
    # ==========================================
    print("Creating data subsets for rapid demonstration...")

    # Load small chunks of metadata
    # We need enough data to form batches, but small enough to run fast
    df_train_sub = pd.read_csv(config.TRAIN_METADATA, nrows=500)
    df_val_sub = pd.read_csv(config.VAL_METADATA, nrows=100)
    df_test_sub = pd.read_csv(config.TEST_METADATA, nrows=100)

    # Identify relevant molecules
    train_mols = df_train_sub["molecule_name"].unique()
    val_mols = df_val_sub["molecule_name"].unique()
    test_mols = df_test_sub["molecule_name"].unique()
    all_mols = np.concatenate([train_mols, val_mols, test_mols])

    # Load structures and filter
    # Reading full structures csv is fast enough to filter once
    df_struct = pd.read_csv(config.STRUCTURES_CSV)
    df_struct_sub = df_struct[df_struct["molecule_name"].isin(all_mols)].copy()

    # Save subsets
    df_train_sub.to_csv(demo_train_path, index=False)
    df_val_sub.to_csv(demo_val_path, index=False)
    df_test_sub.to_csv(demo_test_path, index=False)
    df_struct_sub.to_csv(demo_struct_path, index=False)

    # Override Config Inputs
    config.TRAIN_METADATA = demo_train_path
    config.VAL_METADATA = demo_val_path
    config.TEST_METADATA = demo_test_path
    config.STRUCTURES_CSV = demo_struct_path

    # Override Model Hyperparameters for Speed
    config.GNN_PARAMS["epochs"] = 2
    config.GNN_PARAMS["batch_size"] = 32
    config.GNN_PARAMS["hidden_dim"] = 32
    config.GNN_PARAMS["num_rbf"] = 16
    config.GNN_PARAMS["node_dim"] = 32
    config.GNN_PARAMS["output_dim"] = 32

    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 4
    config.XGB_PARAMS["learning_rate"] = 0.1
    # Use 'hist' to be safe on small data or if GPU init overhead is high,
    # though 'gpu_hist' is default in config. Let's keep config default (gpu_hist)
    # but reduce capacity.

    # ==========================================
    # 3. Symbolic Feature Engineering
    # ==========================================
    print("\n--- Step 1: Symbolic Feature Engineering ---")
    # Load the subset structures (will be cached in demo dir)
    df_structures = utils.load_structures(load_cached_data=False)

    # Generate features for all splits
    # We pass load_cached_data=False to force generation in our new demo cache
    df_feats_train = feature_engineering.generate_symbolic_features(
        utils.load_metadata("train"), df_structures, "train", load_cached_data=False
    )
    df_feats_val = feature_engineering.generate_symbolic_features(
        utils.load_metadata("val"), df_structures, "val", load_cached_data=False
    )
    df_feats_test = feature_engineering.generate_symbolic_features(
        utils.load_metadata("test"), df_structures, "test", load_cached_data=False
    )

    print(f"Train features shape: {df_feats_train.shape}")

    # ==========================================
    # 4. GNN Training & Embedding Extraction
    # ==========================================
    print("\n--- Step 2: GNN Training ---")
    # Train GNN
    # load_cached_data=False forces the graph dataset to be built from our subset csvs
    engine_gnn.train_gnn(load_cached_data=False)

    print("\n--- Step 3: GNN Embedding Extraction ---")
    # Extract embeddings
    emb_train = engine_gnn.extract_embeddings("train", load_cached_data=False)
    emb_val = engine_gnn.extract_embeddings("val", load_cached_data=False)
    emb_test = engine_gnn.extract_embeddings("test", load_cached_data=False)

    print(f"Train embeddings shape: {emb_train.shape}")

    # ==========================================
    # 5. Merge Features
    # ==========================================
    print("\n--- Step 4: Merging Features ---")

    def merge_data(df_feats, df_emb):
        # Merge on ID
        return df_feats.merge(df_emb, on="id", how="left")

    X_train_full = merge_data(df_feats_train, emb_train)
    X_val_full = merge_data(df_feats_val, emb_val)
    X_test_full = merge_data(df_feats_test, emb_test)

    # Ensure target is present in train/val
    # Symbolic features df includes the target from metadata
    print(f"Merged Train Shape: {X_train_full.shape}")

    # ==========================================
    # 6. Stratified XGBoost Training
    # ==========================================
    print("\n--- Step 5: XGBoost Ensemble Training ---")

    ensemble = engine_xgb.StratifiedEnsemble()

    # Fit the ensemble
    ensemble.fit(X_train_full, X_val_full)

    # ==========================================
    # 7. Prediction & Submission
    # ==========================================
    print("\n--- Step 6: Prediction ---")

    preds = ensemble.predict(X_test_full)

    print("Predictions generated:")
    print(preds.head())

    # ==========================================
    # 8. Validation
    # ==========================================
    print("\n--- Step 7: Validation ---")

    # Check 1: Output shape matches test input
    assert len(preds) == len(
        df_test_sub
    ), f"Prediction count ({len(preds)}) does not match test set size ({len(df_test_sub)})"

    # Check 2: No NaNs in prediction
    assert (
        not preds["scalar_coupling_constant"].isnull().any()
    ), "Predictions contain NaNs"

    # Check 3: ID alignment
    # Sort both to ensure alignment for check
    test_ids = df_test_sub["id"].sort_values().values
    pred_ids = preds["id"].sort_values().values
    np.testing.assert_array_equal(test_ids, pred_ids, err_msg="IDs do not match")

    print("SUCCESS: Pipeline executed and verified.")


if __name__ == "__main__":
    main()
