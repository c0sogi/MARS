import os
import sys
import pandas as pd
import numpy as np
import shutil

# =============================================================================
# 1. PREPARATION & SUBSETTING (BEFORE LIBRARY IMPORTS)
# =============================================================================
# We prepare a small subset of data to ensure the demo runs quickly.

DEMO_DIR = "./working/demo_run"
META_DIR = os.path.join(DEMO_DIR, "metadata")
OS_INPUT_DIR = "./input"
ORIG_META_DIR = "./metadata"

os.makedirs(META_DIR, exist_ok=True)

print("Preparing demo data subsets...")


# Load original metadata
# We take a small number of molecules to keep runtime low
# Note: The split is by molecule, so we filter by molecule_name
def create_subset(filename, n_molecules=50):
    df = pd.read_csv(os.path.join(ORIG_META_DIR, filename))
    mols = df["molecule_name"].unique()[:n_molecules]
    df_subset = df[df["molecule_name"].isin(mols)].copy()

    # Save to demo dir
    save_path = os.path.join(META_DIR, filename)
    df_subset.to_csv(save_path, index=False)
    return mols


train_mols = create_subset("train.csv", n_molecules=100)
val_mols = create_subset("val.csv", n_molecules=20)
test_mols = create_subset("test.csv", n_molecules=20)

all_demo_mols = np.concatenate([train_mols, val_mols, test_mols])

# Subset structures.csv
print("Subsetting structures...")
df_struct = pd.read_csv(os.path.join(OS_INPUT_DIR, "structures.csv"))
df_struct_subset = df_struct[df_struct["molecule_name"].isin(all_demo_mols)].copy()
struct_path = os.path.join(DEMO_DIR, "structures.csv")
df_struct_subset.to_csv(struct_path, index=False)

# =============================================================================
# 2. CONFIGURATION PATCHING
# =============================================================================
# We must import config and patch it BEFORE importing features/model
# so that default arguments and global variables are initialized correctly.

import library.config as config

# Patch Data Paths
config.DATA_PATHS["train_meta"] = os.path.join(META_DIR, "train.csv")
config.DATA_PATHS["val_meta"] = os.path.join(META_DIR, "val.csv")
config.DATA_PATHS["test_meta"] = os.path.join(META_DIR, "test.csv")
config.DATA_PATHS["structures"] = struct_path

# Redirect cache paths to demo dir to avoid conflicts/permissions
config.DATA_PATHS["structures_processed"] = os.path.join(
    DEMO_DIR, "structures_processed.parquet"
)
config.DATA_PATHS["graph_edges"] = os.path.join(DEMO_DIR, "graph_edges.parquet")
config.DATA_PATHS["node_features"] = os.path.join(DEMO_DIR, "node_features.parquet")
config.DATA_PATHS["train_features"] = os.path.join(DEMO_DIR, "train_features.parquet")
config.DATA_PATHS["val_features"] = os.path.join(DEMO_DIR, "val_features.parquet")
config.DATA_PATHS["test_features"] = os.path.join(DEMO_DIR, "test_features.parquet")
config.DATA_PATHS["submission_output"] = os.path.join(DEMO_DIR, "submission.csv")

# Patch XGBoost Parameters for Speed
# Drastically reduce complexity for the demo
config.XGB_PARAMS.update(
    {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "early_stopping_rounds": 5,
        "n_jobs": 4,
        "tree_method": "hist",  # CPU compatible if GPU fails
    }
)

# Patch Working Directory for Models
# The StratifiedEnsemble uses WORKING_DIR/xgb_models
config.WORKING_DIR = DEMO_DIR

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Import libraries now that config is patched
    from library.utils import set_seed
    from library.features import generate_features
    from library.model import StratifiedEnsemble
    import library.model  # Access module to patch WORKING_DIR if needed

    # Ensure model module uses our demo dir
    library.model.WORKING_DIR = DEMO_DIR

    # 1. Set Seed
    print("\n[Step 1] Setting Random Seed")
    set_seed(42)

    # 2. Feature Generation
    print("\n[Step 2] Generating Features (Force Scratch)")
    # We force generation to test the pipeline logic on our subset
    train_df, val_df, test_df = generate_features(load_cached_data=False)

    # Validation of Feature Generation
    print(f"  Train Features Shape: {train_df.shape}")
    assert train_df.shape[0] > 0, "Train DataFrame is empty"
    assert "scalar_coupling_constant" in train_df.columns, "Target missing in train"
    assert "a0_L1_is_C" in train_df.columns, "Expected node feature missing"
    assert "cos_angle_mean" in train_df.columns, "Expected geometric feature missing"

    # 3. Model Training
    print("\n[Step 3] Training Stratified Ensemble")
    model = StratifiedEnsemble()

    # Fit models
    val_preds = model.fit(train_df, val_df)

    # Validation of Training
    assert (
        "prediction" in val_preds.columns
    ), "Prediction column missing in validation output"
    assert len(model.models) > 0, "No models were trained"

    # Check if model files were saved
    model_files = os.listdir(os.path.join(DEMO_DIR, "xgb_models"))
    print(f"  Saved {len(model_files)} files in model directory.")
    assert len(model_files) > 0, "Model files not saved"

    # 4. Inference
    print("\n[Step 4] Running Inference on Test Set")
    submission = model.predict(test_df)

    # Validation of Inference
    print(f"  Submission Shape: {submission.shape}")
    assert submission.shape[0] == len(test_df), "Submission row count mismatch"
    assert list(submission.columns) == [
        "id",
        "scalar_coupling_constant",
    ], "Incorrect submission columns"

    # Save Submission
    save_path = config.DATA_PATHS["submission_output"]
    submission.to_csv(save_path, index=False)
    print(f"  Submission saved to {save_path}")

    assert os.path.exists(save_path), "Submission file was not created"

    print("\n=== Demo Completed Successfully ===")
