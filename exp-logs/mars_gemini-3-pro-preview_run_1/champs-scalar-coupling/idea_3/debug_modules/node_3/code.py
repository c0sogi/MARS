import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set fixed seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------
from library.config import GASEConfig

# Define a temporary directory for this demo run
DEMO_ROOT = "./working/demo_run"
DEMO_META = os.path.join(DEMO_ROOT, "metadata")
DEMO_WORK = os.path.join(DEMO_ROOT, "working")

# Ensure directories exist
os.makedirs(DEMO_META, exist_ok=True)
os.makedirs(DEMO_WORK, exist_ok=True)

print(">>> Patching GASEConfig for Demo Execution...")

# Override Input Data Paths to point to our subsampled metadata
GASEConfig.TRAIN_CSV = os.path.join(DEMO_META, "train.csv")
GASEConfig.VAL_CSV = os.path.join(DEMO_META, "val.csv")
GASEConfig.TEST_CSV = os.path.join(DEMO_META, "test.csv")

# Override Working Directory
GASEConfig.WORKING_DIR = DEMO_WORK

# Manually update dependent paths in GASEConfig since they were initialized at import
GASEConfig.PROCESSED_TRAIN_PATH = os.path.join(DEMO_WORK, "processed_train.parquet")
GASEConfig.PROCESSED_VAL_PATH = os.path.join(DEMO_WORK, "processed_val.parquet")
GASEConfig.PROCESSED_TEST_PATH = os.path.join(DEMO_WORK, "processed_test.parquet")

GASEConfig.GRAPH_CACHE_DIR = os.path.join(DEMO_WORK, "graph_cache")
os.makedirs(GASEConfig.GRAPH_CACHE_DIR, exist_ok=True)

GASEConfig.EMBEDDINGS_TRAIN_PATH = os.path.join(DEMO_WORK, "embeddings_train.npy")
GASEConfig.EMBEDDINGS_VAL_PATH = os.path.join(DEMO_WORK, "embeddings_val.npy")
GASEConfig.EMBEDDINGS_TEST_PATH = os.path.join(DEMO_WORK, "embeddings_test.npy")

GASEConfig.MPNN_MODEL_PATH = os.path.join(DEMO_WORK, "mpnn_best_model.pth")
GASEConfig.XGB_MODELS_DIR = os.path.join(DEMO_WORK, "xgb_models")
os.makedirs(GASEConfig.XGB_MODELS_DIR, exist_ok=True)

GASEConfig.SUBMISSION_PATH = os.path.join(DEMO_WORK, "submission.csv")

# Override Hyperparameters for Speed
GASEConfig.MPNN_EPOCHS = 2
GASEConfig.MPNN_BATCH_SIZE = 32
GASEConfig.XGB_BASE_PARAMS["n_estimators"] = 10
GASEConfig.XGB_BASE_PARAMS["early_stopping_rounds"] = 2
# Reducing neighbors slightly to speed up graph building for demo
GASEConfig.MAX_NEIGHBORS = 16

# -----------------------------------------------------------------------------
# 2. Create Subsampled Datasets
# -----------------------------------------------------------------------------
print(">>> Creating Subsampled Datasets...")


def create_subset(source_path, dest_path, n_mols):
    df = pd.read_csv(source_path)
    # Select first n_mols unique molecules
    mols = df["molecule_name"].unique()[:n_mols]
    subset = df[df["molecule_name"].isin(mols)].copy()
    subset.to_csv(dest_path, index=False)
    return len(subset), len(mols)


# Create subsets (100 train, 20 val, 20 test molecules)
n_train_rows, n_train_mols = create_subset(
    "./metadata/train.csv", GASEConfig.TRAIN_CSV, 100
)
n_val_rows, n_val_mols = create_subset("./metadata/val.csv", GASEConfig.VAL_CSV, 20)
n_test_rows, n_test_mols = create_subset("./metadata/test.csv", GASEConfig.TEST_CSV, 20)

print(f"    Train: {n_train_rows} rows ({n_train_mols} molecules)")
print(f"    Val:   {n_val_rows} rows ({n_val_mols} molecules)")
print(f"    Test:  {n_test_rows} rows ({n_test_mols} molecules)")

# -----------------------------------------------------------------------------
# 3. Import Library Modules (Post-Patching)
# -----------------------------------------------------------------------------
from library.data_utils import process_and_cache_data
from library.graph_utils import MoleculeGraphBuilder
from library.mpnn_trainer import MPNNRunner
from library.feature_eng import FeatureAssembler
from library.xgb_model import StratifiedEnsemble

# -----------------------------------------------------------------------------
# 4. Main Execution Pipeline
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Step 1: Geometric Feature Engineering ---
    print("\n>>> [Step 1] Processing Geometric Features...")
    # This function loads metadata, merges structures, calculates distances/angles, and saves parquet
    df_train, df_val, df_test = process_and_cache_data(load_cached_data=False)

    # Verification
    assert os.path.exists(
        GASEConfig.PROCESSED_TRAIN_PATH
    ), "Processed train parquet missing"
    assert len(df_train) == n_train_rows, "Processed train row count mismatch"
    assert "dist" in df_train.columns, "Distance feature missing"
    assert "cos_c0_c1" in df_train.columns, "Angle feature missing"
    print("    Geometric features processed successfully.")

    # --- Step 2: Graph Construction ---
    print("\n>>> [Step 2] Building Molecular Graphs...")
    builder = MoleculeGraphBuilder()

    # Process graph data for all splits
    # Note: We force load_cached_data=False to ensure we process our new subsampled data
    train_graph = builder.process_data("train", load_cached_data=False)
    val_graph = builder.process_data("val", load_cached_data=False)
    test_graph = builder.process_data("test", load_cached_data=False)

    # Verification
    assert train_graph["nodes"].shape[1] == (
        GASEConfig.NUM_ATOM_TYPES * 2
    ), "Node feature dim mismatch"
    assert os.path.exists(
        os.path.join(GASEConfig.GRAPH_CACHE_DIR, "train_nodes.npy")
    ), "Graph cache file missing"
    print("    Graph construction complete.")

    # --- Step 3: MPNN Training ---
    print("\n>>> [Step 3] Training MPNN...")
    runner = MPNNRunner()

    # Train the model (uses the patched EPOCHS=2)
    # We pass load_cached_data=True because we just created the cache in Step 2
    runner.train(load_cached_data=True, epochs=GASEConfig.MPNN_EPOCHS)

    # Verification
    assert os.path.exists(GASEConfig.MPNN_MODEL_PATH), "MPNN model file not generated"
    print("    MPNN training complete.")

    # --- Step 4: Embedding Extraction ---
    print("\n>>> [Step 4] Extracting Embeddings...")
    # This generates .npy files for train, val, and test
    runner.extract_embeddings(load_cached_data=True)

    # Verification
    assert os.path.exists(GASEConfig.EMBEDDINGS_TRAIN_PATH), "Train embeddings missing"
    train_embeds = np.load(GASEConfig.EMBEDDINGS_TRAIN_PATH)
    # Embedding dim = Hidden * 2 + Edge_Attr_Dim (RBF+Inv)
    # Hidden=128, Edge=19. Head input = 128*2 + 19 = 275.
    # Wait, the MPNN returns `embedding` which is the input to the head.
    # Check mpnn_model.py: embedding = torch.cat([h_0, h_1, pairs_edge_attr], dim=1)
    expected_dim = GASEConfig.MPNN_HIDDEN_DIM * 2 + (GASEConfig.MPNN_NUM_RBF + 3)
    assert (
        train_embeds.shape[1] == expected_dim
    ), f"Embedding dim mismatch. Got {train_embeds.shape[1]}, expected {expected_dim}"
    print(f"    Embeddings extracted. Shape: {train_embeds.shape}")

    # --- Step 5: Feature Assembly ---
    print("\n>>> [Step 5] Assembling Hybrid Features...")
    assembler = FeatureAssembler()

    # Assemble train data to verify
    full_train_df = assembler.assemble_data("train", load_cached_data=True)

    # Verification
    assert (
        "embed_0" in full_train_df.columns
    ), "Embedding columns missing in assembled dataframe"
    assert (
        "dist" in full_train_df.columns
    ), "Geometric columns missing in assembled dataframe"
    print("    Feature assembly successful.")

    # --- Step 6: XGBoost Ensemble Training & Inference ---
    print("\n>>> [Step 6] Running Stratified XGBoost Ensemble...")
    ensemble = StratifiedEnsemble()

    # Train
    ensemble.train_ensemble(load_cached_data=True)

    # Verify at least one model exists (e.g., 1JHC is usually present)
    model_files = os.listdir(GASEConfig.XGB_MODELS_DIR)
    assert len(model_files) > 0, "No XGBoost models were saved"

    # Predict
    ensemble.predict_ensemble(load_cached_data=True)

    # Verification
    assert os.path.exists(GASEConfig.SUBMISSION_PATH), "Submission file missing"
    submission = pd.read_csv(GASEConfig.SUBMISSION_PATH)
    assert (
        "id" in submission.columns and "scalar_coupling_constant" in submission.columns
    ), "Submission schema incorrect"
    assert (
        len(submission) == n_test_rows
    ), f"Submission row count mismatch. Expected {n_test_rows}, got {len(submission)}"

    print("\n>>> Demo Execution Completed Successfully!")
    print(f"    Submission saved to: {GASEConfig.SUBMISSION_PATH}")
    print(f"    First 5 predictions:\n{submission.head().to_string(index=False)}")
