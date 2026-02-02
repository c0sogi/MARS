import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import provided library components
from library.config import Config
import library.data
import library.layers
import library.trainer
import library.utils


def run_demo():
    print("=== Starting Scalar Coupling Prediction Demo ===")

    # ==========================================
    # 1. Setup & Mini-Dataset Creation
    # ==========================================
    # Define paths for the demo
    DEMO_BASE = "./working/demo_execution"
    DEMO_INPUT = os.path.join(DEMO_BASE, "input_meta")
    DEMO_WORKING = os.path.join(DEMO_BASE, "working")

    # Clean up previous runs
    if os.path.exists(DEMO_BASE):
        shutil.rmtree(DEMO_BASE)
    os.makedirs(DEMO_INPUT)
    os.makedirs(DEMO_WORKING)

    print("Creating mini-datasets for rapid verification...")

    # Helper to sample molecules and save mini-metadata
    def create_mini_meta(original_path, save_path, n_mols, filter_submission=False):
        df = pd.read_csv(original_path)
        mols = df["molecule_name"].unique()
        # Deterministic sample
        np.random.seed(42)
        sampled_mols = np.random.choice(mols, min(n_mols, len(mols)), replace=False)

        df_mini = df[df["molecule_name"].isin(sampled_mols)].copy()
        df_mini.to_csv(save_path, index=False)
        return df_mini

    # Create Mini Train/Val/Test Metadata
    # We use the original paths defined in Config to find the source files
    create_mini_meta(
        Config.TRAIN_META_PATH,
        os.path.join(DEMO_INPUT, "train_metadata.csv"),
        n_mols=20,
    )
    create_mini_meta(
        Config.VAL_META_PATH, os.path.join(DEMO_INPUT, "val_metadata.csv"), n_mols=5
    )
    df_test_mini = create_mini_meta(
        Config.TEST_META_PATH, os.path.join(DEMO_INPUT, "test_metadata.csv"), n_mols=5
    )

    # Create Mini Sample Submission
    # We must ensure the submission file only contains IDs present in our mini test set
    full_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    test_ids = df_test_mini["id"].values
    mini_sub = full_sub[full_sub["id"].isin(test_ids)].copy()
    mini_sub_path = os.path.join(DEMO_INPUT, "sample_submission.csv")
    mini_sub.to_csv(mini_sub_path, index=False)

    print(f"Mini-datasets created in {DEMO_INPUT}")

    # ==========================================
    # 2. Configure Environment (Monkey Patching)
    # ==========================================
    # Override Config attributes to point to our demo files and reduce compute
    print("Configuring environment...")

    # Paths
    Config.WORKING_DIR = DEMO_WORKING
    Config.TRAIN_META_PATH = os.path.join(DEMO_INPUT, "train_metadata.csv")
    Config.VAL_META_PATH = os.path.join(DEMO_INPUT, "val_metadata.csv")
    Config.TEST_META_PATH = os.path.join(DEMO_INPUT, "test_metadata.csv")
    Config.SAMPLE_SUBMISSION_PATH = mini_sub_path
    Config.SUBMISSION_PATH = os.path.join(DEMO_BASE, "submission", "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_WORKING, "best_model.pth")

    # Model Hyperparameters (Small for speed)
    Config.HIDDEN_DIM = 64
    Config.NUM_LAYERS = 2
    Config.NUM_RBF = 16
    Config.NUM_ANGLE_RBF = 8

    # Training Hyperparameters
    Config.BATCH_SIZE = 8
    Config.MAX_EPOCHS = 2
    Config.NUM_WORKERS = 2  # Low worker count for small data overhead
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seeds
    library.utils.seed_everything(Config.SEED)

    # ==========================================
    # 3. Data Processing Pipeline
    # ==========================================
    print("Running Data Processing (library.data)...")

    # generate dataloaders (this triggers preprocess_data internally)
    train_loader, val_loader, test_loader = library.data.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
        load_cached_data=False,  # Force reprocessing for the demo
    )

    # Verify Data Loader
    print("Verifying Data Loader...")
    batch = next(iter(train_loader))

    # Check required keys
    required_keys = [
        "atom_types",
        "edge_index",
        "edge_dist",
        "triplet_index",
        "coupling_values",
        "coupling_types",
    ]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Check shapes
    assert batch["edge_index"].shape[0] == 2, "Edge index must be (2, N)"
    assert batch["coupling_values"].ndim == 1, "Coupling values must be 1D"

    print(
        f"Batch verification passed. Loaded {len(train_loader.dataset.mol_map)} training molecules."
    )

    # ==========================================
    # 4. Model Initialization & Verification
    # ==========================================
    print("Initializing Model (library.layers)...")
    model = library.layers.get_model(Config()).to(Config.DEVICE)

    # Run a dummy forward pass to verify architecture
    model.eval()
    with torch.no_grad():
        out = model(
            batch["atom_types"],
            batch["edge_index"],
            batch["edge_dist"],
            batch["triplet_index"],
            batch["triplet_angle"],
            batch["coupling_node_indices"],
            batch["coupling_edge_indices"],
            batch["coupling_types"],
        )

    assert "coupling" in out, "Model output missing 'coupling' prediction"
    assert (
        out["coupling"].shape[0] == batch["coupling_values"].shape[0]
    ), f"Output shape mismatch: {out['coupling'].shape} vs {batch['coupling_values'].shape}"

    print("Model forward pass successful.")

    # ==========================================
    # 5. Training Loop (library.trainer)
    # ==========================================
    print("Starting Training Loop (library.trainer)...")

    trainer = library.trainer.Trainer(Config())

    # Train for defined epochs (2)
    trainer.train(train_loader, val_loader)

    # Verify model checkpoint exists
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not saved."
    print("Training complete.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("Generating Submission...")

    # Generate submission file
    trainer.generate_submission(test_loader)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_sub) == len(
        mini_sub
    ), f"Submission length mismatch: {len(df_sub)} vs {len(mini_sub)}"

    # Check for NaNs
    assert (
        not df_sub["scalar_coupling_constant"].isnull().any()
    ), "Submission contains NaNs."

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
