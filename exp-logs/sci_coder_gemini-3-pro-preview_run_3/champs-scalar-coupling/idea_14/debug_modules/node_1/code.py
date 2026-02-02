import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import (
    preprocess_dataset,
    TargetStandardizer,
    SoADataset,
    get_dataloaders,
)
from library.model import SPCFN
from library.train import Trainer


def create_subset_metadata(
    base_meta_dir, target_meta_dir, n_train=50, n_val=10, n_test=10
):
    """
    Creates a small subset of metadata files to ensure the demo runs quickly.
    """
    print(f"Creating data subset in {target_meta_dir}...")
    os.makedirs(target_meta_dir, exist_ok=True)

    # Load original metadata
    train_full = pd.read_csv(os.path.join(base_meta_dir, "train_metadata.csv"))
    val_full = pd.read_csv(os.path.join(base_meta_dir, "val_metadata.csv"))
    test_full = pd.read_csv(os.path.join(base_meta_dir, "test_metadata.csv"))

    # Select unique molecules
    train_mols = train_full["molecule_name"].unique()[:n_train]
    val_mols = val_full["molecule_name"].unique()[:n_val]
    test_mols = test_full["molecule_name"].unique()[:n_test]

    # Filter dataframes
    train_subset = train_full[train_full["molecule_name"].isin(train_mols)].copy()
    val_subset = val_full[val_full["molecule_name"].isin(val_mols)].copy()
    test_subset = test_full[test_full["molecule_name"].isin(test_mols)].copy()

    # Save subsets
    train_path = os.path.join(target_meta_dir, "train_metadata.csv")
    val_path = os.path.join(target_meta_dir, "val_metadata.csv")
    test_path = os.path.join(target_meta_dir, "test_metadata.csv")

    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    print(
        f"Subset created: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)} rows."
    )
    return train_path, val_path, test_path


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("\n=== 1. Setup & Configuration ===")

    # Define temporary directories
    demo_dir = "./working/demo_execution"
    meta_subset_dir = os.path.join(demo_dir, "input_meta")
    working_subset_dir = os.path.join(demo_dir, "working")
    submission_subset_dir = os.path.join(demo_dir, "submission")

    # Clean up previous run if exists
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(working_subset_dir, exist_ok=True)
    os.makedirs(submission_subset_dir, exist_ok=True)

    # Create subset metadata
    new_train_path, new_val_path, new_test_path = create_subset_metadata(
        Config.METADATA_DIR, meta_subset_dir
    )

    # Override Config attributes to use the subset and temporary directories
    # This works because Config is a class and we are modifying its static attributes
    Config.TRAIN_META_PATH = new_train_path
    Config.VAL_META_PATH = new_val_path
    Config.TEST_META_PATH = new_test_path
    Config.WORKING_DIR = working_subset_dir
    Config.SUBMISSION_DIR = submission_subset_dir
    Config.SUBMISSION_PATH = os.path.join(submission_subset_dir, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(working_subset_dir, "best_model.pth")

    # Reduce training parameters for speed
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed
    set_seed(Config.SEED)
    print("Configuration updated for demo run.")

    # ==========================================
    # 2. Data Processing
    # ==========================================
    print("\n=== 2. Data Processing ===")

    # Trigger preprocessing (this will use the paths we just updated in Config)
    # We force load_cached_data=False to ensure it runs on our new subset
    preprocess_dataset("train", load_cached_data=False)
    preprocess_dataset("val", load_cached_data=False)
    preprocess_dataset("test", load_cached_data=False)

    # Verify processed files exist
    processed_dir = os.path.join(Config.WORKING_DIR, "processed")
    assert os.path.exists(
        os.path.join(processed_dir, "train_node_types.npy")
    ), "Train node types file missing"
    assert os.path.exists(
        os.path.join(processed_dir, "test_edge_indices.npy")
    ), "Test edge indices file missing"
    print("Preprocessing verification passed: Files generated.")

    # Fit Standardizer
    standardizer = TargetStandardizer()
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    aux_s = np.load(os.path.join(processed_dir, "train_aux_shielding.npy"))
    aux_c = np.load(os.path.join(processed_dir, "train_aux_charge.npy"))
    standardizer.fit(df_train, aux_s, aux_c)

    assert os.path.exists(standardizer.stats_path), "Stats file not saved"
    print("Standardizer fitted and stats saved.")

    # ==========================================
    # 3. Dataset & DataLoader
    # ==========================================
    print("\n=== 3. Dataset & DataLoader ===")

    # Instantiate Dataset
    train_ds = SoADataset("train", mode="train")
    sample_data = train_ds[0]

    # Verify Dataset Item
    required_keys = ["z", "pos", "edge_index", "edge_attr", "y", "coupling_index"]
    for key in required_keys:
        assert key in sample_data, f"Missing key {key} in dataset item"

    print(f"Sample molecule nodes: {sample_data['z'].shape[0]}")
    print(f"Sample molecule couplings: {sample_data['y'].shape[0]}")

    # Get DataLoaders
    train_loader, val_loader, test_loader, _ = get_dataloaders(load_cached_data=True)

    # Verify Batch
    batch = next(iter(train_loader))
    assert "batch" in batch, "Batch index missing in collated data"
    assert batch["z"].dim() == 1, "Node features should be 1D (types)"
    print(
        f"Batch loaded. Total nodes: {batch['z'].shape[0]}, Total couplings: {batch['y'].shape[0]}"
    )

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("\n=== 4. Model Initialization ===")

    device = torch.device(Config.DEVICE)
    model = SPCFN().to(device)

    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred_c, pred_s, pred_ch = model(batch)

    # Verify Output Shapes
    num_couplings = batch["y"].shape[0]
    num_nodes = batch["z"].shape[0]

    assert pred_c.shape == (
        num_couplings,
    ), f"Expected coupling pred shape ({num_couplings},), got {pred_c.shape}"
    assert pred_s.shape == (
        num_nodes,
        9,
    ), f"Expected shielding pred shape ({num_nodes}, 9), got {pred_s.shape}"
    assert pred_ch.shape == (
        num_nodes,
    ), f"Expected charge pred shape ({num_nodes},), got {pred_ch.shape}"

    print("Model forward pass successful. Output shapes verified.")

    # ==========================================
    # 5. Training Loop (Trainer)
    # ==========================================
    print("\n=== 5. Training Loop ===")

    trainer = Trainer(load_cached_data=True)

    # Run training
    # Note: We set MAX_EPOCHS to 2 in step 1
    trainer.fit()

    # Verify Model Checkpoint
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Best model checkpoint not found after training"
    print("Training complete. Model checkpoint saved.")

    # ==========================================
    # 6. Inference
    # ==========================================
    print("\n=== 6. Inference ===")

    trainer.predict()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission columns incorrect"

    # Check length matches test metadata
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)
    expected_len = len(df_test_meta)
    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"

    print(f"Submission generated with {len(df_sub)} rows.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nFAILED: {e}")
        # Re-raise to ensure non-zero exit code if wrapper checks it,
        # though the prompt asks to fail explicitly.
        raise e
