import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.features import radial_basis_functions, spherical_basis_functions
from library.data import get_dataloaders, COUPLING_TYPES
from library.model import HGANet
from library.train import Trainer, generate_submission


def main():
    print("=== Starting Demonstration and Verification Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1/6] Setting up Configuration...")

    # Initialize Config with debug=True to use a small subset of data
    config = Config(debug=True, epochs=1, batch_size=16, hidden_dim=32, num_workers=2)

    # Override specific settings for the demo to ensure speed
    config.WORKING_DIR = "./working/demo_run"
    config.TRAIN_CACHE = os.path.join(config.WORKING_DIR, "cached_train.npz")
    config.VAL_CACHE = os.path.join(config.WORKING_DIR, "cached_val.npz")
    config.TEST_CACHE = os.path.join(config.WORKING_DIR, "cached_test.npz")
    config.MODEL_SAVE_PATH = os.path.join(config.WORKING_DIR, "best_model.pt")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Reduce model complexity for demo
    config.NUM_MPNN_LAYERS = 2
    config.NUM_TRANSFORMER_LAYERS = 1
    config.RBF_SIZE = 20
    config.SBF_SIZE = 20
    config.NUM_HEADS = 4

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.SEED)
    print(f"Configuration initialized. Working dir: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Feature Engineering Verification
    # ---------------------------------------------------------
    print("\n[2/6] Verifying Feature Engineering Functions...")

    # Test Radial Basis Functions
    dists = torch.tensor([0.5, 1.0, 2.5, 4.9, 5.1], dtype=torch.float32)
    rbf = radial_basis_functions(dists, start=0.0, end=5.0, num_basis=10)

    assert rbf.shape == (
        5,
        10,
    ), f"RBF shape mismatch: expected (5, 10), got {rbf.shape}"
    assert (rbf >= 0).all() and (rbf <= 1).all(), "RBF values should be in [0, 1]"
    assert rbf[-1].sum() == 0, "RBF should be zero beyond cutoff"
    print("  Radial Basis Functions: OK")

    # Test Spherical Basis Functions
    # 3 triplets, distances and angles
    t_dists = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    t_angles = torch.tensor(
        [0.0, 1.57, 3.14], dtype=torch.float32
    )  # 0, 90, 180 degrees
    sbf = spherical_basis_functions(t_dists, t_angles, start=0.0, end=5.0, num_basis=20)

    assert sbf.shape == (
        3,
        20,
    ), f"SBF shape mismatch: expected (3, 20), got {sbf.shape}"
    print("  Spherical Basis Functions: OK")

    # ---------------------------------------------------------
    # 3. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[3/6] Verifying Data Loading and Processing...")

    # Clean up previous cache to force processing (ensures logic runs)
    for cache_file in [config.TRAIN_CACHE, config.VAL_CACHE, config.TEST_CACHE]:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    # Get dataloaders
    # This will trigger process_and_cache_data since cache is deleted
    train_loader, val_loader, test_loader, standardizer = get_dataloaders(
        config, load_cached_data=False
    )

    # Verify Standardizer
    assert standardizer.fitted, "Standardizer should be fitted"
    assert len(standardizer.means) > 0, "Standardizer means should not be empty"
    print(f"  Standardizer fitted on types: {list(standardizer.means.keys())}")

    # Verify Batch Structure
    batch = next(iter(train_loader))

    # Move batch to device for subsequent steps
    batch = {
        k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    required_keys = [
        "x",
        "edge_index",
        "edge_attr",
        "triplet_index",
        "triplet_attr",
        "coupling_index",
        "y",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    print(
        f"  Batch loaded successfully. Nodes: {batch['x'].shape[0]}, Couplings: {batch['y'].shape[0]}"
    )

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[4/6] Verifying Model Architecture...")

    model = HGANet(config).to(config.DEVICE)

    # Forward Pass
    preds = model(batch)

    assert (
        preds.shape == batch["y"].shape
    ), f"Prediction shape mismatch: {preds.shape} vs {batch['y'].shape}"
    print("  Forward pass successful.")

    # Backward Pass check (Gradient flow)
    loss = torch.nn.L1Loss()(preds, batch["y"])
    loss.backward()

    # Check if gradients exist for a key parameter
    param = list(model.readout_mlp.parameters())[0]
    assert param.grad is not None, "Gradients not computed."
    print("  Backward pass successful.")

    # ---------------------------------------------------------
    # 5. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[5/6] Verifying Training Loop...")

    # Re-instantiate model and optimizer to clear gradients/state
    model = HGANet(config).to(config.DEVICE)
    trainer = Trainer(config, model, train_loader, val_loader, standardizer)

    # Run one epoch
    print("  Running 1 training epoch...")
    train_mae = trainer.train_epoch()
    print(f"  Train MAE: {train_mae:.4f}")
    assert train_mae >= 0, "Training MAE should be non-negative"

    # Run validation
    print("  Running validation...")
    val_lmae = trainer.validate()
    print(f"  Val LMAE: {val_lmae:.4f}")

    # Save this model as "best" for the submission step
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint not found."

    # ---------------------------------------------------------
    # 6. Submission Workflow Verification
    # ---------------------------------------------------------
    print("\n[6/6] Verifying Submission Generation...")

    # Generate submission using the saved model
    generate_submission(config)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content format
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check against test metadata size (in debug mode, test size is also small)
    # Note: process_and_cache_data slices metadata to 1000 molecules in DEBUG mode.
    # We need to verify that the submission rows match the processed test data.
    # The generate_submission function loads test_metadata.csv.
    # In a real run, it loads the full file. In this script, we haven't modified test_metadata.csv,
    # but the dataloader (in DEBUG mode) only processes a subset.
    # The provided generate_submission function loads the FULL test_metadata.csv to create the dataframe structure,
    # but the dataloader yields predictions only for the processed subset (1000 molecules).
    # This would normally cause a mismatch error in `generate_submission` line:
    # `if len(df_test) != len(final_preds): raise ValueError...`
    #
    # However, for this demo to succeed without erroring on that mismatch, we must acknowledge that
    # `generate_submission` in `library.train` reads `config.TEST_METADATA`.
    # To make this verification pass in DEBUG mode without modifying library code,
    # we can temporarily mock the TEST_METADATA file to match the subset processed by the dataloader.

    print("  Submission file generated successfully.")
    print(f"  Submission head:\n{df_sub.head(3)}")

    print("\n=== All Verifications Passed Successfully ===")


if __name__ == "__main__":
    # We need to handle the potential mismatch in submission generation caused by DEBUG mode
    # The library code loads the full metadata but processes only a subset in debug mode.
    # To allow the script to finish successfully, we will patch the TEST_METADATA path in config
    # to point to a subset file matching the debug logic, JUST for the submission step verification.

    # Create a subset metadata file for the demo
    demo_meta_dir = "./working/demo_run/metadata"
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Load original test metadata
    orig_test_path = "./metadata/test_metadata.csv"
    df_test = pd.read_csv(orig_test_path)

    # The data loader in debug mode takes the first 1000 molecules.
    # We replicate that logic to create a matching metadata file.
    unique_mols = df_test["molecule_name"].unique()
    subset_mols = unique_mols[
        :1000
    ]  # Matches library/data.py:84 logic roughly (it takes head(1000) of df usually)

    # Actually library/data.py line 84: df_meta = df_meta.iloc[:1000]
    # So we just take the first 1000 rows of the dataframe.
    df_test_subset = df_test.iloc[:1000]

    subset_test_path = os.path.join(demo_meta_dir, "test_metadata.csv")
    df_test_subset.to_csv(subset_test_path, index=False)

    # Run main
    try:
        # We need to patch the config inside main, but main creates its own config.
        # We will rely on the fact that we can modify the file path in the config object inside main.
        # But since we can't inject into main easily, we'll just run main.
        # Wait, if I run main as is, generate_submission will fail due to length mismatch.
        # I must modify main to point config.TEST_METADATA to my subset file.

        # Let's redefine main slightly to accept the patch or do it inside.
        # I will modify the main function in the code block above to use this subset path.
        pass
    except Exception as e:
        print(e)

    # Execute the logic
    # Monkey-patching Config for the purpose of this single-file execution context
    # to ensure the submission step uses the subset metadata.
    original_init = Config.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Point to the subset file we just created, if it exists
        if os.path.exists("./working/demo_run/metadata/test_metadata.csv"):
            self.TEST_METADATA = "./working/demo_run/metadata/test_metadata.csv"

    Config.__init__ = patched_init

    main()
