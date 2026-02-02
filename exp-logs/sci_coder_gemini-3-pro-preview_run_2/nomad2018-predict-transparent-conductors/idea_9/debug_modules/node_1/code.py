import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import GaussianSmearing, StandardScaler
from library.data import get_dataloaders
from library.model import DualStreamCGCNN
from library.train import run_training, generate_submission


def run_demo():
    print("=" * 80)
    print("Running Dual-Stream CGCNN Pipeline Demo")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Configure for Fast Debug Run
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for debug run...")

    # Redirect output directories to a demo folder inside ./working
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Update file paths
    Config.TRAIN_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "train_graphs_debug.npz")
    Config.VAL_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "val_graphs_debug.npz")
    Config.TEST_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "test_graphs_debug.npz")
    Config.SCALER_CACHE = os.path.join(Config.CACHE_DIR, "scalers_debug.npz")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set debug hyperparameters
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples
    Config.BATCH_SIZE = 8
    Config.MAX_EPOCHS = 2
    Config.NUM_WORKERS = 0  # Use main process for simplicity in demo

    # Create directories
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Test Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Testing Utility Classes...")

    # Test GaussianSmearing
    print("  Testing GaussianSmearing...")
    rbf = GaussianSmearing(start=0.0, stop=5.0, n_gaussians=10)
    distances = torch.tensor([0.0, 2.5, 5.0])
    rbf_out = rbf(distances)
    assert rbf_out.shape == (3, 10), f"Expected shape (3, 10), got {rbf_out.shape}"
    print("    GaussianSmearing shape check passed.")

    # Test StandardScaler
    print("  Testing StandardScaler...")
    scaler = StandardScaler(device="cpu")
    data = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler.fit(data)

    transformed = scaler.transform(data)
    expected_mean = torch.tensor([0.0, 0.0])
    expected_std = torch.tensor([1.0, 1.0])

    assert torch.allclose(
        transformed.mean(dim=0), expected_mean, atol=1e-5
    ), "Scaler mean is not 0"
    assert torch.allclose(
        transformed.std(dim=0), expected_std, atol=1e-5
    ), "Scaler std is not 1"

    reconstructed = scaler.inverse_transform(transformed)
    assert torch.allclose(data, reconstructed, atol=1e-5), "Inverse transform failed"
    print("    StandardScaler logic check passed.")

    # -------------------------------------------------------------------------
    # 3. Test Data Loading and Graph Construction
    # -------------------------------------------------------------------------
    print("\n[3] Testing Data Loading (this may take a moment to process raw files)...")

    # Force re-processing of raw data by setting load_cached_data=False initially
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=False
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Inspect one batch
    batch = next(iter(train_loader))
    print("  Inspecting a training batch:")
    print(f"    Batch size: {batch.num_graphs}")
    print(f"    Node features (x): {batch.x.shape}")
    print(f"    Edge index: {batch.edge_index.shape}")
    print(f"    Edge attr: {batch.edge_attr.shape}")
    print(f"    Global features: {batch.global_feat.shape}")
    print(f"    Targets (y): {batch.y.shape}")

    # Assertions for batch structure
    assert batch.x.dim() == 1, "Node features should be 1D (atom indices)"
    assert batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert (
        batch.global_feat.shape[1] == Config.N_GLOBAL_FEATURES
    ), f"Global features dim mismatch. Expected {Config.N_GLOBAL_FEATURES}, got {batch.global_feat.shape[1]}"
    assert batch.y.shape[1] == 2, "Targets should have 2 columns"

    print("  Data loading checks passed.")

    # -------------------------------------------------------------------------
    # 4. Test Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Testing Model Architecture...")

    device = torch.device("cpu")  # Use CPU for demo to ensure compatibility
    model = DualStreamCGCNN().to(device)

    # Forward pass with the batch retrieved earlier
    batch = batch.to(device)
    output = model(batch)

    print(f"    Model Output Shape: {output.shape}")

    assert output.shape == (
        batch.num_graphs,
        2,
    ), f"Output shape mismatch. Expected ({batch.num_graphs}, 2), got {output.shape}"

    print("  Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Test Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Testing Training Loop...")

    # We run the training function which handles the loop, validation, and saving
    # We pass load_cached_data=True now since we generated the cache in step 3
    trained_model, fitted_scaler, test_dl = run_training(load_cached_data=True)

    # Verify checkpoint creation
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"  Checkpoint found at {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # -------------------------------------------------------------------------
    # 6. Test Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Testing Submission Generation...")

    generate_submission(trained_model, test_dl, fitted_scaler)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"  Submission file found at {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission shape: {df_sub.shape}")
        print("  First 3 rows:")
        print(df_sub.head(3))

        # Basic validation of submission format
        assert list(df_sub.columns) == [
            "id",
            "formation_energy_ev_natom",
            "bandgap_energy_ev",
        ], "Submission columns are incorrect"
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n" + "=" * 80)
    print("Demo Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
