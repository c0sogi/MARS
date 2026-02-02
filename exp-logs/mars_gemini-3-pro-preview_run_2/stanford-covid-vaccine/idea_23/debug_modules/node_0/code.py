import os
import shutil
import torch
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, GlobalMCRMSE
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMCRMSELoss
from library.engine import train_fn, eval_fn

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def demo_implementation():
    print("==== RNA Degradation Prediction Library Demo ====\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demo execution...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.SUBSET_SIZE = 32  # Use only 32 samples for this demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect cache files to demo folder to avoid messing up real training cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_demo.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_demo.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_demo.npz")

    # Set reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Load data with debug=True to trigger subsetting
    # load_cached_data=False forces processing from CSV to NPZ in the demo dir
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch one batch
    inputs, partner_indices, targets, ids = next(iter(train_loader))

    print(f"    Batch Size: {inputs.size(0)}")
    print(f"    Inputs Shape: {inputs.shape} (Expected: B, 107, 19)")
    print(f"    Partner Indices Shape: {partner_indices.shape} (Expected: B, 107)")
    print(f"    Targets Shape: {targets.shape} (Expected: B, 107, 5)")

    # Assertions
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_NODE_FEATURES,
    )
    assert partner_indices.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    assert inputs.dtype == torch.float32
    assert partner_indices.dtype == torch.long
    print("    -> Data shapes and types verified.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = RNAModel().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(inputs, partner_indices)

    print(f"    Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    assert not torch.isnan(outputs).any(), "Model produced NaN values!"
    print("    -> Forward pass successful.")

    # --------------------------------------------------------------------------
    # 4. Loss Function
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Loss Function (MaskedMCRMSELoss)...")

    criterion = MaskedMCRMSELoss()
    loss = criterion(outputs, targets)

    print(f"    Loss Value: {loss.item():.6f}")

    # Assertions
    assert loss.dim() == 0, "Loss must be a scalar"
    assert loss.item() >= 0, "Loss must be non-negative"
    print("    -> Loss calculation verified.")

    # --------------------------------------------------------------------------
    # 5. Metric Calculation
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Global Metric (GlobalMCRMSE)...")

    metric = GlobalMCRMSE()
    metric.update(outputs, targets)
    score = metric.compute()

    print(f"    Metric Score: {score:.6f}")

    # Assertions
    assert isinstance(score, float)
    assert score >= 0
    print("    -> Metric calculation verified.")

    # --------------------------------------------------------------------------
    # 6. Training & Evaluation Loop
    # --------------------------------------------------------------------------
    print("\n[6] Verifying Training and Evaluation Steps...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one training epoch (subset)
    print("    Running training step...")
    train_loss = train_fn(model, train_loader, optimizer, criterion, device)
    print(f"    Avg Train Loss: {train_loss:.6f}")

    # Run validation (subset)
    print("    Running evaluation step...")
    val_score = eval_fn(model, val_loader, device)
    print(f"    Val MCRMSE: {val_score:.6f}")

    # Assertions
    assert train_loss > 0
    assert val_score > 0
    print("    -> Engine steps executed successfully.")

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------
    print("\n[7] Cleanup...")
    # Optional: Remove demo directory if desired, but keeping it for inspection
    # shutil.rmtree(Config.WORKING_DIR)
    print(f"    Demo artifacts stored in {Config.WORKING_DIR}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    demo_implementation()
