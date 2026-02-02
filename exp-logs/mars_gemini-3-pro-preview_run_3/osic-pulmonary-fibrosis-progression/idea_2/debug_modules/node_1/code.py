import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import OSICDataset, get_scalers
from library.model import OSICModel
from library.train import run_training, LaplaceNLLLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== OSIC Library Demonstration Script ===\n")

    # 1. Setup and Reproducibility
    print("[1] Setting up environment and seeds...")
    seed_everything(Config.SEED)
    Config.setup()

    # Ensure we are using a deterministic device setup
    device = Config.DEVICE
    print(f"    Device: {device}")

    # 2. Data Loading and Preprocessing Demonstration
    print("\n[2] Demonstrating Data Loading & Preprocessing...")

    # Load metadata (pre-generated)
    train_df = pd.read_csv(Config.TRAIN_CSV)
    print(f"    Loaded training metadata: {len(train_df)} rows")

    # For demonstration speed, work with a tiny subset
    demo_df = train_df.head(10).copy()

    # Compute scalers based on this subset (normally done on full train set)
    print("    Computing scalers...")
    scalers = get_scalers(demo_df)
    for k, v in scalers.items():
        print(f"      {k}: {v:.4f}")

    # Instantiate Dataset
    print("    Instantiating OSICDataset...")
    dataset = OSICDataset(
        demo_df,
        split_type="train",
        scalers=scalers,
        load_cached_data=False,  # Force processing for demo
    )

    # Verify Dataset Item Structure
    sample = dataset[0]
    print("    Verifying sample structure:")
    print(f"      Keys: {list(sample.keys())}")

    # Assertions for Data Shapes
    # Image: (3, 256, 256) -> 3 channels (Apical, Middle, Basal slices)
    assert sample["image"].shape == (
        3,
        256,
        256,
    ), f"Expected image shape (3, 256, 256), got {sample['image'].shape}"

    # Tabular: (8,) -> 8 features
    assert sample["tabular"].shape == (
        8,
    ), f"Expected tabular shape (8,), got {sample['tabular'].shape}"

    # Target: (1,) -> Scaled FVC
    assert sample["target"].shape == (
        1,
    ), f"Expected target shape (1,), got {sample['target'].shape}"

    print("    Dataset assertions passed.")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Demonstrating Model Architecture...")

    model = OSICModel().to(device)
    model.eval()
    print("    Model initialized successfully.")

    # Create a dummy batch
    batch_size = 2
    dummy_imgs = torch.stack([dataset[i]["image"] for i in range(batch_size)]).to(
        device
    )
    dummy_tabs = torch.stack([dataset[i]["tabular"] for i in range(batch_size)]).to(
        device
    )

    print(f"    Forward pass with batch size {batch_size}...")
    with torch.no_grad():
        outputs = model(dummy_imgs, dummy_tabs)

    # Verify Output Shape: (Batch_Size, 2) -> [FVC_pred, Confidence]
    print(f"    Output shape: {outputs.shape}")
    assert outputs.shape == (
        batch_size,
        2,
    ), f"Expected output shape ({batch_size}, 2), got {outputs.shape}"

    print("    Model forward pass assertions passed.")

    # 4. Loss Function Verification
    print("\n[4] Demonstrating Loss Function...")
    criterion = LaplaceNLLLoss()

    # Dummy targets
    dummy_targets = torch.stack([dataset[i]["target"] for i in range(batch_size)]).to(
        device
    )

    loss = criterion(outputs, dummy_targets)
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() != 0, "Loss is zero!"
    print("    Loss function verification passed.")

    # 5. Full Training Loop Execution (Debug Mode)
    print("\n[5] Running Full Training Loop (Debug Mode)...")
    print("    This utilizes library.train.run_training with debug=True")

    # run_training handles loading, training loop, validation, and checkpointing
    # We set epochs=2 and debug=True to ensure it finishes quickly
    run_training(debug=True, epochs=2, load_cached_data=True)

    # 6. Checkpoint Verification
    print("\n[6] Verifying Output Artifacts...")
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"    SUCCESS: Checkpoint found at {checkpoint_path}")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"    Checkpoint size: {file_size:.2f} MB")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
