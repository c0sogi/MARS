import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import RepVGGCactus, RepVGGBlock
from library.trainer import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting Cactus Identification Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Demo
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Create a separate directory for demo outputs to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config class attributes directly
    # Note: Since Config is imported by other modules, these changes propagate
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Reduce compute load for demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print("    Configuration complete.\n")

    # ---------------------------------------------------------
    # 2. Data Loader Verification
    # ---------------------------------------------------------
    print("[2] Initializing DataLoaders...")

    # This will trigger caching in the DEMO_DIR
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    # Assertions
    # Image shape: (Batch, 3, 32, 32)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, 32, 32)}, got {images.shape}"

    # Label shape: DataLoader returns (Batch,) or (Batch, 1) depending on construction.
    # In data_loader.py: return image, torch.tensor(label, dtype=torch.float32)
    # Default collate stacks scalar tensors -> (Batch,)
    assert (
        labels.shape[0] == Config.BATCH_SIZE
    ), f"Expected label batch size {Config.BATCH_SIZE}, got {labels.shape[0]}"

    print("    Data Loading verification PASSED.\n")

    # ---------------------------------------------------------
    # 3. Model Logic Verification (RepVGG)
    # ---------------------------------------------------------
    print("[3] Verifying Model Architecture and Reparameterization...")

    device = Config.DEVICE
    model = RepVGGCactus(num_classes=1).to(device)

    # Forward pass check
    demo_input = images.to(device)
    with torch.no_grad():
        output = model(demo_input)

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {output.shape}"
    print("    Forward pass shape check PASSED.")

    # Verify Structural Re-parameterization Logic
    # We create a single block, run input, switch to deploy, run input, and compare.
    print("    Testing RepVGGBlock fusion logic...")
    block = RepVGGBlock(in_channels=3, out_channels=32, stride=1, deploy=False).to(
        device
    )
    block.eval()

    # Random input
    x = torch.randn(2, 3, 32, 32).to(device)

    with torch.no_grad():
        out_train = block(x)

    # Switch to deploy (fuse branches)
    block.switch_to_deploy()

    with torch.no_grad():
        out_deploy = block(x)

    # Check difference
    diff = (out_train - out_deploy).abs().sum().item()
    print(f"    Difference between Train and Deploy mode outputs: {diff:.6f}")

    # Allow small floating point error
    assert diff < 1e-4, "Reparameterization produced inconsistent results!"
    print("    Reparameterization logic verification PASSED.\n")

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("[4] Executing Training Loop (Short Run)...")

    # Re-initialize model to reset weights
    model = RepVGGCactus(num_classes=1)
    trainer = Trainer(model, device=device)

    # Run training
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify checkpoint creation
    if not os.path.exists(Config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {Config.CHECKPOINT_PATH}")

    print(f"    Checkpoint successfully saved to {Config.CHECKPOINT_PATH}")
    print("    Training loop verification PASSED.\n")

    # ---------------------------------------------------------
    # 5. Inference and Submission
    # ---------------------------------------------------------
    print("[5] Generating Submission with TTA...")

    # We can use the trainer's method which handles loading best weights,
    # reparameterization, and TTA internally.
    trainer.generate_submission(test_loader)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check rows (Test set size is 3325 based on metadata info provided in prompt)
    # We can check against the length of the test_loader dataset
    expected_len = len(test_loader.dataset)
    assert (
        len(df_sub) == expected_len
    ), f"Submission rows {len(df_sub)} != Test set size {expected_len}"

    # Check columns
    expected_cols = ["id", "has_cactus"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Check values are probabilities
    assert (
        df_sub["has_cactus"].min() >= 0.0 and df_sub["has_cactus"].max() <= 1.0
    ), "Predictions are not valid probabilities [0, 1]"

    print(f"    Submission generated at {Config.SUBMISSION_PATH}")
    print(f"    First 3 rows:\n{df_sub.head(3)}")
    print("    Inference verification PASSED.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
