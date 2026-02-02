import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import UltraWideSERepNeXt
from library.engine import train_model, predict


def main():
    print("Starting Cactus Identification Demo...")

    # 1. Setup & Configuration Overrides
    # We override Config values to ensure the demo runs quickly and writes to a specific demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 32  # Small batch size for demo
    Config.NUM_WORKERS = 2
    Config.SEEDS = [42]  # Only run one seed for demo

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.BASE_SEED)
    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")

    # 2. Data Loading & Verification
    print("\n--- Step 1: Data Loading & Verification ---")
    # We use load_cached_data=False to demonstrate the processing logic,
    # though in production True is preferred.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    try:
        images, labels, ids = next(iter(train_loader))
        print(
            f"Train Batch - Images Shape: {images.shape}, Labels Shape: {labels.shape}"
        )

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            32,
            32,
        ), "Incorrect image batch shape"
        assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
        assert images.dtype == torch.float32, "Images should be float tensors"
        # Check normalization (approximate range check, though standardization makes strict bounds hard)
        # Just ensuring it's not raw 0-255 uint8
        assert (
            images.max() <= 50.0 and images.min() >= -50.0
        ), "Images appear unnormalized"
        print("Data loading verification passed.")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 3. Model Instantiation & Logic Check
    print("\n--- Step 2: Model Instantiation & Logic Check ---")
    device = Config.DEVICE
    # Instantiate model with fewer blocks for speed in this demo
    model = UltraWideSERepNeXt(num_blocks=[1, 1, 1])
    model.to(device)

    # Forward pass verification
    dummy_input = images.to(device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    # Verify RepNeXt Deployment Switch (Fusion)
    print("Verifying RepNeXt 'switch_to_deploy' fusion...")
    # Save state before switch
    output_before = output.clone()

    # Switch to deploy (fuses conv+bn, branches)
    model.switch_to_deploy()
    model.eval()
    with torch.no_grad():
        output_after = model(dummy_input)

    # Check difference
    diff = (output_before - output_after).abs().max().item()
    print(f"Max difference after fusion: {diff:.6f}")

    # Tolerance is usually very low (< 1e-4), but float precision can vary.
    assert diff < 1e-3, "Fusion resulted in significant output deviation!"
    print("Model logic verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- Step 3: Training Loop Demonstration ---")
    # Re-instantiate model for training (since previous one is in deploy mode)
    model = UltraWideSERepNeXt(num_blocks=[1, 1, 1])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    checkpoint_name = "model_seed_42.pth"

    # Run training for 1 epoch, limiting to 5 batches to save time
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        filename=checkpoint_name,
        max_batches=5,  # Limit batches for demo speed
    )

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, checkpoint_name)
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print(f"Training demonstration complete. Checkpoint saved to {checkpoint_path}")

    # 5. Inference & Submission
    print("\n--- Step 4: Inference & Submission ---")

    # Load the best model
    loaded_model = UltraWideSERepNeXt(num_blocks=[1, 1, 1])
    loaded_model.to(device)
    load_checkpoint(loaded_model, checkpoint_name, device=device)

    # Predict
    # We use the full test loader here, it's small enough (~3k images)
    ids, probs = predict(loaded_model, test_loader, device)

    print(f"Generated {len(ids)} predictions.")

    # Assertions
    assert len(ids) == len(probs), "Mismatch between IDs and probabilities count"
    assert len(ids) > 0, "No predictions generated"
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of [0, 1] range"

    # Create Submission
    submission_df = pd.DataFrame({"id": ids, "has_cactus": probs})

    sub_path = Config.SUBMISSION_PATH
    submission_df.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print("Head of submission:")
    print(submission_df.head())

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
