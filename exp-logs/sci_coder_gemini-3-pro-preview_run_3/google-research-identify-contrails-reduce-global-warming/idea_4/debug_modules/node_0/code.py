import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import get_dataloader, ContrailDataset
from library.model import ContrailUNetPlusPlus
from library.loss import BCEDiceLoss
from library.checkpointing import CheckpointManager
from library.engine import fit, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("=== Starting Contrail Detection Pipeline Demonstration ===\n")

    # 1. Setup and Configuration Override
    # We override defaults to make this run fast as a demo
    print("[1/7] Configuring environment...")
    set_seed(42)
    device = get_device()
    print(f"      Device: {device}")

    # Override Config for speed
    Config.DEBUG = True  # Use small subset of data (50 samples)
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers for simple script

    # Ensure working directory exists (handled by Config usually, but double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print("      Configuration updated for fast demonstration.")

    # 2. Data Loading Demonstration
    print("\n[2/7] Initializing DataLoaders...")

    # Create DataLoaders
    train_loader = get_dataloader("train", batch_size=Config.BATCH_SIZE)
    val_loader = get_dataloader("validation", batch_size=Config.BATCH_SIZE)
    test_loader = get_dataloader("test", batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"      Train Loader batches: {len(train_loader)}")
    print(f"      Val Loader batches:   {len(val_loader)}")
    print(f"      Test Loader batches:  {len(test_loader)}")

    # Verify a single batch structure
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]

    # Assertions to verify data pipeline
    assert images.dim() == 4, "Image batch should be 4D (B, C, H, W)"
    assert masks.dim() == 4, "Mask batch should be 4D (B, C, H, W)"
    assert (
        images.shape[1] == Config.IN_CHANNELS
    ), f"Expected {Config.IN_CHANNELS} input channels"
    assert masks.shape[1] == 1, "Expected 1 mask channel"
    assert images.shape[2:] == (
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect spatial dimensions"

    print(f"      Verified Batch Shapes - Image: {images.shape}, Mask: {masks.shape}")

    # 3. Model Initialization and Verification
    print("\n[3/7] Initializing Model...")
    model = ContrailUNetPlusPlus()
    model.to(device)

    # Verify Forward Pass (Training Mode - Deep Supervision)
    model.train()
    with torch.no_grad():
        dummy_input = torch.randn(
            2, Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE
        ).to(device)
        train_outputs = model(dummy_input)

        assert isinstance(
            train_outputs, list
        ), "Model in train mode should return a list (Deep Supervision)"
        assert len(train_outputs) == 4, "Expected 4 output heads for Deep Supervision"
        assert train_outputs[0].shape == (
            2,
            1,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), "Output shape mismatch"
        print("      Training forward pass successful (Deep Supervision active).")

    # Verify Forward Pass (Eval Mode)
    model.eval()
    with torch.no_grad():
        eval_output = model(dummy_input)
        assert isinstance(
            eval_output, torch.Tensor
        ), "Model in eval mode should return a single Tensor"
        assert eval_output.shape == (
            2,
            1,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), "Eval output shape mismatch"
        print("      Evaluation forward pass successful.")

    # 4. Loss Function Demonstration
    print("\n[4/7] Testing Loss Function...")
    criterion = BCEDiceLoss()

    # Create dummy logits and targets
    dummy_logits = torch.randn(2, 1, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    dummy_targets = (
        torch.randint(0, 2, (2, 1, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        .float()
        .to(device)
    )

    loss = criterion(dummy_logits, dummy_targets)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"      Calculated Dummy Loss: {loss.item():.4f}")

    # 5. Training Loop Demonstration
    print("\n[5/7] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Checkpoint Manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=os.path.join(Config.WORKING_DIR, "checkpoints"),
        top_k=1,
        mode="max",
    )

    # Run Fit
    # Note: Scheduler is optional, passing None for simplicity in this short demo
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
        checkpoint_manager=ckpt_manager,
        patience=1,
    )
    print("      Training loop completed.")

    # 6. Inference and Submission
    print("\n[6/7] Generating Submission...")

    # Define output path
    submission_path = Config.SUBMISSION_PATH

    # Run prediction
    predict_and_submit(model, test_loader, device, output_path=submission_path)

    # 7. Verify Submission File
    print("\n[7/7] Verifying Submission File...")
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"      Submission file found at: {submission_path}")
        print(f"      Rows: {len(df_sub)}")
        print(f"      Columns: {list(df_sub.columns)}")

        # Basic content check
        assert "record_id" in df_sub.columns
        assert "encoded_pixels" in df_sub.columns
        # Check if we have rows (Config.DEBUG limits dataset, but test set might be loaded fully or partially depending on logic)
        # In dataset.py, debug limits all splits.
        assert len(df_sub) > 0, "Submission file is empty"
        print("      Submission format verified.")
    else:
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demonstration()
