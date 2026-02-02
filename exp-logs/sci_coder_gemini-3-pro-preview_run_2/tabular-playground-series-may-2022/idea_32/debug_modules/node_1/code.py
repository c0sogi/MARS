import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders, process_and_cache_data
from library.modules import DualStemResFunnelModel
from library.engine import (
    get_optimizer_params,
    initialize_weights,
    train_one_epoch,
    evaluate,
)
import library.network as network


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def setup_demo_config():
    """
    Overrides the default Config parameters to ensure the demo runs quickly
    and uses a separate working directory.
    """
    print("--- Configuring Demo Environment ---")

    # Use a specific directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths
    Config.CACHE_FILE_PATH = os.path.join(Config.WORKING_DIR, "processed_data_demo.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Enable Debug mode to use a small subset of data (5000 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small enough for fast demo

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128

    # Disable multiprocessing for simple script execution to avoid overhead
    Config.NUM_WORKERS = 0

    # Ensure device is set correctly
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")


def verify_data_pipeline():
    """
    Demonstrates and verifies the data loading and processing logic.
    """
    print("\n--- Verifying Data Pipeline ---")

    # Force reprocessing to ensure we test the logic
    if os.path.exists(Config.CACHE_FILE_PATH):
        os.remove(Config.CACHE_FILE_PATH)

    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Verify Loader lengths
    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))

    # Check keys
    assert "continuous" in batch, "Batch missing 'continuous' key"
    assert "sequence" in batch, "Batch missing 'sequence' key"
    assert "target" in batch, "Batch missing 'target' key"

    continuous = batch["continuous"]
    sequence = batch["sequence"]
    target = batch["target"]

    # Check Shapes
    # Continuous: (Batch, 30)
    assert continuous.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CONT_FEATURES,
    ), f"Continuous shape mismatch: {continuous.shape}"

    # Sequence: (Batch, 10)
    assert sequence.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Sequence shape mismatch: {sequence.shape}"

    # Target: (Batch,)
    assert target.shape == (
        Config.BATCH_SIZE,
    ), f"Target shape mismatch: {target.shape}"

    print("Data Pipeline verification passed: Shapes are correct.")
    return train_loader, val_loader


def verify_model_logic(train_loader):
    """
    Demonstrates model instantiation, forward pass, and backward pass.
    """
    print("\n--- Verifying Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = DualStemResFunnelModel(config=Config).to(device)

    # Get a batch
    batch = next(iter(train_loader))
    continuous = batch["continuous"].to(device)
    sequence = batch["sequence"].to(device)
    target = batch["target"].to(device).unsqueeze(1)  # (B, 1)

    # Forward Pass
    logits = model(continuous, sequence)

    # Check Output Shape
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Logits shape mismatch: {logits.shape}, expected ({Config.BATCH_SIZE}, 1)"

    # Backward Pass Verification (Gradient Check)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, target)
    loss.backward()

    # Check if gradients are populated for a key layer
    assert model.head.weight.grad is not None, "Gradients not computed for head layer"

    print("Model verification passed: Forward/Backward successful.")
    return model


def verify_engine_functions(model, train_loader, val_loader):
    """
    Demonstrates utility functions in engine.py: initialization, optimizer setup, training loop.
    """
    print("\n--- Verifying Engine Functions ---")
    device = torch.device(Config.DEVICE)

    # 1. Initialize Weights
    initialize_weights(model)
    print("Weights initialized.")

    # 2. Optimizer Setup
    optimizer_groups = get_optimizer_params(
        model,
        weight_decay_encoder=Config.WEIGHT_DECAY_ENCODER,
        weight_decay_bias=Config.WEIGHT_DECAY_BIAS,
    )
    optimizer = optim.AdamW(optimizer_groups, lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Train One Epoch (Shortened for demo)
    print("Running training step...")
    avg_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=0
    )
    print(f"Average Training Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # 4. Evaluate
    print("Running evaluation...")
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    assert 0.0 <= val_auc <= 1.0, "AUC score out of bounds"

    print("Engine verification passed.")


def run_full_pipeline():
    """
    Executes the full pipeline defined in network.py using the modified config.
    """
    print("\n--- Running Full Network Pipeline (network.py) ---")

    # We invoke the run method. Since we modified Config globally,
    # network.py will use our settings (DEBUG=True, EPOCHS=1).
    try:
        network.run()
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise e

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df.shape}")

        # Check format
        assert list(df.columns) == ["id", "target"], "Submission columns incorrect"
        assert len(df) > 0, "Submission file is empty"
        print("Full pipeline verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    setup_demo_config()

    # 2. Data Verification
    train_loader, val_loader = verify_data_pipeline()

    # 3. Model Verification
    model = verify_model_logic(train_loader)

    # 4. Engine Verification
    verify_engine_functions(model, train_loader, val_loader)

    # 5. Full Pipeline Execution
    # Note: This will re-initialize the model and run the training loop
    # defined in network.py from scratch.
    run_full_pipeline()

    print("\n=== All Demonstrations Completed Successfully ===")
