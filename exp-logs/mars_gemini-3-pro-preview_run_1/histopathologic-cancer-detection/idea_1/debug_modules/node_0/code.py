import os
import sys
import torch
import pandas as pd
import numpy as np
from functools import partial

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ShallowCNN
from library.engine import train_one_epoch, evaluate, run as engine_run
import library.engine  # Import module object for monkeypatching


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration for Speed and Reproducibility
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")
    Config.SEED = 42
    Config.set_seed(Config.SEED)

    # Override Config defaults to ensure rapid execution
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small samples
    Config.DEBUG_SAMPLE_SIZE = None  # We will control this via arguments/patching

    # Ensure working directories exist
    Config.setup_directories()

    print(f"Configured: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Dataset & DataLoader Usage
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Explicitly request a small sample size for verification
    debug_size = 50
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug_sample_size=debug_size, num_workers=0
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch Shape: Images={images.shape}, Labels={labels.shape}")

    # Assertions
    assert len(train_loader.dataset) <= debug_size, "Train dataset size mismatch"
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        48,
        48,
    ), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label shape: {labels.shape}"
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images should be normalized to [0, 1]"

    print("Data loading verification passed.")

    # ---------------------------------------------------------
    # 3. Model Usage
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    model = ShallowCNN().to(device)

    # Forward pass check
    images = images.to(device)
    outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model output shape should be (B, 1)"
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    print("Model verification passed.")

    # ---------------------------------------------------------
    # 4. Training & Evaluation Functions
    # ---------------------------------------------------------
    print("\n[4] Verifying Training and Evaluation Steps...")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test train_one_epoch
    train_loss, train_auc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Single Epoch Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    assert train_loss > 0, "Training loss should be positive"
    assert 0 <= train_auc <= 1, "AUC should be between 0 and 1"

    # Test evaluate
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)
    print(f"Single Epoch Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    assert val_loss > 0, "Validation loss should be positive"

    print("Training/Evaluation step verification passed.")

    # ---------------------------------------------------------
    # 5. Full Pipeline Execution (Engine)
    # ---------------------------------------------------------
    print("\n[5] Running Full Engine Pipeline...")

    # Monkeypatch library.engine.get_dataloaders to enforce debug size
    # This ensures engine.run() uses a small dataset without modifying the source file
    original_get_dataloaders = library.engine.get_dataloaders

    def mocked_get_dataloaders(*args, **kwargs):
        # Force debug_sample_size to be small regardless of Config defaults
        kwargs["debug_sample_size"] = 100
        # Pass other args through
        return get_dataloaders(*args, **kwargs)

    # Apply patch
    library.engine.get_dataloaders = mocked_get_dataloaders

    try:
        # Run the engine (this will train for 1 epoch on 100 samples due to our config and patch)
        engine_run()
    finally:
        # Restore original function (good practice)
        library.engine.get_dataloaders = original_get_dataloaders

    print("Engine run completed.")

    # ---------------------------------------------------------
    # 6. Submission Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Submission File...")

    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {list(df_sub.columns)}")
    print(df_sub.head(3))

    # Assertions
    assert (
        "id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        df_sub["label"].dtype == float or df_sub["label"].dtype == np.float64
    ), "Label column must be float probabilities"
    assert (
        df_sub["label"].min() >= 0 and df_sub["label"].max() <= 1
    ), "Probabilities must be in [0, 1]"

    print("Submission verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
