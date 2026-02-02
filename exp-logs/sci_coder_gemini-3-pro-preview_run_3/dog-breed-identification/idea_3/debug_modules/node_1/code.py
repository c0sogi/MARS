import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter
from library.dataset import get_dataloaders
from library.model import build_model
from library.train import train_one_epoch, validate, predict_tta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.debug = True
    Config.debug_sample_size = 64  # Small subset for quick execution
    Config.batch_size = 8  # Small batch size
    Config.epochs = 1
    Config.warmup_epochs = 0
    Config.working_dir = "./working/demo_run"
    Config.best_model_path = os.path.join(Config.working_dir, "demo_best_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "demo_submission.csv")

    # Ensure demo directory exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Working Directory: {Config.working_dir}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[2] Demonstrating Data Loading...")

    # Load data
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=False
    )

    # Validation assertions
    assert (
        len(classes) == Config.num_classes
    ), f"Expected {Config.num_classes} classes, got {len(classes)}"
    print(f"    Classes loaded: {len(classes)}")

    # Check Train Loader
    images, targets = next(iter(train_loader))
    print(f"    Train Batch Shape - Images: {images.shape}, Targets: {targets.shape}")
    assert images.shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    ), "Incorrect image batch shape"
    assert targets.shape == (Config.batch_size,), "Incorrect target batch shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.long, "Targets should be long (int64)"

    # Check Test Loader (returns images and ids)
    test_images, test_ids = next(iter(test_loader))
    print(f"    Test Batch Shape - Images: {test_images.shape}, IDs: {len(test_ids)}")
    assert len(test_ids) == Config.batch_size, "Incorrect number of test IDs"

    # ==========================================
    # 3. Model Building
    # ==========================================
    print("\n[3] Demonstrating Model Building...")

    model = build_model(
        num_classes=Config.num_classes, pretrained=True
    )  # Use pretrained for demo
    model = model.to(device)

    # Verify model structure
    assert isinstance(model, nn.Module), "Model is not a torch.nn.Module"

    # Test Forward Pass
    dummy_input = images.to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.batch_size,
        Config.num_classes,
    ), "Model output shape mismatch"

    # ==========================================
    # 4. Training & Validation Loop
    # ==========================================
    print("\n[4] Demonstrating Training and Validation Steps...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch of training
    print("    Running training step (1 epoch on subset)...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )
    print(f"    Training Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Run validation
    print("    Running validation step...")
    val_loss = validate(model, val_loader, criterion, device)
    print(f"    Validation Loss: {val_loss:.4f}")
    assert val_loss > 0, "Validation loss should be positive"

    # ==========================================
    # 5. Checkpointing
    # ==========================================
    print("\n[5] Demonstrating Checkpointing...")

    # Save checkpoint
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": val_loss,
    }
    save_checkpoint(state, is_best=True, filename=Config.best_model_path)
    assert os.path.exists(Config.best_model_path), "Checkpoint file was not created"

    # Load checkpoint
    print("    Loading checkpoint...")
    loaded_epoch, loaded_metric = load_checkpoint(
        model, Config.best_model_path, optimizer
    )
    print(f"    Loaded Epoch: {loaded_epoch}, Metric: {loaded_metric:.4f}")
    assert loaded_epoch == 1, "Loaded epoch mismatch"
    assert loaded_metric == val_loss, "Loaded metric mismatch"

    # ==========================================
    # 6. Inference
    # ==========================================
    print("\n[6] Demonstrating Inference (TTA)...")

    df_submission = predict_tta(model, test_loader, device, classes)

    # Verify Submission DataFrame
    print(f"    Submission Shape: {df_submission.shape}")
    print(f"    Columns: {df_submission.columns[:5].tolist()} ...")

    # Check dimensions: rows = num_test_samples (limited by debug size), cols = id + 120 breeds
    # Note: Dataset debug sampling might not be exactly batch_size aligned if total < batch_size,
    # but here we requested 64 samples.
    expected_rows = min(
        len(pd.read_csv("./metadata/test.csv")), Config.debug_sample_size
    )
    assert (
        df_submission.shape[1] == Config.num_classes + 1
    ), "Incorrect number of columns in submission"
    assert "id" in df_submission.columns, "Submission missing 'id' column"

    # Verify probabilities sum to 1
    # Drop ID column, sum across rows
    probs = df_submission.drop(columns=["id"]).values
    row_sums = np.sum(probs, axis=1)

    # Allow small floating point error
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"
    print("    Probability check passed (Sum ~ 1.0)")

    # Save submission
    df_submission.to_csv(Config.submission_path, index=False)
    print(f"    Submission saved to {Config.submission_path}")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demo()
