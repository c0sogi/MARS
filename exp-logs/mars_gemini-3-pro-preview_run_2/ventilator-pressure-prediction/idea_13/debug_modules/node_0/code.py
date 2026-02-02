import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, WeightedL1Loss, save_checkpoint, load_checkpoint
from library.data_loader import get_dataloaders, VentilatorDataset
from library.model import DGC_BiLSTM
from library.train import train_one_epoch, validate


def setup_demo_config():
    """
    Overrides the default configuration for the purpose of this quick demonstration.
    """
    print(">>> Setting up Demo Configuration...")

    # 1. Set a unique working directory for this demo to avoid cache conflicts
    demo_dir = "./working/demo_execution_script"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir

    # 2. Update dependent paths (since they are static class attributes initialized at import)
    Config.TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "train_processed_debug.parquet"
    )
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_processed_debug.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_processed_debug.parquet")
    Config.SCALER_CACHE = os.path.join(Config.WORKING_DIR, "scaler_params_debug.npy")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # 3. Set Hyperparameters for speed
    Config.DEBUG = True  # Use subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce overhead
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.INJECTION_DIM = 16
    Config.NUM_LAYERS = 2

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")


def demonstrate_data_loading():
    """
    Demonstrates getting dataloaders and verifying batch shapes.
    """
    print("\n>>> Demonstrating Data Loading...")

    # Force reload by ensuring cache doesn't exist (handled by rmtree above)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=False
    )

    # Fetch one batch
    features, targets, u_out = next(iter(train_loader))

    print(f"Feature Batch Shape: {features.shape}")
    print(f"Target Batch Shape: {targets.shape}")
    print(f"u_out Batch Shape: {u_out.shape}")

    # Assertions
    # Expected shape: (Batch_Size, Seq_Len=80, Features=12)
    assert features.shape == (
        Config.BATCH_SIZE,
        80,
        12,
    ), f"Unexpected feature shape: {features.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Unexpected target shape: {targets.shape}"
    assert u_out.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Unexpected u_out shape: {u_out.shape}"

    # Verify Data Types
    assert features.dtype == torch.float32
    assert targets.dtype == torch.float32

    return train_loader, val_loader, test_loader


def demonstrate_model_forward(train_loader):
    """
    Demonstrates model instantiation and a forward pass.
    """
    print("\n>>> Demonstrating Model Forward Pass...")

    device = torch.device(Config.DEVICE)
    model = DGC_BiLSTM().to(device)

    # Get a batch
    features, _, _ = next(iter(train_loader))
    features = features.to(device)

    # Forward pass
    preds = model(features)

    print(f"Prediction Shape: {preds.shape}")

    # Assertions
    # Output should be (Batch_Size, Seq_Len) - squeeze(-1) is done in model
    assert preds.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Unexpected prediction shape: {preds.shape}"

    return model


def demonstrate_loss_function():
    """
    Demonstrates the WeightedL1Loss logic.
    """
    print("\n>>> Demonstrating Loss Function...")

    criterion = WeightedL1Loss()

    # Create dummy data
    # Preds: all 10
    # Targets: all 0
    # Error is 10 everywhere
    preds = torch.full((2, 80), 10.0)
    targets = torch.zeros((2, 80))

    # u_out: First half 0 (Inspiratory), Second half 1 (Expiratory)
    u_out = torch.zeros((2, 80))
    u_out[:, 40:] = 1.0

    # Calculate Loss
    loss = criterion(preds, targets, u_out)

    # Manual Calculation
    # Insp Weight = 1.0, Exp Weight = 0.1
    # Insp Error = 10 * 1.0 = 10
    # Exp Error = 10 * 0.1 = 1
    # Mean = (10 * 40 + 1 * 40) / 80 = (400 + 40) / 80 = 440 / 80 = 5.5

    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    expected_loss = 5.5
    assert (
        abs(loss.item() - expected_loss) < 1e-5
    ), f"Loss mismatch. Expected {expected_loss}, got {loss.item()}"

    print("Loss function logic verified.")


def demonstrate_training_loop(model, train_loader, val_loader):
    """
    Demonstrates a single epoch of training and validation.
    """
    print("\n>>> Demonstrating Training Loop...")

    device = torch.device(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = WeightedL1Loss()

    # Train one epoch
    print("Running training step...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")

    assert train_loss > 0, "Training loss should be positive"

    # Validate
    print("Running validation step...")
    val_metrics = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation MAE (Insp): {val_metrics['mae_inspiratory']:.4f}")

    assert val_metrics["mae_inspiratory"] >= 0, "MAE should be non-negative"

    # Save Checkpoint
    print("Saving checkpoint...")
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        is_best=True,
    )

    assert os.path.exists(Config.BEST_MODEL_PATH), "Checkpoint file was not created."


def demonstrate_inference(model, test_loader):
    """
    Demonstrates inference on the test set.
    """
    print("\n>>> Demonstrating Inference...")

    device = torch.device(Config.DEVICE)
    model.eval()

    all_preds = []

    # Run inference on a few batches
    with torch.no_grad():
        for i, (features, _, _) in enumerate(test_loader):
            features = features.to(device)
            preds = model(features)
            all_preds.append(preds.cpu().numpy())
            if i >= 2:
                break  # Limit to 3 batches for demo

    all_preds = np.concatenate(all_preds, axis=0)
    print(f"Generated predictions for {all_preds.shape[0]} breaths.")
    print(
        f"Prediction stats - Mean: {np.mean(all_preds):.4f}, Std: {np.std(all_preds):.4f}"
    )

    # Flatten for submission format check
    flat_preds = all_preds.flatten()

    # Create a dummy submission file
    submission = pd.DataFrame(
        {"id": np.arange(1, len(flat_preds) + 1), "pressure": flat_preds}
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    assert os.path.exists(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    # 0. Set Seed
    set_seed(Config.SEED)

    # 1. Setup Config
    setup_demo_config()

    # 2. Data Loading
    train_loader, val_loader, test_loader = demonstrate_data_loading()

    # 3. Model & Forward
    model = demonstrate_model_forward(train_loader)

    # 4. Loss Logic
    demonstrate_loss_function()

    # 5. Training Loop
    demonstrate_training_loop(model, train_loader, val_loader)

    # 6. Inference
    demonstrate_inference(model, test_loader)

    print("\n>>> Demonstration Complete Successfully.")
