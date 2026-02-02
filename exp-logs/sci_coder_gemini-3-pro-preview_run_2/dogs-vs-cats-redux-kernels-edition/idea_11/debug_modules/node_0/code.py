import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.dataset import CatDogDataset, get_transforms
from library.models import get_model
from library.engine import train_one_epoch, valid_one_epoch, train_model, inference_fn


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup
    seed_everything(Config.SEED)

    # Define a temporary working directory for this demo
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)
    checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Working directory: {demo_dir}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Preparation (Sampling for Speed)
    print("\n--- Loading and Sampling Metadata ---")

    # Load metadata
    train_df_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df_full = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df_full = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample a tiny subset for demonstration (e.g., 16 samples)
    # This ensures the code runs in seconds.
    train_subset = train_df_full.sample(n=16, random_state=Config.SEED).reset_index(
        drop=True
    )
    val_subset = val_df_full.sample(n=16, random_state=Config.SEED).reset_index(
        drop=True
    )
    test_subset = test_df_full.sample(n=16, random_state=Config.SEED).reset_index(
        drop=True
    )

    print(f"Train subset size: {len(train_subset)}")
    print(f"Val subset size: {len(val_subset)}")
    print(f"Test subset size: {len(test_subset)}")

    # 3. Dataset and DataLoader
    print("\n--- Initializing Datasets and Loaders ---")

    # Initialize Datasets
    train_dataset = CatDogDataset(
        df=train_subset, transforms=get_transforms("train"), input_dir=Config.INPUT_DIR
    )
    val_dataset = CatDogDataset(
        df=val_subset, transforms=get_transforms("valid"), input_dir=Config.INPUT_DIR
    )
    test_dataset = CatDogDataset(
        df=test_subset, transforms=get_transforms("valid"), input_dir=Config.INPUT_DIR
    )

    # Verification: Check __getitem__
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Dataset should return a tensor image"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    test_img, test_id = test_dataset[0]
    assert isinstance(test_id, int), "Test dataset should return int ID"

    # Initialize DataLoaders
    # Use a small batch size for the demo
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("DataLoaders initialized successfully.")

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    # Using 'resnet18' for speed in this demo, though Config suggests others.
    # We set pretrained=False to avoid downloading weights during this quick check,
    # or True if we want to test that path (usually cached). Let's use False for pure speed.
    model_name = "resnet18"
    model = get_model(model_name, pretrained=False)
    model = model.to(Config.DEVICE)

    # Verification: Check output shape
    dummy_input = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output shape: (batch_size, 1) because NUM_CLASSES=1 in Config
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected ({batch_size}, 1), got {output.shape}"
    print(f"Model {model_name} initialized and verified.")

    # 5. Training Components
    print("\n--- Setting up Optimizer and Scheduler ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # 6. Execution: Train One Epoch
    print("\n--- Testing train_one_epoch ---")
    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epoch=1,
    )
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"
    print(f"Train One Epoch Passed. Loss: {train_loss:.4f}")

    # 7. Execution: Valid One Epoch
    print("\n--- Testing valid_one_epoch ---")
    val_loss, val_log_loss = valid_one_epoch(
        model=model, val_loader=val_loader, device=Config.DEVICE, epoch=1
    )
    assert isinstance(val_loss, float), "Val loss should be a float"
    assert isinstance(val_log_loss, float), "Val log loss should be a float"
    print(f"Valid One Epoch Passed. Loss: {val_loss:.4f}, LogLoss: {val_log_loss:.4f}")

    # 8. Execution: Full Training Loop (Shortened)
    print("\n--- Testing train_model (Short Run) ---")
    save_path = os.path.join(checkpoint_dir, "demo_best_model.pth")

    best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        num_epochs=2,  # Only 2 epochs for demo
        patience=1,  # Strict patience
        save_path=save_path,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print(f"Full training loop finished. Best LogLoss: {best_loss:.4f}")

    # 9. Inference
    print("\n--- Testing inference_fn ---")
    # Load the best saved model state
    model.load_state_dict(torch.load(save_path, map_location=Config.DEVICE))

    preds = inference_fn(model, test_loader, Config.DEVICE)

    assert len(preds) == len(
        test_subset
    ), f"Prediction count mismatch. Expected {len(test_subset)}, got {len(preds)}"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions should be probabilities between 0 and 1"

    # Create submission dataframe
    submission = pd.DataFrame({"id": test_subset["id"], "label": preds})
    print("Inference successful. Sample predictions:")
    print(submission.head())

    # 10. Utility Verification
    print("\n--- Testing Utilities ---")
    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.8, 0.2]
    loss_calc = calculate_log_loss(y_true, y_pred)
    # Expected: - (log(0.9) + log(0.9) + log(0.8) + log(0.8)) / 4 approx 0.16
    assert loss_calc < 1.0, "Log loss calculation seems incorrect for good predictions"
    print(f"Utility calculate_log_loss verified. Result: {loss_calc:.4f}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
