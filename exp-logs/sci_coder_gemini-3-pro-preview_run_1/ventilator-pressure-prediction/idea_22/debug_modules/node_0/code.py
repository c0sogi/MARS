import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import sys

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.features import prepare_data
from library.dataset import VentilatorDataset, get_data_loaders
from library.model import WideProjectedNet
from library.loss import MaskedL1Loss
from library.train import Trainer


def create_mini_dataset(source_path, dest_path, num_rows=8000):
    """
    Reads a subset of the csv, ensures complete breaths, and saves to dest.
    """
    # Read a chunk
    df = pd.read_csv(source_path, nrows=num_rows)

    # The dataset is grouped by breath_id.
    # To ensure we don't have a partial breath at the end, we drop the last breath_id found.
    last_breath_id = df["breath_id"].iloc[-1]
    df_subset = df[df["breath_id"] != last_breath_id].copy()

    # Verify we have data
    assert len(df_subset) > 0, f"Subset creation failed for {source_path}"

    # Save
    df_subset.to_csv(dest_path, index=False)
    print(f"Created mini dataset at {dest_path}: {df_subset.shape}")
    return df_subset


def run_demo():
    # 1. Setup
    print("=== 1. Setup & Configuration ===")
    seed_everything(42)

    # Define demo paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    mini_train_path = os.path.join(demo_dir, "train.csv")
    mini_val_path = os.path.join(demo_dir, "val.csv")
    mini_test_path = os.path.join(demo_dir, "test.csv")

    # 2. Create Mini Datasets (Speed Optimization)
    print("\n=== 2. Creating Mini Datasets ===")
    # We need enough rows to form batches. 80 rows per breath.
    # 8000 rows = 100 breaths approx.
    create_mini_dataset(Config.TRAIN_PATH, mini_train_path, num_rows=8000)
    create_mini_dataset(Config.VAL_PATH, mini_val_path, num_rows=4000)
    create_mini_dataset(Config.TEST_PATH, mini_test_path, num_rows=4000)

    # 3. Override Config
    print("\n=== 3. Overriding Configuration ===")
    Config.TRAIN_PATH = mini_train_path
    Config.VAL_PATH = mini_val_path
    Config.TEST_PATH = mini_test_path
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce compute load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Re-setup directories based on new config
    Config.setup()

    # 4. Feature Engineering & Data Preparation
    print("\n=== 4. Feature Engineering & Data Loading ===")
    # Force recompute to test logic
    train_data, val_data, test_data = prepare_data(load_cached_data=False)

    train_X, train_y, train_uout = train_data
    test_X, test_ids, test_uout = test_data

    # Assertions
    print("Verifying Data Shapes...")
    # Shape should be (N_breaths, 80, N_features)
    assert len(train_X.shape) == 3
    assert train_X.shape[1] == 80
    assert train_y.shape == (train_X.shape[0], 80)
    assert train_uout.shape == (train_X.shape[0], 80)
    assert test_ids.shape == (test_X.shape[0], 80)

    print(f"Train X Shape: {train_X.shape}")
    print(f"Test X Shape: {test_X.shape}")

    # 5. DataLoader Verification
    print("\n=== 5. DataLoader Verification ===")
    train_loader, val_loader, test_loader = get_data_loaders(
        load_cached_data=True,  # Should pick up the cache we just made
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    batch = next(iter(train_loader))
    assert "X" in batch and "y" in batch and "u_out" in batch
    assert batch["X"].shape[0] == Config.BATCH_SIZE
    assert batch["X"].shape[1] == 80

    input_dim = batch["X"].shape[-1]
    print(f"Input Feature Dimension: {input_dim}")

    # 6. Model & Loss Verification
    print("\n=== 6. Model & Loss Logic Verification ===")
    device = get_device()
    model = WideProjectedNet(input_dim=input_dim).to(device)
    criterion = MaskedL1Loss()

    # Forward pass check
    x_sample = batch["X"].to(device)
    y_sample = batch["y"].to(device)
    u_out_sample = batch["u_out"].to(device)

    final_pred, aux_pred = model(x_sample)

    assert final_pred.shape == (Config.BATCH_SIZE, 80, 1)
    if aux_pred is not None:
        assert aux_pred.shape == (Config.BATCH_SIZE, 80, 1)

    # Loss check
    loss = criterion((final_pred, aux_pred), y_sample, u_out_sample)
    assert loss.dim() == 0  # Scalar
    assert loss.requires_grad
    print(f"Initial Loss: {loss.item():.4f}")

    # 7. Training Loop Execution
    print("\n=== 7. Training Execution ===")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=Config.EPOCHS, steps_per_epoch=len(train_loader)
    )

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=Config,
    )

    # Capture weights before training to verify update
    initial_weights = model.head.weight.clone()

    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify weights updated
    final_weights = model.head.weight
    assert not torch.equal(
        initial_weights, final_weights
    ), "Model weights did not update!"

    # Verify model saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not found!"

    # 8. Inference Demonstration
    print("\n=== 8. Inference Demonstration ===")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    predictions = []
    ids_list = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["X"].to(device)
            ids = batch["id"]  # CPU tensor

            preds, _ = model(x)
            # Flatten predictions (B, 80, 1) -> (B*80)
            preds_flat = preds.squeeze(-1).cpu().numpy().flatten()
            ids_flat = ids.numpy().flatten()

            predictions.extend(preds_flat)
            ids_list.extend(ids_flat)

    # Create submission dataframe
    sub_df = pd.DataFrame({"id": ids_list, "pressure": predictions})

    # Verify submission format
    assert sub_df.shape[1] == 2
    assert "id" in sub_df.columns and "pressure" in sub_df.columns

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission generated at {Config.SUBMISSION_PATH} with shape {sub_df.shape}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
