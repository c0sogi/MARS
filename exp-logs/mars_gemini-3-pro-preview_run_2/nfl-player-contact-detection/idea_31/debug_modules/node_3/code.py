import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_mcc, optimize_threshold
from library.feature_engineering import FeatureEngineer
from library.dataset import DataProcessor
from library.models import SSERVN
from library.losses import FocalLoss
from library.train_eval import train_epoch, evaluate


def run_demo():
    print("=== 1. Configuration Setup ===")
    # Override Config for a fast demonstration run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Use a tiny subset of data
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small debug batch

    # Initialize environment (creates dirs, sets seeds)
    Config.setup()
    Config.print_config()

    print("\n=== 2. Feature Engineering ===")
    # Instantiate FeatureEngineer
    fe = FeatureEngineer()

    # Generate features for the training split
    # load_cached=False ensures we generate fresh data based on the DEBUG_SAMPLE_SIZE
    print("Generating training features (Debug Mode)...")
    df_train = fe.generate_features(split="train", load_cached=False)

    # Verification
    print(f"Generated DataFrame Shape: {df_train.shape}")
    if df_train.empty:
        raise AssertionError("Feature generation returned an empty DataFrame.")

    expected_cols = ["contact", "game_play", "step", "nfl_player_id_1"]
    for col in expected_cols:
        if col not in df_train.columns:
            raise AssertionError(f"Expected column {col} missing from features.")
    print("Feature Engineering Verification Passed.")

    print("\n=== 3. Dataset Processing ===")
    # Instantiate DataProcessor
    dp = DataProcessor()

    # Create PyTorch Dataset
    # fit_scalers=True because this is the training set
    print("Converting features to PyTorch Dataset...")
    train_dataset = dp.get_dataset(df_train, split="train", fit_scalers=True)

    # Verification
    if len(train_dataset) != len(df_train):
        raise AssertionError("Dataset length does not match source DataFrame.")

    # Inspect a single sample
    sample = train_dataset[0]
    x_kin = sample["x_kin"]
    x_vis = sample["x_vis"]
    x_cat = sample["x_cat"]
    y = sample["y"]

    print(
        f"Sample Shapes -> Kinematic: {x_kin.shape}, Visual: {x_vis.shape}, Categorical: {x_cat.shape}, Target: {y.shape}"
    )

    if not torch.is_tensor(x_kin):
        raise AssertionError("Dataset did not return Tensors.")
    print("Dataset Processing Verification Passed.")

    print("\n=== 4. Model Initialization ===")
    # Determine input dimensions dynamically
    kin_dim = train_dataset.X_kin.shape[1]
    vis_dim = train_dataset.X_vis.shape[1]

    # Determine categorical cardinalities (max index + 1 for embedding layers)
    # In a real scenario, we would compute this across train and val sets
    cat_cardinalities = [
        int(train_dataset.X_cat[:, i].max().item() + 1) for i in range(4)
    ]

    print(f"Input Dims: Kinematic={kin_dim}, Visual={vis_dim}")
    print(f"Categorical Cardinalities: {cat_cardinalities}")

    # Instantiate Model
    device = Config.DEVICE
    model = SSERVN(kin_dim, vis_dim, cat_cardinalities).to(device)
    print("Model instantiated successfully.")

    # Verification: Forward pass with a dummy batch
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    with torch.no_grad():
        logits = model(
            batch["x_kin"].to(device),
            batch["x_vis"].to(device),
            batch["x_cat"].to(device),
        )

    print(f"Forward Pass Output Shape: {logits.shape}")
    if logits.shape != (4, 1):
        raise AssertionError(f"Expected output shape (4, 1), got {logits.shape}")
    print("Model Verification Passed.")

    print("\n=== 5. Training Loop Demonstration ===")
    # Setup Loss and Optimizer
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run a single training epoch
    print("Running 1 Training Epoch...")
    train_loss = train_epoch(model, loader, optimizer, criterion, device)

    print(f"Epoch Complete. Train Loss: {train_loss:.6f}")
    if np.isnan(train_loss):
        raise AssertionError("Training loss is NaN.")
    print("Training Loop Verification Passed.")

    print("\n=== 6. Evaluation & Inference ===")
    # Evaluate on the same loader (just for demonstration)
    print("Evaluating model...")
    val_loss, val_probs, val_targets = evaluate(model, loader, criterion, device)

    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Predictions (First 5): {val_probs[:5]}")
    print(f"Targets (First 5): {val_targets[:5]}")

    # Threshold Optimization
    print("Optimizing Threshold...")
    # We use a small number of steps for speed
    best_thresh, best_mcc = optimize_threshold(val_targets, val_probs, steps=20)

    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Best MCC: {best_mcc:.4f}")

    if not (0 <= best_thresh <= 1):
        raise AssertionError("Optimized threshold is out of bounds [0, 1].")
    print("Evaluation Verification Passed.")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    run_demo()
