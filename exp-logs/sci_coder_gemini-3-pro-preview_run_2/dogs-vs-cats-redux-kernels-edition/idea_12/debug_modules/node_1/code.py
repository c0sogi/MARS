import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, generate_model_soup, get_logger
from library.dataset import get_metadata, PetDataset, get_transforms
from library.models import PetModel
from library.engine import train_one_epoch, evaluate, predict_tta


def run_demo():
    # =========================================================================
    # 1. Setup & Configuration Override
    # =========================================================================
    print("Step 1: Setting up configuration and environment...")

    # Override Config for a fast demo run
    Config.exp_name = "demo_execution"
    Config.working_dir = os.path.join("./working", Config.exp_name)
    Config.checkpoint_dir = os.path.join(Config.working_dir, "checkpoints")
    Config.oof_dir = os.path.join(Config.working_dir, "oof")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.submission_dir = os.path.join(Config.working_dir, "submission")

    # Setup directories
    Config.setup_directories()

    # Set parameters for speed
    Config.img_size = 128  # Smaller image size for speed
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for tiny demo
    Config.epochs = 1
    Config.device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(Config.seed)
    print(f"  Running on device: {Config.device}")
    print(f"  Working directory: {Config.working_dir}")

    # =========================================================================
    # 2. Data Loading & Dataset Verification
    # =========================================================================
    print("\nStep 2: Loading metadata and creating datasets...")

    # Load metadata
    train_df_full = get_metadata("train")
    val_df_full = get_metadata("val")
    test_df_full = get_metadata("test")

    # Slice to tiny subsets
    train_subset = train_df_full.head(20).reset_index(drop=True)
    val_subset = val_df_full.head(10).reset_index(drop=True)
    test_subset = test_df_full.head(10).reset_index(drop=True)

    print(f"  Train subset size: {len(train_subset)}")
    print(f"  Val subset size: {len(val_subset)}")
    print(f"  Test subset size: {len(test_subset)}")

    # Create Datasets
    train_ds = PetDataset(
        train_subset, mode="train", transforms=get_transforms("train")
    )
    val_ds = PetDataset(val_subset, mode="val", transforms=get_transforms("val"))
    test_ds = PetDataset(test_subset, mode="test", transforms=get_transforms("test"))

    # Verify Dataset Output
    img, target = train_ds[0]
    assert isinstance(img, torch.Tensor), "Dataset image must be a tensor"
    assert img.shape == (
        3,
        Config.img_size,
        Config.img_size,
    ), f"Image shape mismatch. Expected (3, {Config.img_size}, {Config.img_size}), got {img.shape}"
    assert isinstance(target, torch.Tensor), "Target must be a tensor"
    print("  Dataset verification passed: Image and Target shapes are correct.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # =========================================================================
    # 3. Model Instantiation & Verification
    # =========================================================================
    print("\nStep 3: Instantiating Model...")

    # Use a lightweight model for the demo
    model_name = "resnet18"
    model = PetModel(model_name=model_name, pretrained=True)
    model.to(Config.device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.img_size, Config.img_size).to(Config.device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print(f"  Model {model_name} instantiated and forward pass verified.")

    # =========================================================================
    # 4. Training Loop Demonstration
    # =========================================================================
    print("\nStep 4: Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train one epoch
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=Config.device,
        epoch=0,
        mixup_fn=None,  # Skip mixup for simple demo
    )

    print(f"  Epoch 0 Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"

    # Evaluate
    val_metrics = evaluate(model, val_loader, Config.device)
    print(f"  Validation Metrics: {val_metrics}")
    assert "log_loss" in val_metrics
    assert "accuracy" in val_metrics
    assert val_metrics["preds"].shape == (
        len(val_subset),
        1,
    ), "Validation predictions shape mismatch"

    # =========================================================================
    # 5. Inference (TTA) Demonstration
    # =========================================================================
    print("\nStep 5: Running Test-Time Augmentation (TTA) Prediction...")

    preds, ids = predict_tta(model, test_loader, Config.device)

    print(f"  Predictions shape: {preds.shape}")
    print(f"  IDs shape: {ids.shape}")

    assert len(preds) == len(
        test_subset
    ), "Number of predictions does not match test set size"
    assert len(ids) == len(test_subset), "Number of IDs does not match test set size"

    # Save submission
    sub_df = pd.DataFrame({"id": ids, "label": preds})
    sub_path = os.path.join(Config.submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"  Submission saved to {sub_path}")

    # =========================================================================
    # 6. Model Soup Demonstration
    # =========================================================================
    print("\nStep 6: Demonstrating Model Soup...")

    # Save current model as checkpoint A
    ckpt_path_a = os.path.join(Config.checkpoint_dir, "resnet18_fold_0.pth")
    torch.save(model.state_dict(), ckpt_path_a)

    # Modify model weights slightly (simulate another epoch)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.01)

    # Save modified model as checkpoint B
    ckpt_path_b = os.path.join(Config.checkpoint_dir, "resnet18_fold_1.pth")
    torch.save(model.state_dict(), ckpt_path_b)

    # Generate Soup
    soup_save_path = os.path.join(Config.checkpoint_dir, "best_resnet18_soup.pth")
    soup_state = generate_model_soup([ckpt_path_a, ckpt_path_b], soup_save_path)

    # Verification: Check if weights are averaged
    # Load A and B back to CPU
    state_a = torch.load(ckpt_path_a, map_location="cpu")
    state_b = torch.load(ckpt_path_b, map_location="cpu")

    # Pick a random key to verify
    key = list(state_a.keys())[0]
    val_a = state_a[key]
    val_b = state_b[key]
    val_soup = soup_state[key]

    # Expected average
    expected_soup = (val_a + val_b) / 2.0

    # Check closeness
    diff = torch.abs(val_soup - expected_soup).max().item()
    print(f"  Soup Verification Diff: {diff:.8f}")
    assert diff < 1e-6, "Model soup averaging failed"

    print("  Model soup generated and verified successfully.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
