import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config, seed_everything
from library.dataset import (
    CactusDataset,
    get_transforms,
    load_and_cache_data,
    mixup_data,
)
from library.model import SelfEnsemblingRepVGG
from library.utils import SAM
from library.train import train_one_epoch, validate, predict_test


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("Setting up configuration for rapid demonstration...")
    Config.setup()

    # Override Config for a fast debug run
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading and Verification
    print("Loading data...")
    (train_imgs, train_labels), (test_imgs, test_ids) = load_and_cache_data(
        load_cached_data=True
    )

    # Assertions to verify data loading logic
    assert (
        len(train_imgs) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train data size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_imgs)}"
    assert train_imgs.shape[1:] == (
        32,
        32,
        3,
    ), f"Image shape mismatch. Expected (32, 32, 3), got {train_imgs.shape[1:]}"
    assert len(train_labels) == len(train_imgs), "Label count mismatch"

    print("Data loaded and verified.")

    # 3. Dataset and DataLoader Verification
    print("Initializing Datasets and Loaders...")
    train_ds = CactusDataset(
        train_imgs, train_labels, transform=get_transforms("train")
    )
    val_ds = CactusDataset(
        train_imgs, train_labels, transform=get_transforms("valid")
    )  # Reusing train for val in demo
    test_ds = CactusDataset(test_imgs, labels=None, transform=get_transforms("test"))

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify batch structure
    images, targets = next(iter(train_loader))
    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Batch image shape incorrect"
    assert targets.shape == (Config.BATCH_SIZE,), "Batch target shape incorrect"
    print("DataLoader verified.")

    # 4. Model Architecture Verification
    print("Initializing Model...")
    model = SelfEnsemblingRepVGG(num_classes=1, deploy=False).to(device)

    # Test Training Mode (Deep Supervision: Main + Aux heads)
    model.train()
    dummy_input = torch.randn(Config.BATCH_SIZE, 3, 32, 32).to(device)
    out_main, out_aux = model(dummy_input)
    assert out_main.shape == (Config.BATCH_SIZE, 1), "Main head output shape incorrect"
    assert out_aux.shape == (Config.BATCH_SIZE, 1), "Aux head output shape incorrect"

    # Test Eval Mode (Internal Ensemble)
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_input)
    assert out_eval.shape == (Config.BATCH_SIZE, 1), "Eval output shape incorrect"

    print("Model architecture verified.")

    # 5. Optimizer and Training Loop Verification
    print("Initializing Optimizer (SAM)...")
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        model.parameters(), base_optimizer, lr=Config.LEARNING_RATE, rho=0.05
    )
    criterion = nn.BCEWithLogitsLoss()

    print("Running Training Loop (1 Epoch)...")
    # Run one epoch of training
    train_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, epoch=0, device=device
    )
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"
    print(f"Training verified. Loss: {train_loss:.4f}")

    # 6. Validation Verification
    print("Running Validation...")
    val_loss, val_auc = validate(val_loader, model, criterion, device)
    assert 0 <= val_auc <= 1, f"AUC score out of range: {val_auc}"
    print(f"Validation verified. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 7. Model Deployment Switch Verification
    print("Testing RepVGG Deployment Switch...")
    # Save state before switch to compare outputs (roughly)
    model.eval()
    with torch.no_grad():
        pre_deploy_out = model(dummy_input)

    model.switch_to_deploy()
    assert model.deploy is True, "Model deploy flag not set"
    assert not hasattr(
        model.stage1[0], "rbr_dense"
    ), "Branches should be removed after deploy switch"

    with torch.no_grad():
        post_deploy_out = model(dummy_input)

    # Outputs should be very close (numerical precision differences allowed)
    diff = (pre_deploy_out - post_deploy_out).abs().max().item()
    assert diff < 1e-4, f"Deploy switch altered output significantly. Max diff: {diff}"
    print("Deployment switch verified.")

    # 8. Inference and Submission Verification
    print("Running Inference on Test Set...")
    # We pass a list of models to simulate the ensemble, here just one model
    predictions = predict_test(test_loader, [model], device)

    assert len(predictions) == len(test_ids), "Prediction count mismatch"
    assert predictions.shape == (len(test_ids), 1), "Prediction shape mismatch"

    # Create submission dataframe
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": predictions.flatten()})

    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)

    assert os.path.exists(output_path), "Submission file not created"
    print(f"Submission generated at {output_path}")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)
    run_demo()
