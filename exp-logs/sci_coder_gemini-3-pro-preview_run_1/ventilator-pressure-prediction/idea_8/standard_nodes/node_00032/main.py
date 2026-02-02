import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data_processing import prepare_data
from library.dataset import VentilatorDataset
from library.model import VentilatorNet
from library.train import train_epoch, validate_epoch, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load cached data (or process if not available).
    # This uses the 'Physics-Fidelity' features defined in data_processing.py
    print("Loading data...")
    train_x, train_y, val_x, val_y, test_x, test_ids = prepare_data(
        load_cached_data=True
    )

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)

    # Create DataLoaders
    # Pin memory enables faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Initialize the Dual-Gated Multi-Scale CNN-LSTM
    input_dim = train_x.shape[2]
    model = VentilatorNet(input_dim=input_dim).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler for super-convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Training Loop
    best_mae = float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_mae = validate_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Final Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()

    # Collect predictions and inputs for analysis
    val_inputs_list = []
    val_preds_list = []
    val_targets_list = []
    val_u_out_list = []

    # Run inference on validation set manually to gather tensors
    with torch.no_grad():
        for X, y, u_out in val_loader:
            X = X.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            preds = model(X)

            val_inputs_list.append(X.cpu())
            val_preds_list.append(preds.cpu())
            val_targets_list.append(y.cpu())
            val_u_out_list.append(u_out.cpu())

    val_inputs = torch.cat(val_inputs_list)
    val_preds = torch.cat(val_preds_list)
    val_targets = torch.cat(val_targets_list)
    val_u_out = torch.cat(val_u_out_list)

    # Calculate Final Metric (MAE on inspiratory phase)
    mask = val_u_out == 0
    abs_errors = torch.abs(val_preds - val_targets)
    masked_errors = abs_errors[mask]

    final_metric = masked_errors.mean().item()
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Features
    print("Performing Failure Analysis...")

    # Flatten inputs and errors based on mask to analyze only relevant steps
    flat_inputs = val_inputs[mask].numpy()
    flat_errors = masked_errors.numpy()

    correlations = []
    num_features = flat_inputs.shape[1]

    for i in range(num_features):
        feat_vals = flat_inputs[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, flat_errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Feature Correlations with Error Magnitude:")
    for i, corr in correlations[:5]:
        print(f"  Feature Index {i}: {corr:.4f}")

    # 7. Submission Generation
    THRESHOLD = 0.3096454441547394
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Prepare Test Loader
        test_dataset = VentilatorDataset(test_x, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Inference
        preds_matrix = inference(model, test_loader, device)
        preds_flat = preds_matrix.flatten()

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "pressure": preds_flat})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
