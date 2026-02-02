import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.utils import seed_everything, get_device, compute_metric
from library.dataset import get_data_loaders
from library.model import HybridModel, train_epoch, validate_epoch
from library.inference import generate_predictions


def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    SEED = 42
    BATCH_SIZE = 512  # Increased for A100 efficiency
    EPOCHS = 15  # Reduced for fast baseline execution
    LR = 1e-3
    WORK_DIR = "./working/runfile_execution"
    MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    METRIC_THRESHOLD = 0.36414578557014465

    seed_everything(SEED)
    device = get_device()
    os.makedirs(WORK_DIR, exist_ok=True)

    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load cached data to save time
    # Set load_cached_data=False to invalidate stale debug cache and force full data processing. Cite debug_lesson_1
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE, load_cached_data=False, debug=False
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    # Architecture params match library/model.py defaults
    model = HybridModel(
        input_dim=12, lstm_dim=512, num_lstm_layers=4, emb_dim=8, cnn_dim=256
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.L1Loss()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_mae = float("inf")

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Training finished. Best Val MAE: {best_mae}")

    # ==========================================
    # 5. Final Validation & Failure Analysis
    # ==========================================
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []
    all_u_out = []

    # Collect all validation data
    with torch.no_grad():
        for batch in val_loader:
            X = batch["input"].to(device)
            y = batch["target"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(X)

            # Move to CPU for analysis to save GPU memory
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_inputs.append(X.cpu())
            all_u_out.append(u_out.cpu())

    # Concatenate
    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    inputs_cat = torch.cat(all_inputs)
    u_out_cat = torch.cat(all_u_out)

    # Compute Final Metric
    final_metric = compute_metric(preds_cat, targets_cat, u_out_cat)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error per time step
    abs_errors = torch.abs(preds_cat - targets_cat)

    # Flatten tensors for correlation calculation: (N_samples * 80)
    errors_flat = abs_errors.flatten().numpy()
    # Inputs shape: (N, 80, 14) -> Flatten to (N*80, 14)
    inputs_flat = inputs_cat.flatten(0, 1).numpy()

    # Feature names corresponding to dataset.py processing order:
    # Scale cols (11) + u_out + R_cat + C_cat
    feature_names = [
        "time_step",
        "u_in",
        "u_in_cumsum",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag_back1",
        "u_in_lag_back2",
        "u_in_diff1",
        "u_in_diff2",
        "R_u_in",
        "vol_C",
        "u_out",
        "R_cat",
        "C_cat",
    ]

    print("Correlation between Input Features and Absolute Error:")
    for i, name in enumerate(feature_names):
        # Calculate Pearson correlation
        feat_values = inputs_flat[:, i]
        if np.std(feat_values) == 0 or np.std(errors_flat) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors_flat)[0, 1]
        print(f"{name}: {corr:.6f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    if final_metric < METRIC_THRESHOLD:
        print(f"\nMetric {final_metric} < {METRIC_THRESHOLD}. Generating submission...")
        generate_predictions(
            model_path=MODEL_PATH,
            batch_size=BATCH_SIZE,
            submission_output_dir=SUBMISSION_DIR,
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric {final_metric} >= {METRIC_THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
