import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
import sys

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.dataset import get_dataloaders
from library.model import HFSI_BiLSTM
from library.utils import seed_everything, get_device, WeightedL1Loss, compute_metric


def run():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    config = Config()

    # Use full dataset and extended training
    config.DEBUG = False

    # Ensure output directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(config)

    # Determine input dimension dynamically
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["input"].shape[-1]
    print(f"Input feature dimension: {input_dim}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = HFSI_BiLSTM(config, input_dim).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_MAX, eta_min=1e-6
    )

    # Loss Function
    criterion = WeightedL1Loss(
        w_insp=config.LOSS_WEIGHT_INSPIRATORY, w_exp=config.LOSS_WEIGHT_EXPIRATORY
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_val_mae = float("inf")
    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            u_out = batch["u_out"].to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(inputs)
            preds = preds.squeeze(-1)  # (B, L)

            loss = criterion(preds, targets, u_out)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds_list = []
        val_targets_list = []
        val_u_out_list = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                u_out = batch["u_out"].to(device)

                preds = model(inputs).squeeze(-1)

                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())
                val_u_out_list.append(u_out.cpu().numpy())

        val_preds_arr = np.concatenate(val_preds_list)
        val_targets_arr = np.concatenate(val_targets_list)
        val_u_out_arr = np.concatenate(val_u_out_list)

        val_mae = compute_metric(val_preds_arr, val_targets_arr, val_u_out_arr)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.6f} | Val MAE: {val_mae:.8f}"
        )

        # Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), config.MODEL_CHECKPOINT)

    # ==========================================
    # 5. Final Reporting
    # ==========================================
    print(f"Final Validation Metric: {best_val_mae}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # Retrieve feature names from cached parquet file
    try:
        df_sample = pd.read_parquet(config.TRAIN_CACHE)
        exclude_cols = ["id", "breath_id", "pressure"]
        feature_cols = [c for c in df_sample.columns if c not in exclude_cols]
        if "u_out" not in feature_cols:
            feature_cols.append("u_out")
        feature_cols = sorted(feature_cols)
    except Exception as e:
        print(f"Warning: Could not retrieve feature names ({e}). Using indices.")
        feature_cols = [f"feat_{i}" for i in range(input_dim)]

    # Run inference on validation set to get errors
    val_inputs_list = []
    val_preds_list = []
    val_targets_list = []
    val_u_out_list = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            u_out = batch["u_out"].to(device)
            preds = model(inputs).squeeze(-1)

            val_inputs_list.append(inputs.cpu().numpy())
            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())
            val_u_out_list.append(u_out.cpu().numpy())

    val_inputs_arr = np.concatenate(val_inputs_list)  # (N, 80, F)
    val_preds_arr = np.concatenate(val_preds_list)  # (N, 80)
    val_targets_arr = np.concatenate(val_targets_list)  # (N, 80)
    val_u_out_arr = np.concatenate(val_u_out_list)  # (N, 80)

    # Flatten for correlation analysis
    flat_inputs = val_inputs_arr.reshape(-1, input_dim)
    flat_preds = val_preds_arr.flatten()
    flat_targets = val_targets_arr.flatten()
    flat_u_out = val_u_out_arr.flatten()

    # Analyze only Inspiratory Phase (where metric is calculated)
    insp_mask = flat_u_out == 0

    if np.sum(insp_mask) > 0:
        insp_inputs = flat_inputs[insp_mask]
        insp_preds = flat_preds[insp_mask]
        insp_targets = flat_targets[insp_mask]

        # Calculate Absolute Error
        errors = np.abs(insp_preds - insp_targets)

        print(
            "Correlation between Absolute Error and Input Features (Inspiratory Phase):"
        )
        correlations = {}
        for i, col_name in enumerate(feature_cols):
            if i < insp_inputs.shape[1]:
                feat_vals = insp_inputs[:, i]
                # Avoid correlation with constant features
                if np.std(feat_vals) > 1e-9:
                    corr = np.corrcoef(errors, feat_vals)[0, 1]
                    correlations[col_name] = corr
                else:
                    correlations[col_name] = 0.0

        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for name, val in sorted_corr[:10]:
            print(f"{name}: {val:.4f}")
    else:
        print("No inspiratory phase data available for failure analysis.")

    # ==========================================
    # 7. Conditional Submission
    # ==========================================
    THRESHOLD = 0.1619843989610672

    if best_val_mae < THRESHOLD:
        print(
            f"\nValidation MAE ({best_val_mae}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds_list = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(device)
                preds = model(inputs).squeeze(-1)
                test_preds_list.append(preds.cpu().numpy())

        test_preds_flat = np.concatenate(test_preds_list).flatten()
        test_ids_flat = test_ids.flatten()

        submission_df = pd.DataFrame({"id": test_ids_flat, "pressure": test_preds_flat})
        submission_df.sort_values(by="id", inplace=True)

        print(f"Saving submission to {config.SUBMISSION_PATH}...")
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    else:
        print(
            f"\nValidation MAE ({best_val_mae}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
