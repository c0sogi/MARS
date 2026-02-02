import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import NonLinearChannelGatedBiGRU
from library.train import train_epoch, validate, generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override defaults for a fast baseline execution
    EPOCHS = 10
    BATCH_SIZE = 32

    # Initialize Config
    cfg = Config(epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Set reproducibility
    set_seed(cfg.SEED)

    # Device detection
    device = torch.device(cfg.DEVICE)
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True, batch_size=cfg.BATCH_SIZE
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = NonLinearChannelGatedBiGRU().to(device)

    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("Starting training...")
    best_mcrmse = float("inf")

    for epoch in range(cfg.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)

    # Print required metric format
    print(f"Final Validation Metric: {best_mcrmse}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nStarting Failure Analysis...")

    # Load best model
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect predictions and targets per sample
    val_ids = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(inputs, pair_indices)

            # Slice to scored length
            outputs_scored = outputs[:, : cfg.PRED_LEN, :]

            val_preds.append(outputs_scored.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_ids.extend(ids)

    # Concatenate
    val_preds = np.concatenate(val_preds, axis=0)  # (N, 68, 5)
    val_targets = np.concatenate(val_targets, axis=0)  # (N, 68, 5)

    # Calculate error per sample (Mean of RMSEs of scored columns)
    # Scored columns: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = [0, 1, 3]

    # Squared Error: (N, 68, 5)
    squared_error = (val_preds - val_targets) ** 2

    # MSE per column per sample: (N, 5) -> Mean over seq_len (axis 1)
    mse_per_sample = np.mean(squared_error, axis=1)

    # RMSE per column per sample: (N, 5)
    rmse_per_sample = np.sqrt(mse_per_sample)

    # MCRMSE per sample (average over scored columns): (N,)
    sample_mcrmse = np.mean(rmse_per_sample[:, scored_indices], axis=1)

    # Load Validation Metadata
    df_val = pd.read_parquet(cfg.VAL_PATH)

    # Align data
    df_val = df_val.set_index("id")
    # Select and reorder based on val_ids from DataLoader
    df_val = df_val.loc[val_ids]

    # Add error to dataframe
    df_val["model_error"] = sample_mcrmse

    # Feature Engineering for Correlation
    df_val["pct_A"] = df_val["sequence"].apply(lambda x: x.count("A") / len(x))
    df_val["pct_G"] = df_val["sequence"].apply(lambda x: x.count("G") / len(x))
    df_val["pct_C"] = df_val["sequence"].apply(lambda x: x.count("C") / len(x))
    df_val["pct_U"] = df_val["sequence"].apply(lambda x: x.count("U") / len(x))

    # Calculate Correlations
    corr_cols = [
        "model_error",
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
    ]
    # Ensure columns are numeric
    for c in corr_cols:
        df_val[c] = pd.to_numeric(df_val[c], errors="coerce")

    correlations = df_val[corr_cols].corr()["model_error"].sort_values(ascending=False)

    print("\nCorrelation of features with Model Error:")
    print(correlations.drop("model_error"))

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.5978901386

    if best_mcrmse < THRESHOLD:
        print(f"\nMetric {best_mcrmse} < {THRESHOLD}. Generating submission...")
        generate_submission(model, test_loader, device, cfg.SUBMISSION_PATH)
    else:
        print(f"\nMetric {best_mcrmse} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
