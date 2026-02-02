import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

# Import components from the provided library
from library.config import Config
from library.dataset import get_dataloader
from library.model import AMSP_DS_Net
from library.train import train_one_epoch, evaluate, predict_and_submit, set_seed


def main():
    # 1. Setup and Configuration
    Config.setup()
    set_seed(Config.SEED)

    # Override Config for fast baseline execution
    Config.EPOCHS = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading DataLoaders...")
    # get_dataloader handles feature extraction and caching automatically
    train_loader, scaler = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, scaler=scaler
    )

    # 3. Model Initialization
    model = AMSP_DS_Net().to(device)

    # 4. Optimization Setup
    # Using MSELoss because targets are already log1p transformed in the Dataset
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pt")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["val_loss"]

        # Adjust Learning Rate
        scheduler.step(val_loss)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"RMSLE: {val_metrics['rmsle_mean']:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Final Validation Assessment
    print("\nPerforming Final Validation Assessment...")
    # Load best model weights
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Collect predictions and features for analysis
    all_preds = []
    all_targets = []
    all_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, global_features, batch_indices)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_global_feats.append(global_features.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_global_feats = np.vstack(all_global_feats)

    # Compute Final Metric (Mean Column-wise RMSLE)
    # Targets are log1p, so RMSE on them is RMSLE on original scale
    mse_per_col = np.mean((all_preds - all_targets) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute errors (on log scale)
    errors = np.abs(all_preds - all_targets)
    mean_errors = np.mean(errors, axis=1)  # Mean error per sample

    # Feature names corresponding to GlobalStream inputs
    global_feat_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "alpha",
        "beta",
        "gamma",
        "ratio_ab",
        "ratio_bc",
        "ratio_ca",
        "frac_Al",
        "frac_Ga",
        "frac_In",
        "frac_O",
        "w_mass",
        "w_rad",
        "w_en",
        "vol",
        "density",
        "n_atoms",
    ]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(all_global_feats, columns=global_feat_names)
    analysis_df["mean_error"] = mean_errors

    # Compute correlation
    correlations = analysis_df.corr()["mean_error"].drop("mean_error")
    print("Correlation between Global Features and Mean Error:")
    print(correlations.sort_values(key=abs, ascending=False))

    # 8. Submission Generation
    THRESHOLD = 0.05366557091474533
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        predict_and_submit(model, scaler, device)
    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
