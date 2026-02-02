import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import time

# Import from library
from library.config import Config
from library.utils import set_seed, TargetScaler
from library.data import get_dataloaders
from library.model import SS_CGCNN
from library.train import train_one_epoch, evaluate


def calculate_rmsle(y_true, y_pred):
    """
    Calculates Column-wise Root Mean Squared Logarithmic Error.
    """
    # Ensure non-negative for log
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)

    squared_error = (log_pred - log_true) ** 2
    mean_squared_error = np.mean(squared_error, axis=0)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return np.mean(root_mean_squared_error)


def perform_failure_analysis(val_ids, val_preds, val_targets, metadata_path):
    print("\nPerforming Failure Analysis...")

    # Load metadata
    try:
        meta_df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Could not load metadata for failure analysis: {e}")
        return

    # Filter metadata to match validation set IDs
    val_meta = meta_df[meta_df["id"].isin(val_ids)].copy()

    # Create a mapping from ID to error
    # Calculate Mean Absolute Error per sample (averaged over targets)
    abs_errors = np.abs(val_preds - val_targets)
    mean_abs_error = np.mean(abs_errors, axis=1)

    error_df = pd.DataFrame(
        {
            "id": val_ids,
            "error": mean_abs_error,
            "formation_error": abs_errors[:, 0],
            "bandgap_error": abs_errors[:, 1],
        }
    )

    # Merge errors with metadata
    analysis_df = pd.merge(val_meta, error_df, on="id")

    # Select numerical columns for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    # Exclude ID and targets from correlation check
    exclude = [
        "id",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
        "error",
        "formation_error",
        "bandgap_error",
    ]
    features = [c for c in numeric_cols if c not in exclude]

    print("Correlation between Mean Absolute Error and Features:")
    correlations = {}
    for feat in features:
        if analysis_df[feat].std() > 0:  # Avoid constant columns
            corr = analysis_df["error"].corr(analysis_df[feat])
            correlations[feat] = corr

    # Sort and print top correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat:<30}: {corr:.4f}")


def main():
    # Set up environment
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Prepare Data
    # load_cached_data=True to use pre-processed npz files if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Initialize Model
    model = SS_CGCNN(config=Config).to(device)

    # 3. Setup Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    # We use a limited number of epochs for the baseline to ensure it finishes quickly
    # The dataset is small (~2k), so 60 epochs is actually quite fast.
    num_epochs = 60
    print(f"Training for {num_epochs} epochs...")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss {train_loss:.5f}, Val Loss {val_loss:.5f}"
            )

    print("Training complete.")

    # 5. Validation Assessment
    print("Loading best model for validation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Load Scaler
    scaler = TargetScaler()
    if os.path.exists(Config.TARGET_SCALER_CACHE):
        scaler.load(Config.TARGET_SCALER_CACHE)
    else:
        print("Error: Target scaler not found.")
        return

    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)

            # Inverse transform
            pred_inv = scaler.inverse_transform(out)
            target_inv = scaler.inverse_transform(batch.y)

            val_preds.append(pred_inv.cpu().numpy())
            val_targets.append(target_inv.cpu().numpy())
            val_ids.extend(batch.id.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Metric
    final_metric = calculate_rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(val_ids, val_preds, val_targets, Config.VAL_METADATA_PATH)

    # 7. Generate Submission
    THRESHOLD = 0.049412816762924194
    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                # Inverse transform
                pred_inv = scaler.inverse_transform(out)

                test_preds.append(pred_inv.cpu().numpy())
                test_ids.extend(batch.id.cpu().numpy())

        if test_preds:
            test_preds = np.concatenate(test_preds, axis=0)

            submission_df = pd.DataFrame(
                {
                    "id": test_ids,
                    "formation_energy_ev_natom": test_preds[:, 0],
                    "bandgap_energy_ev": test_preds[:, 1],
                }
            )

            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric ({final_metric}) is NOT better than threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
