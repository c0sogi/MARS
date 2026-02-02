import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import get_dataloaders
from library.model import LAWDS
from library.trainer import train_step, validate_step, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # Limit epochs to ensure execution finishes well within the 2-hour limit.
    # The dataset is small (1728 train samples), so convergence is relatively fast.
    Config.MAX_EPOCHS = 50
    Config.set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    # load_cached_data=True allows using precomputed features if available in the cache dir
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    model = LAWDS().to(device)

    # Optimizer and Scheduler setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cite debug_lesson_1: Remove Deprecated `verbose` Parameter from PyTorch Scheduler Initialization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Loss Function: MSE on log-transformed targets
    # This effectively optimizes for MSLE (Mean Squared Logarithmic Error)
    criterion = nn.MSELoss()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        val_loss = validate_step(model, val_loader, criterion, device)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Log progress periodically
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
            )

    # Load the best model state for final evaluation and inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val Loss: {best_val_loss:.5f}")

    # -------------------------------------------------------------------------
    # 5. Validation Assessment & Metric Calculation
    # -------------------------------------------------------------------------
    print("Performing validation assessment...")
    model.eval()

    all_preds = []
    all_targets = []
    all_global_feats = []

    with torch.no_grad():
        for atom_x, batch_indices, global_x, targets, _ in val_loader:
            atom_x = atom_x.to(device)
            batch_indices = batch_indices.to(device)
            global_x_gpu = global_x.to(device)

            outputs = model(atom_x, batch_indices, global_x_gpu)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_global_feats.append(global_x.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_global_feats = np.vstack(all_global_feats)

    # Calculate Metric: Column-wise Root Mean Squared Logarithmic Error
    # Note: The model predicts log(1+y), and targets are log(1+y).
    # Therefore, MSE on these values is MSLE.
    # RMSLE per column = sqrt(mean((log_pred - log_target)^2))
    mse_per_col = np.mean((all_targets - all_preds) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing failure analysis...")
    # Calculate error magnitude (Mean Absolute Error of logs) per sample
    errors = np.mean(np.abs(all_targets - all_preds), axis=1)

    # Feature names corresponding to GlobalEncoder inputs (from data_loader.py)
    feature_names = [
        "lattice_vector_1",
        "lattice_vector_2",
        "lattice_vector_3",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "atomic_density",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
        "stoich_O",
        "num_atoms",
    ]

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(all_global_feats, columns=feature_names)
    df_analysis["error"] = errors

    # Compute correlations between global features and error magnitude
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(key=abs, ascending=False)
    )
    print("Top 5 correlations with error magnitude:")
    print(correlations.head(5))

    # -------------------------------------------------------------------------
    # 7. Conditional Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
