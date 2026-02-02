import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Add current directory to path to ensure imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_mcrmse
from library.data import get_dataloaders
from library.model import RNAGRUModel
from library.train import train_epoch, validate, generate_submission


def main():
    # 1. Configuration
    # We use 50 epochs to ensure convergence as per Lesson 00031.
    # Cite solution_lesson_node_00031
    config = Config(epochs=50, batch_size=32)

    # Ensure reproducibility
    set_seed(config.seed)

    print("Initializing Configuration...")
    print(config.get_config_info())

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    print(f"Initializing model on {config.device}...")
    model = RNAGRUModel(config).to(config.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min
    )

    criterion = MCRMSELoss()

    # 4. Training Loop
    best_score = float("inf")

    print("Starting training...")

    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, config.device
        )

        # Validate (returns MCRMSE on scored columns)
        val_score = validate(model, val_loader, config.device, config)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f} | LR: {current_lr:.2e}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  New best model saved! Score: {best_score:.6f}")

    print(f"Training complete. Best Validation Score: {best_score:.6f}")

    # 5. Final Validation & Failure Analysis
    print("\nPerforming Final Evaluation and Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()

    # Generate predictions on Validation set for analysis
    all_preds = []
    all_targets = []

    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(config.device)
            preds = model(features)
            # Slice to prediction length (68)
            preds_sliced = preds[:, : config.pred_len, :]

            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (N, 68, 5)
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (N, 68, 5)

    # Filter to scored columns for metric calculation
    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Calculate Final Metric
    final_metric = compute_mcrmse(preds_scored, targets_scored)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # --- Failure Analysis ---
    # Calculate error magnitude per sample (Mean of RMSEs across scored columns)
    # Squared error: (N, 68, 3)
    sq_error = (preds_scored - targets_scored) ** 2
    # MSE per sample per column: Mean over sequence length (axis 1) -> (N, 3)
    mse_per_sample_col = np.mean(sq_error, axis=1)
    # RMSE per sample per column: (N, 3)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    # Mean RMSE per sample (Error Magnitude): Mean over columns (axis 1) -> (N,)
    sample_errors = np.mean(rmse_per_sample_col, axis=1)

    # Load Metadata to correlate with features
    val_df = pd.read_parquet(config.val_metadata_path)

    # Ensure alignment (val_loader is not shuffled, so order should match)
    if len(val_df) != len(sample_errors):
        print(
            f"Warning: Validation dataframe length ({len(val_df)}) mismatch with predictions ({len(sample_errors)}). Truncating to match."
        )
        val_df = val_df.iloc[: len(sample_errors)]

    val_df["error_magnitude"] = sample_errors

    # Feature Engineering for Correlation
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))
    val_df["pct_unpaired"] = val_df["structure"].apply(lambda s: s.count(".") / len(s))

    corr_cols = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_U",
        "pct_C",
        "pct_unpaired",
    ]
    valid_corr_cols = [c for c in corr_cols if c in val_df.columns]

    print("\nCorrelation between Error Magnitude and Input Features:")
    correlations = val_df[valid_corr_cols].corrwith(val_df["error_magnitude"])
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Update submission path to meet requirement
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        config.submission_path = os.path.join(submission_dir, "submission.csv")

        generate_submission(model, test_loader, config)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
