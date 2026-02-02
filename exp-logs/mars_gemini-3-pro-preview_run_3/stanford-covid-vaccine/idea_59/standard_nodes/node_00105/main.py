import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.data import get_dataloaders
from library.model import HC_DBR_BiGRU
from library.train import train_one_epoch, validate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify error correlations.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    # 1. Collect predictions and targets per sample
    all_ids = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["target"].to(device)
            ids = batch["id"]

            outputs = model(features, pair_indices, pair_mask)

            all_ids.extend(ids)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # 2. Calculate MCRMSE per sample
    # Slice to scored length and scored columns
    pred_sliced = global_preds[:, : Config.PRED_LEN, Config.SCORED_TARGET_INDICES]
    target_sliced = global_targets[:, : Config.PRED_LEN, Config.SCORED_TARGET_INDICES]

    # MSE: (N, 68, 3) -> Mean over (68, 3) -> (N,)
    # Note: MCRMSE definition is Mean(Sqrt(Mean(Error^2))).
    # For per-sample analysis, we approximate "Error Magnitude" as the mean RMSE across columns.
    mse = (pred_sliced - target_sliced) ** 2
    rmse_per_col = torch.sqrt(torch.mean(mse, dim=1))  # (N, 3)
    sample_errors = torch.mean(rmse_per_col, dim=1).numpy()  # (N,)

    # 3. Load Metadata
    val_meta_path = "./metadata/val.parquet"
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping analysis.")
        return

    val_df = pd.read_parquet(val_meta_path)

    # Create Error DataFrame
    error_df = pd.DataFrame({"id": all_ids, "model_error": sample_errors})

    # Merge with metadata
    analysis_df = pd.merge(error_df, val_df, on="id", how="inner")

    # 4. Compute Correlations
    # Select numerical columns of interest
    cols_to_check = ["signal_to_noise", "SN_filter", "model_error"]

    # Add mean reactivity error if available
    if "reactivity_error" in analysis_df.columns:
        # reactivity_error is a list, take mean
        analysis_df["mean_reactivity_error"] = analysis_df["reactivity_error"].apply(
            lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        cols_to_check.append("mean_reactivity_error")

    corr_matrix = analysis_df[cols_to_check].corr()
    error_corrs = (
        corr_matrix["model_error"].drop("model_error").sort_values(ascending=False)
    )

    print("Correlation between Model Error and Features:")
    print(error_corrs)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Ensure submission dir exists
    os.makedirs("./submission", exist_ok=True)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = HC_DBR_BiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_metric = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metric = validate(model, val_loader, device)

        # Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_metric:.5f}"
        )

        # Save Best
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        submission_path = "./submission/submission.csv"
        generate_submission(model, test_loader, device, submission_path)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
