import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import SF_DCN
from library.train import train_one_epoch, validate, generate_submission


def perform_failure_analysis(model, val_loader, config, device):
    """
    Analyzes model performance on the validation set by correlating error
    with metadata features.
    """
    print("Performing failure analysis...")
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    # Inference on validation set
    with torch.no_grad():
        for inputs, partner_indices, targets, ids in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Get refined prediction (y2)
            _, y2 = model(inputs, partner_indices)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(ids)

    # Concatenate results
    preds = np.concatenate(all_preds, axis=0)  # (N, L, 5)
    targets = np.concatenate(all_targets, axis=0)  # (N, L, 5)

    # Slice to scored positions and columns
    # Predictions/Targets are (N, 107, 5), we need first 68 positions and scored columns
    preds_scored = preds[:, : config.seq_scored, config.scored_indices]
    targets_scored = targets[:, : config.seq_scored, config.scored_indices]

    # Calculate RMSE per sample (averaged over positions and scored columns)
    # Shape: (N, 68, 3) -> Mean squared error per sample -> Sqrt
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata
    val_df = pd.read_csv(config.val_csv)

    # Create a dataframe for errors
    error_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = val_df.merge(error_df, on="id")

    # Define features to analyze
    features = ["signal_to_noise", "mean_reactivity"]

    # Add sequence composition features
    if "sequence" in analysis_df.columns:
        analysis_df["count_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
        analysis_df["count_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
        analysis_df["count_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
        analysis_df["count_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))
        features.extend(["count_A", "count_G", "count_C", "count_U"])

    if "structure" in analysis_df.columns:
        analysis_df["paired_bases"] = analysis_df["structure"].apply(
            lambda x: x.count("(")
        )
        features.append("paired_bases")

    # Calculate and print correlations
    print(f"{'Feature':<20} | {'Correlation with Error':<20}")
    print("-" * 45)
    for feat in features:
        if feat in analysis_df.columns:
            # Drop NaNs for correlation calculation
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"{feat:<20} | {corr:.4f}")


def main():
    # 1. Configuration
    # Set epochs to 15 for a fast baseline execution
    config = Config(epochs=15)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    model = SF_DCN(config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 5. Training Loop
    best_metric = float("inf")
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    print(f"Starting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        val_metric = validate(model, val_loader, config, device)

        scheduler.step(val_metric)

        # Save best model
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            # print(f"Epoch {epoch+1}: New best model (MCRMSE: {best_metric:.6f})")
        # else:
        # print(f"Epoch {epoch+1}: Train Loss {train_loss:.6f}, Val MCRMSE {val_metric:.6f}")

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_val_metric = validate(model, val_loader, config, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, config, device)

    # 8. Submission Logic
    threshold = 0.5403054356575012
    if final_val_metric < threshold:
        print(
            f"Metric ({final_val_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, test_loader, config, device)
    else:
        print(
            f"Metric ({final_val_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
