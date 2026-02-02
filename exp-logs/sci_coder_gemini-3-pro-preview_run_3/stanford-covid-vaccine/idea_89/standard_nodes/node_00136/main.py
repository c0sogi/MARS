import os
import torch
import pandas as pd
import numpy as np
import time
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_metric, get_scored_indices
from library.data import get_dataloaders
from library.model import RNARegressor
from library.train import train_one_epoch, validate


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating sample-wise errors with metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            adj_indices = batch["adjacency_indices"].to(device)
            adj_mask = batch["adjacency_mask"].to(device)
            targets = batch["targets"]  # CPU
            ids = batch["id"]

            outputs = model(inputs, adj_indices, adj_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # 2. Calculate RMSE per sample (averaged over scored columns and positions)
    # Slice to scored positions
    y_pred_scored = y_pred[:, : Config.SEQ_SCORED, :]
    y_true_scored = y_true[:, : Config.SEQ_SCORED, :]

    # Get scored target indices
    scored_indices = get_scored_indices()

    # Squared Error: (N, 68, 5)
    squared_error = (y_pred_scored - y_true_scored) ** 2

    # Mean over positions (dim 1) -> (N, 5)
    mse_per_sample_col = torch.mean(squared_error, dim=1)

    # Filter for scored columns
    mse_per_sample_scored = mse_per_sample_col[:, scored_indices]

    # Mean over columns -> (N,)
    mse_per_sample = torch.mean(mse_per_sample_scored, dim=1)
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata to correlate
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Ensure alignment by ID
    analysis_df = pd.DataFrame({"id": all_ids, "error_rmse": rmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id", how="left")

    # Calculate Features for correlation
    # Nucleotide content
    merged_df["pct_A"] = merged_df["sequence"].apply(lambda s: s.count("A") / len(s))
    merged_df["pct_U"] = merged_df["sequence"].apply(lambda s: s.count("U") / len(s))
    merged_df["pct_G"] = merged_df["sequence"].apply(lambda s: s.count("G") / len(s))
    merged_df["pct_C"] = merged_df["sequence"].apply(lambda s: s.count("C") / len(s))

    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_U",
        "pct_G",
        "pct_C",
    ]

    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 50)

    for feat in features_to_check:
        if feat in merged_df.columns:
            # Drop NaNs just in case
            valid_data = merged_df[[feat, "error_rmse"]].dropna()
            if len(valid_data) > 1:
                corr, p_val = pearsonr(valid_data[feat], valid_data["error_rmse"])
                print(f"{feat:<20} | {corr:<12.4f} | {p_val:<12.4e}")


def generate_submission(model, test_loader, device):
    """
    Generates submission file for the test set.
    """
    print("\nGenerating submission...")
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            adj_indices = batch["adjacency_indices"].to(device)
            adj_mask = batch["adjacency_mask"].to(device)
            ids = batch["id"]

            outputs = model(inputs, adj_indices, adj_mask)
            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate: (N_samples, 107, 5)
    preds_array = np.concatenate(all_preds, axis=0)

    # Prepare data for DataFrame
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        sample_preds = preds_array[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()
            submission_rows.append([row_id] + row_values)

    submission_df = pd.DataFrame(submission_rows, columns=["id_seqpos"] + target_cols)

    # Save
    os.makedirs("./submission", exist_ok=True)
    save_path = "./submission/submission.csv"
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}. Shape: {submission_df.shape}")


def main():
    # 1. Configuration & Setup
    # Override epochs for fast baseline execution
    Config.EPOCHS = 15

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    model = RNARegressor().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f} | Time: {elapsed:.1f}s"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print("Training complete.")

    # 5. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute Final Metric
    final_metric = validate(model, val_loader, device)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, device)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
