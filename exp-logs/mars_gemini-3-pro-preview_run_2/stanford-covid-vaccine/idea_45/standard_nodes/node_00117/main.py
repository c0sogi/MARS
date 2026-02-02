import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.stats as stats

# Import provided library functions
from library.loss_metric import MCRMSELoss
from library.data_processor import get_dataloaders
from library.model_architecture import RDFRN
from library.trainer import train_one_epoch, validate, generate_submission, set_seed

# Configuration
SEED = 42
EPOCHS = 15
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBMISSION_THRESHOLD = 0.47142532743789534
BEST_MODEL_PATH = "./working/best_model.pth"


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify error correlations.
    """
    print("\n==== Starting Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Collect predictions and targets
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Get refined predictions (y2)
            _, y2 = model(inputs, partner_indices)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE per sample
    # Scored columns: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    # Scored length: 68
    scored_cols = [0, 1, 3]
    seq_scored = 68

    p_scored = all_preds[:, :seq_scored, scored_cols]
    t_scored = all_targets[:, :seq_scored, scored_cols]

    # MSE per sample: mean over length (68) and channels (3)
    mse_per_sample = np.mean((p_scored - t_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create Error DataFrame
    error_df = pd.DataFrame({"id": all_ids, "error_mcrmse": rmse_per_sample})

    # Load Metadata
    val_meta_path = "./metadata/val.csv"
    if not os.path.exists(val_meta_path):
        print(
            f"Warning: Validation metadata not found at {val_meta_path}. Skipping correlation analysis."
        )
        return

    val_meta = pd.read_csv(val_meta_path)

    # Merge
    merged_df = pd.merge(error_df, val_meta, on="id", how="inner")

    # Define features to correlate
    features = ["signal_to_noise", "mean_reactivity"]

    # Add simple sequence features
    if "sequence" in merged_df.columns:
        merged_df["count_A"] = merged_df["sequence"].apply(lambda x: x.count("A"))
        merged_df["count_G"] = merged_df["sequence"].apply(lambda x: x.count("G"))
        merged_df["count_C"] = merged_df["sequence"].apply(lambda x: x.count("C"))
        merged_df["count_U"] = merged_df["sequence"].apply(lambda x: x.count("U"))
        features.extend(["count_A", "count_G", "count_C", "count_U"])

    print(f"Correlations with Error (MCRMSE) on {len(merged_df)} validation samples:")
    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 50)

    for feat in features:
        if feat in merged_df.columns:
            # Drop NaNs if any
            valid_data = merged_df[[feat, "error_mcrmse"]].dropna()
            if len(valid_data) > 1:
                corr, p_val = stats.pearsonr(
                    valid_data[feat], valid_data["error_mcrmse"]
                )
                print(f"{feat:<20} | {corr:.4f}       | {p_val:.4e}")
            else:
                print(f"{feat:<20} | N/A (Insufficient Data)")
    print("===================================\n")


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs("./working", exist_ok=True)

    print(f"Initializing RDF-RN pipeline on {DEVICE}")

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True
    )

    # 3. Model Initialization
    model = RDFRN().to(DEVICE)
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0
    early_stopping_patience = 6

    print(f"Starting training for {EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_metrics = validate(model, val_loader, DEVICE)
        val_mcrmse = val_metrics["mcrmse"]

        duration = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Time: {duration:.1f}s | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

        # Scheduler Step
        scheduler.step(val_mcrmse)

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"Training completed in {time.time() - start_time:.1f}s")

    # 5. Final Validation & Metric
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    final_metrics = validate(model, val_loader, DEVICE)
    final_score = final_metrics["mcrmse"]

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, DEVICE)

    # 7. Submission
    if final_score < SUBMISSION_THRESHOLD:
        print(
            f"Validation score ({final_score:.5f}) meets threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model, test_loader, DEVICE, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"Validation score ({final_score:.5f}) did not meet threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
