import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_device, mcrmse_loss
from library.data import get_dataloaders
from library.model import DR_RHN
from library.train import train_epoch, validate


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating errors with metadata features.
    """
    model.eval()
    all_losses = []
    all_ids = []

    # scored_indices = [0, 1, 3] -> reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = Config.SCORED_TARGET_INDICES

    print("\nRunning Failure Analysis...")

    with torch.no_grad():
        for x, y, mask, p_idx, p_mask, sample_ids in val_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            p_idx = p_idx.to(device)
            p_mask = p_mask.to(device)

            # Forward pass (get refined prediction y2)
            _, y2 = model(x, p_idx, p_mask)

            # Calculate MCRMSE per sample manually
            # y2: (B, L, 5)
            pred_scored = y2[:, :, scored_indices]
            target_scored = y[:, :, scored_indices]

            # Squared Error: (B, L, 3)
            squared_diff = (pred_scored - target_scored) ** 2

            # Apply mask: (B, L) -> (B, L, 1)
            mask_expanded = mask.unsqueeze(-1)
            squared_diff = squared_diff * mask_expanded

            # Mean over Sequence (dim 1) for valid positions
            # Sum valid positions per sequence
            valid_counts = mask.sum(dim=1).clamp(min=1)  # (B,)

            # MSE per column per sample: (B, 3)
            mse_per_col = squared_diff.sum(dim=1) / valid_counts.unsqueeze(-1)

            # RMSE per column per sample: (B, 3)
            rmse_per_col = torch.sqrt(mse_per_col)

            # MCRMSE per sample: Mean over columns (dim 1) -> (B,)
            mcrmse_per_sample = rmse_per_col.mean(dim=1)

            all_losses.extend(mcrmse_per_sample.cpu().numpy())
            all_ids.extend(sample_ids)

    # Create DataFrame of errors
    error_df = pd.DataFrame({"id": all_ids, "error": all_losses})

    # Load metadata
    val_meta_path = Config.VAL_CSV
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping correlation analysis.")
        return

    val_meta = pd.read_csv(val_meta_path)

    # Merge
    analysis_df = pd.merge(error_df, val_meta, on="id", how="left")

    # Calculate correlations
    features_to_check = ["signal_to_noise", "mean_reactivity", "seq_length"]
    print("-" * 50)
    print(f"{'Feature':<20} | {'Correlation with Error':<20}")
    print("-" * 50)

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"{feat:<20} | {corr:.4f}")
    print("-" * 50)


def generate_submission(model, test_loader, device):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")
    model.eval()

    ids_list = []
    preds_list = []

    # Target columns in order of model output
    # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Indices: 0, 1, 2, 3, 4

    with torch.no_grad():
        for x, _, _, p_idx, p_mask, sample_ids in test_loader:
            x = x.to(device)
            p_idx = p_idx.to(device)
            p_mask = p_mask.to(device)

            # Inference
            _, y2 = model(x, p_idx, p_mask)

            # y2 shape: (B, 107, 5)
            preds = y2.cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                # For each sample, we have 107 positions
                sample_preds = preds[i]  # (107, 5)

                for seq_pos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{seq_pos}"
                    ids_list.append(row_id)
                    preds_list.append(sample_preds[seq_pos])

    # Create DataFrame
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    submission_df = pd.DataFrame(preds_list, columns=cols)
    submission_df.insert(0, "id_seqpos", ids_list)

    # Save
    os.makedirs("./submission", exist_ok=True)
    save_path = "./submission/submission.csv"
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # Override Config for Fast Baseline
    Config.EPOCHS = 15
    print(f"Training for {Config.EPOCHS} epochs.")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = DR_RHN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 4. Training Loop
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device
        )  # Calling with 4 args as per definition

        # Validate
        val_loss = validate(model, val_loader, device)

        # Scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print("Training finished.")

    # 5. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))

    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.47142532743789534
    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
