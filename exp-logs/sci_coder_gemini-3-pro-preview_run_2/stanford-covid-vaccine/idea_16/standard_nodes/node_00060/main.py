import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_global_mcrmse
from library.loss import MCRMSELoss
from library.model import ScaleAlignedDenseNet
from library.data import get_loaders


def train_model(device):
    """
    Trains the ScaleAlignedDenseNet model.
    """
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    print("Initializing Model...")
    model = ScaleAlignedDenseNet().to(device)

    criterion = MCRMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    best_val_metric = float("inf")
    best_model_path = Config.BEST_MODEL_PATH

    # Ensure working directory for model exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (inputs, partner_indices, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(inputs, partner_indices)

            # Loss calculation
            loss = criterion(preds, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for inputs, partner_indices, targets in val_loader:
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)
                targets = targets.to(device)

                preds = model(inputs, partner_indices)

                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)

        # Compute Metric (MCRMSE on scored columns)
        val_metric = compute_global_mcrmse(
            val_preds, val_targets, scored_indices=Config.SCORED_TARGET_INDICES
        )

        # Scheduler Step
        scheduler.step(val_metric)

        # Checkpoint
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            # print(f"Epoch {epoch+1}: New Best Val Metric: {best_val_metric:.6f}")
        # else:
        # print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Metric: {val_metric:.6f}")

    print("Training complete.")
    return best_val_metric, val_loader, test_loader


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Collect predictions and ids
    # Note: We need to iterate the dataset directly or loader to get IDs if loader doesn't yield them.
    # The provided RNADataset yields (x, p_idx, y), but the loader doesn't yield IDs.
    # However, the RNADataset stores .ids. We can access them via the loader's dataset.
    # To align correctly, we must ensure the loader is not shuffling if we access dataset.ids by index,
    # or we modify the loop to track indices.
    # The val_loader in library/data.py is defined with shuffle=False.

    val_ids = val_loader.dataset.ids

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            preds = model(inputs, partner_indices)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Calculate RMSE per sample (averaged over scored columns and sequence length)
    # Scored indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = Config.SCORED_TARGET_INDICES

    preds_scored = preds_arr[..., scored_indices]
    targets_scored = targets_arr[..., scored_indices]

    # MSE per sample: mean over sequence(1) and channels(2)
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"id": val_ids, "rmse": rmse_per_sample})

    # Load Metadata
    if os.path.exists(Config.VAL_CSV):
        meta_df = pd.read_csv(Config.VAL_CSV)
        # Merge on ID
        merged_df = pd.merge(analysis_df, meta_df, on="id", how="left")

        # Calculate correlations
        # Features of interest: signal_to_noise, mean_reactivity, seq_length (constant 107), SN_filter
        features = ["signal_to_noise", "mean_reactivity", "SN_filter"]

        print("Correlation between Error (RMSE) and Metadata Features:")
        for feat in features:
            if feat in merged_df.columns:
                # Drop NaNs if any
                valid_data = merged_df[["rmse", feat]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data["rmse"], valid_data[feat])
                    print(f"  {feat}: {corr:.4f}")
    else:
        print("Validation metadata not found, skipping correlation analysis.")

    # Recalculate global metric to print required output
    final_metric = compute_global_mcrmse(preds_arr, targets_arr, scored_indices)
    print(f"Final Validation Metric: {final_metric}")

    return final_metric


def generate_submission(model, test_loader, device):
    """
    Generates submission file for the test set.
    """
    print("\nGenerating Submission...")
    model.eval()

    all_preds = []
    test_ids = test_loader.dataset.ids

    with torch.no_grad():
        for inputs, partner_indices, _ in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            preds = model(inputs, partner_indices)
            all_preds.append(preds.cpu().numpy())

    # Shape: (Num_Samples, Seq_Len, Num_Targets)
    preds_arr = np.concatenate(all_preds, axis=0)

    # Flatten for submission
    # Format requires one row per sequence position
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Target order in model output matches Config.TARGET_COLS:
    # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    submission_data = []
    target_cols = Config.TARGET_COLS

    num_samples, seq_len, num_targets = preds_arr.shape

    for i in range(num_samples):
        sample_id = test_ids[i]
        sample_preds = preds_arr[i]  # (Seq_Len, 5)

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"
            row_values = sample_preds[pos].tolist()

            row_dict = {"id_seqpos": row_id}
            for col_name, val in zip(target_cols, row_values):
                row_dict[col_name] = val

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Ensure submission directory exists
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "submission.csv")

    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def main():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train
    _, val_loader, test_loader = train_model(device)

    # Load best model for evaluation
    model = ScaleAlignedDenseNet().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Failure Analysis & Final Metric
    final_metric = perform_failure_analysis(model, val_loader, device)

    # Submission Threshold
    THRESHOLD = 0.5421870350837708

    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
