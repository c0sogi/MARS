import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library components
from library.config import Config
from library.data_utils import load_or_process_data
from library.model_components import RNAModel
from library.loss_metric import MCRMSELoss, calculate_mcrmse
from library.train_eval import train_epoch, validate


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_gc_content(sequence):
    """Calculates GC content of a sequence."""
    if not sequence:
        return 0.0
    g = sequence.count("G")
    c = sequence.count("C")
    return (g + c) / len(sequence)


def run_failure_analysis(model, val_loader, config, device):
    """
    Performs failure analysis on the validation set.
    Correlates model error with metadata features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Get Predictions and Targets
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["sequence"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            targets = batch["targets"]
            ids = batch["id"]

            preds = model(inputs, bpp_indices, bpp_mask)

            all_preds.append(preds.cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    preds_tensor = torch.cat(all_preds, dim=0)  # (N, 107, 5)
    targets_tensor = torch.cat(all_targets, dim=0)  # (N, 68, 5)

    # 2. Calculate RMSE per sample (on scored columns only)
    # Slice preds to 68
    preds_sliced = preds_tensor[:, : config.pred_len, :]

    # Identify scored indices
    all_cols = config.target_cols
    scored_cols_set = set(config.scored_cols)
    scored_indices = [i for i, col in enumerate(all_cols) if col in scored_cols_set]

    # Filter
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_tensor[:, :, scored_indices]

    # MSE per sample: Mean over (Seq, Channels)
    mse_per_sample = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata for Correlation
    # We need to map IDs to metadata features.
    val_df = pd.read_parquet(config.val_data_path)

    # Ensure alignment
    # Create a map from ID to features
    id_to_sn = dict(zip(val_df["id"], val_df["signal_to_noise"]))
    id_to_filter = dict(zip(val_df["id"], val_df["SN_filter"]))
    id_to_seq = dict(zip(val_df["id"], val_df["sequence"]))

    # Extract aligned lists
    sn_values = []
    filter_values = []
    gc_values = []

    for sample_id in all_ids:
        sn_values.append(id_to_sn.get(sample_id, 0.0))
        filter_values.append(id_to_filter.get(sample_id, 0))
        seq = id_to_seq.get(sample_id, "")
        gc_values.append(calculate_gc_content(seq))

    sn_values = np.array(sn_values)
    filter_values = np.array(filter_values)
    gc_values = np.array(gc_values)

    # 4. Calculate Correlations
    # Handle NaNs if any (though data analysis showed none)
    valid_mask = ~np.isnan(rmse_per_sample)

    if np.sum(valid_mask) > 1:
        corr_sn, _ = pearsonr(rmse_per_sample[valid_mask], sn_values[valid_mask])
        corr_filter, _ = pearsonr(
            rmse_per_sample[valid_mask], filter_values[valid_mask]
        )
        corr_gc, _ = pearsonr(rmse_per_sample[valid_mask], gc_values[valid_mask])

        print(f"Correlation (RMSE vs Signal-to-Noise): {corr_sn:.4f}")
        print(f"Correlation (RMSE vs SN_filter):       {corr_filter:.4f}")
        print(f"Correlation (RMSE vs GC_Content):      {corr_gc:.4f}")

        # Interpretation
        if corr_sn < -0.1:
            print("Observation: Higher signal-to-noise is associated with lower error.")
        elif corr_sn > 0.1:
            print(
                "Observation: Higher signal-to-noise is associated with higher error (Unexpected)."
            )
    else:
        print("Insufficient data for correlation analysis.")


def generate_submission_file(model, config, device):
    """
    Generates submission.csv for the test set.
    """
    print("\nGenerating submission file...")

    # Load Test Data
    test_dataset = load_or_process_data("test", config, load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["sequence"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            ids = batch["id"]

            preds = model(inputs, bpp_indices, bpp_mask)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    all_preds_np = np.concatenate(all_preds, axis=0)  # (N, 107, 5)

    # Format for CSV
    submission_data = []
    target_cols = config.target_cols

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds_np[i]

        for seqpos in range(config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_vals[col_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


def main():
    # 1. Setup
    config = Config()

    # Fast Baseline Override
    config.epochs = 15
    print(f"Running fast baseline with {config.epochs} epochs.")

    set_seed(config.seed)
    device = torch.device(config.device)

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = load_or_process_data("train", config, load_cached_data=True)
    val_dataset = load_or_process_data("val", config, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing model on {device}...")
    model = RNAModel(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    criterion = MCRMSELoss()

    # 4. Training Loop
    best_mcrmse = float("inf")

    print("Starting training...")
    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, config, device
        )

        # Validate
        val_mcrmse = validate(model, val_loader, config, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), config.model_save_path)

    print(f"Training complete. Best MCRMSE: {best_mcrmse}")

    # 5. Final Evaluation & Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))

    # Calculate Final Metric on full set (redundant with loop but required for explicit printing/verification)
    final_val_metric = validate(model, val_loader, config, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, config, device)

    # 6. Submission
    THRESHOLD = 0.5978901386
    if final_val_metric < THRESHOLD:
        print(
            f"Validation metric ({final_val_metric}) meets threshold ({THRESHOLD}). Generating submission."
        )
        generate_submission_file(model, config, device)
    else:
        print(
            f"Validation metric ({final_val_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
