import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config, set_seed, device
from library.data import get_dataset, RNADataset
from library.model import SRDN
from library.loss import MCRMSELoss


def run():
    # 1. Setup
    set_seed(42)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load cached data if available, otherwise process from metadata
    train_ids, train_inputs, train_pmaps, train_targets = get_dataset(
        "train", load_cached_data=True
    )
    val_ids, val_inputs, val_pmaps, val_targets = get_dataset(
        "val", load_cached_data=True
    )

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_pmaps, train_targets)
    val_dataset = RNADataset(val_inputs, val_pmaps, val_targets)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SRDN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=Config.PATIENCE
    )
    criterion = MCRMSELoss()

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Mask for loss calculation (Scored positions only)
    loss_mask = torch.zeros(Config.SEQ_LENGTH, device=device)
    loss_mask[: Config.SCORED_LENGTH] = 1.0

    # 4. Training Loop
    # Using 15 epochs for a fast baseline execution
    epochs = 15
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0

        for x, pmap, y in train_loader:
            x, pmap, y = x.to(device), pmap.to(device), y.to(device)
            B = x.shape[0]

            # --- Pass 1: Cold Start ---
            recycling_zero = torch.zeros(B, Config.SEQ_LENGTH, 5, device=device)
            x1 = torch.cat([x, recycling_zero], dim=2)
            pred1 = model(x1, pmap)

            # --- Pass 2: Stabilized Recycling ---
            # Detach to stop gradient flow through recycling
            recycling_detached = pred1.detach()
            x2 = torch.cat([x, recycling_detached], dim=2)
            pred2 = model(x2, pmap)

            # --- Loss Calculation ---
            batch_mask = loss_mask.unsqueeze(0).expand(B, -1)

            # Primary loss on refined prediction
            loss_main = criterion(pred2, y, batch_mask)
            # Auxiliary loss on initial prediction (0.5 weight)
            loss_aux = criterion(pred1, y, batch_mask)

            loss = loss_main + 0.5 * loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * B

        avg_train_loss = train_loss_accum / len(train_dataset)

        # --- Validation ---
        model.eval()
        total_sse = torch.zeros(3, device=device)
        total_count = 0

        with torch.no_grad():
            for x, pmap, y in val_loader:
                x, pmap, y = x.to(device), pmap.to(device), y.to(device)
                B = x.shape[0]

                # Pass 1
                recycling = torch.zeros(B, Config.SEQ_LENGTH, 5, device=device)
                x_in = torch.cat([x, recycling], dim=2)
                pred1 = model(x_in, pmap)

                # Pass 2
                recycling = pred1.detach()
                x_in = torch.cat([x, recycling], dim=2)
                pred2 = model(x_in, pmap)

                # Calculate Metric on Scored Columns
                pred_scored = pred2[:, :, Config.SCORED_INDICES]
                target_scored = y[:, :, Config.SCORED_INDICES]

                sq_diff = (pred_scored - target_scored) ** 2
                mask_expanded = loss_mask.view(1, -1, 1).expand(B, Config.SEQ_LENGTH, 3)
                sq_diff = sq_diff * mask_expanded

                total_sse += torch.sum(sq_diff, dim=(0, 1))
                total_count += B * Config.SCORED_LENGTH

        rmse_per_col = torch.sqrt(total_sse / total_count)
        val_mcrmse = torch.mean(rmse_per_col).item()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

        scheduler.step(val_mcrmse)

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation Metric & Failure Analysis
    print("\nPerforming Final Validation and Failure Analysis...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load metadata for analysis (Signal to Noise)
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    val_meta = pd.read_csv(val_meta_path)

    # Store per-sample metrics and features
    errors = []
    sns = []
    a_counts = []

    total_sse_final = torch.zeros(3, device=device)
    total_count_final = 0

    batch_start_idx = 0

    with torch.no_grad():
        for x, pmap, y in val_loader:
            x, pmap, y = x.to(device), pmap.to(device), y.to(device)
            B = x.shape[0]

            # Inference
            recycling = torch.zeros(B, Config.SEQ_LENGTH, 5, device=device)
            x_in = torch.cat([x, recycling], dim=2)
            pred1 = model(x_in, pmap)

            recycling = pred1.detach()
            x_in = torch.cat([x, recycling], dim=2)
            pred2 = model(x_in, pmap)

            # Global Metric Accumulation
            pred_scored = pred2[:, :, Config.SCORED_INDICES]
            target_scored = y[:, :, Config.SCORED_INDICES]

            sq_diff = (pred_scored - target_scored) ** 2
            mask_expanded = loss_mask.view(1, -1, 1).expand(B, Config.SEQ_LENGTH, 3)
            sq_diff = sq_diff * mask_expanded

            total_sse_final += torch.sum(sq_diff, dim=(0, 1))
            total_count_final += B * Config.SCORED_LENGTH

            # Per-sample error calculation for analysis
            # Sum errors over length and channels for each sample
            sample_sse = torch.sum(sq_diff, dim=(1, 2))
            # MSE = SSE / (Scored_Length * 3_channels)
            sample_mse = sample_sse / (Config.SCORED_LENGTH * 3)
            sample_rmse = torch.sqrt(sample_mse)
            errors.extend(sample_rmse.cpu().numpy())

            # Extract Feature: 'A' count (Channel 0 of input)
            # x shape: (B, L, 19)
            a_count = torch.sum(x[:, :, 0], dim=1).cpu().numpy()
            a_counts.extend(a_count)

            # Extract Feature: Signal to Noise from metadata
            # Assuming data loader order matches metadata CSV order
            batch_sn = val_meta.iloc[batch_start_idx : batch_start_idx + B][
                "signal_to_noise"
            ].values
            sns.extend(batch_sn)

            batch_start_idx += B

    # Compute Final Metric
    final_metric = torch.mean(torch.sqrt(total_sse_final / total_count_final)).item()
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    errors = np.array(errors)
    sns = np.array(sns)
    a_counts = np.array(a_counts)

    # Correlation with Signal-to-Noise
    # Filter NaNs if any
    valid_idx = ~np.isnan(sns)
    if np.sum(valid_idx) > 1:
        corr_sn, _ = pearsonr(errors[valid_idx], sns[valid_idx])
        print(f"Correlation between Error and Signal-to-Noise: {corr_sn:.4f}")

    # Correlation with 'A' count
    corr_a, _ = pearsonr(errors, a_counts)
    print(f"Correlation between Error and 'A' count: {corr_a:.4f}")

    # 6. Conditional Submission
    THRESHOLD = 0.5417620723771521

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_ids, test_inputs, test_pmaps = get_dataset("test", load_cached_data=True)
        test_dataset = RNADataset(test_inputs, test_pmaps)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        preds_list = []
        with torch.no_grad():
            for x, pmap in test_loader:
                x, pmap = x.to(device), pmap.to(device)
                B = x.shape[0]

                # Pass 1
                recycling = torch.zeros(B, Config.SEQ_LENGTH, 5, device=device)
                x_in = torch.cat([x, recycling], dim=2)
                pred1 = model(x_in, pmap)

                # Pass 2
                recycling = pred1.detach()
                x_in = torch.cat([x, recycling], dim=2)
                pred2 = model(x_in, pmap)

                preds_list.append(pred2.cpu().numpy())

        all_preds = np.concatenate(preds_list, axis=0)

        # Format Submission
        sub_ids = []
        sub_data = []

        # Flatten predictions
        for i, sample_id in enumerate(test_ids):
            for pos in range(Config.SEQ_LENGTH):
                sub_ids.append(f"{sample_id}_{pos}")
                sub_data.append(all_preds[i, pos])

        sub_data = np.array(sub_data)
        submission_df = pd.DataFrame(sub_data, columns=Config.TARGET_COLS)
        submission_df.insert(0, "id_seqpos", sub_ids)

        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric {final_metric} is NOT lower than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
