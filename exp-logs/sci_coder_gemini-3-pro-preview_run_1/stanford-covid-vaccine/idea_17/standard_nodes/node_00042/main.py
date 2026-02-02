import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

# Import from library
from library.config import ModelConfig
from library.utils import set_seed, get_device, mcrmse_loss, save_checkpoint
from library.dataset import load_or_process_data, RNADataset
from library.model import RNARegressor


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()

    # Override config for fast baseline execution
    # 15 epochs is sufficient for convergence on this dataset size (~1700 samples)
    ModelConfig.num_epochs = 15
    ModelConfig.batch_size = 32

    print(f"Device: {device}")
    print(f"Epochs: {ModelConfig.num_epochs}")

    # 2. Data Loading
    # Load cached data if available to save time
    train_data, val_data, test_data = load_or_process_data(load_cached_data=True)

    train_ds = RNADataset(train_data, mode="train")
    val_ds = RNADataset(val_data, mode="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = RNARegressor(config=ModelConfig).to(device)
    optimizer = AdamW(model.parameters(), lr=ModelConfig.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=ModelConfig.num_epochs)

    # Loss function: MSE (L2) as per strategy
    criterion = nn.MSELoss()

    best_mcrmse = float("inf")
    best_model_path = os.path.join(ModelConfig.output_dir, "best_model.pth")
    os.makedirs(ModelConfig.output_dir, exist_ok=True)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(ModelConfig.num_epochs):
        model.train()

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)  # (B, 68, 3)

            optimizer.zero_grad()

            # Forward pass
            preds = model(seq, loop, dist, mask)  # (B, 107, 3)

            # Mask to scored positions (first 68)
            preds_scored = preds[:, :68, :]

            loss = criterion(preds_scored, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)

                preds = model(seq, loop, dist, mask)
                preds_scored = preds[:, :68, :]

                val_preds_list.append(preds_scored)
                val_targets_list.append(target)

        val_preds_tensor = torch.cat(val_preds_list, dim=0)
        val_targets_tensor = torch.cat(val_targets_list, dim=0)

        # Calculate MCRMSE
        val_mcrmse = mcrmse_loss(val_targets_tensor, val_preds_tensor).item()

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation & Metric Printing
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist, mask)
            preds_scored = preds[:, :68, :]

            val_preds_list.append(preds_scored.cpu())
            val_targets_list.append(target.cpu())
            val_ids_list.extend(ids)

    val_preds_tensor = torch.cat(val_preds_list, dim=0)
    val_targets_tensor = torch.cat(val_targets_list, dim=0)

    final_metric = mcrmse_loss(val_targets_tensor, val_preds_tensor).item()
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate MCRMSE per sample
    # MSE per sample per column: mean over dim 1 (seq_len=68)
    mse_per_sample_col = torch.mean(
        (val_preds_tensor - val_targets_tensor) ** 2, dim=1
    )  # (N, 3)
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)  # (N, 3)
    # MCRMSE per sample: mean over dim 1 (columns=3)
    mcrmse_per_sample = torch.mean(rmse_per_sample_col, dim=1).numpy()  # (N,)

    # Load metadata to correlate
    val_meta_df = pd.read_parquet(ModelConfig.val_file)

    # Create a dataframe for analysis
    analysis_df = pd.DataFrame({"id": val_ids_list, "error": mcrmse_per_sample})

    # Merge with metadata (signal_to_noise, SN_filter)
    if "signal_to_noise" in val_meta_df.columns:
        analysis_df = analysis_df.merge(
            val_meta_df[["id", "signal_to_noise", "SN_filter"]], on="id", how="left"
        )

        # Correlation with Signal to Noise
        if not analysis_df["signal_to_noise"].isnull().all():
            corr_sn, _ = pearsonr(analysis_df["error"], analysis_df["signal_to_noise"])
            print(f"Correlation between Error and Signal_to_Noise: {corr_sn:.4f}")

        # Correlation with SN_filter
        if not analysis_df["SN_filter"].isnull().all():
            corr_filter, _ = pearsonr(analysis_df["error"], analysis_df["SN_filter"])
            print(f"Correlation between Error and SN_filter: {corr_filter:.4f}")
    else:
        print("Metadata columns for failure analysis not found.")

    # 7. Conditional Submission
    THRESHOLD = 0.6226052641868591
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_ds = RNADataset(test_data, mode="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=ModelConfig.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                mask = batch["mask"].to(device)
                batch_ids = batch["id"]

                preds = model(seq, loop, dist, mask)  # (B, 107, 3)
                all_preds.append(preds.cpu().numpy())
                all_ids.extend(batch_ids)

        all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 3)

        # Format submission
        submission_rows = []
        seq_len = 107

        for i, sample_id in enumerate(all_ids):
            sample_preds = all_preds[i]
            for pos in range(seq_len):
                row_id = f"{sample_id}_{pos}"
                # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
                reactivity = float(sample_preds[pos, 0])
                deg_Mg_pH10 = float(sample_preds[pos, 1])
                deg_Mg_50C = float(sample_preds[pos, 2])

                # Unscored columns filled with 0.0
                submission_rows.append(
                    [row_id, reactivity, deg_Mg_pH10, 0.0, deg_Mg_50C, 0.0]
                )

        columns = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        sub_df = pd.DataFrame(submission_rows, columns=columns)

        sub_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
