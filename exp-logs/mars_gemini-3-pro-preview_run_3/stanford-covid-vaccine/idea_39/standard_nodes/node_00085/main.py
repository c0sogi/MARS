import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.dataset import get_dataset
from library.model import DeepDecoupledBiGRU
from library.trainer import Trainer
from library.loss_metric import compute_metric


def main():
    # 1. Setup & Configuration
    # Limit epochs to 15 for a fast baseline execution
    config = Config(epochs=15, batch_size=32)
    set_seed(config.seed)

    print(
        f"Configuration: Device={config.device}, Epochs={config.epochs}, Batch Size={config.batch_size}"
    )

    # 2. Data Loading
    print("Loading datasets...")
    # Using load_cached_data=True to leverage any existing preprocessed files
    train_dataset = get_dataset("train", config, load_cached_data=True)
    val_dataset = get_dataset("val", config, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = DeepDecoupledBiGRU(config)

    # 4. Training
    print("Starting training...")
    trainer = Trainer(config, model, train_loader, val_loader)
    trainer.fit()

    # 5. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )
    model.to(config.device)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(config.device)
            bpp_indices = batch["bpp_indices"].to(config.device)
            bpp_mask = batch["bpp_mask"].to(config.device)
            targets = batch["targets"]  # Keep on CPU for accumulation

            outputs = model(inputs, bpp_indices, bpp_mask)
            all_preds.append(outputs.cpu())
            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Metric
    metric = compute_metric(all_preds, all_targets, config)
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Calculate error per sample
    # Slice to scored length (68) and scored columns
    pred_len = config.pred_len
    preds_sliced = all_preds[:, :pred_len, :].numpy()
    targets_sliced = all_targets[:, :pred_len, :].numpy()

    scored_indices = [
        i for i, col in enumerate(config.target_cols) if col in config.scored_cols
    ]

    # Compute RMSE per sample over the scored columns
    # Shape: (N, 68, 3) -> (N,)
    diff_sq = (
        preds_sliced[:, :, scored_indices] - targets_sliced[:, :, scored_indices]
    ) ** 2
    mse_per_sample = np.nanmean(diff_sq, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata to get features
    val_meta_df = pd.read_parquet(config.val_metadata_path)

    # Ensure alignment (dataset loader preserves order)
    if len(val_meta_df) != len(rmse_per_sample):
        print(
            "Warning: Metadata length mismatch. Skipping detailed correlation analysis."
        )
    else:
        analysis_df = val_meta_df.copy()
        analysis_df["error_magnitude"] = rmse_per_sample

        # Feature Engineering
        analysis_df["pct_G"] = analysis_df["sequence"].apply(
            lambda s: s.count("G") / len(s)
        )
        analysis_df["pct_C"] = analysis_df["sequence"].apply(
            lambda s: s.count("C") / len(s)
        )
        analysis_df["pct_A"] = analysis_df["sequence"].apply(
            lambda s: s.count("A") / len(s)
        )
        analysis_df["pct_U"] = analysis_df["sequence"].apply(
            lambda s: s.count("U") / len(s)
        )

        # Correlations
        features = ["signal_to_noise", "SN_filter", "pct_G", "pct_C", "pct_A", "pct_U"]
        print("Correlation between Error Magnitude and Features:")
        for feat in features:
            if feat in analysis_df.columns:
                corr = analysis_df["error_magnitude"].corr(analysis_df[feat])
                print(f"  {feat}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.5978901386
    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = get_dataset("test", config, load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(config.device)
                bpp_indices = batch["bpp_indices"].to(config.device)
                bpp_mask = batch["bpp_mask"].to(config.device)
                ids = batch["ids"]

                outputs = model(inputs, bpp_indices, bpp_mask)
                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        # Concatenate predictions: (N_test, 107, 5)
        test_preds = np.concatenate(test_preds, axis=0)

        # Flatten for submission format
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_rows = []
        target_cols = (
            config.target_cols
        )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds[i]  # (107, 5)
            for seqpos in range(config.seq_len):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos].tolist()
                submission_rows.append([row_id] + row_values)

        submission_df = pd.DataFrame(
            submission_rows, columns=["id_seqpos"] + target_cols
        )

        # Save
        os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
        submission_df.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")

    else:
        print(
            f"\nMetric ({metric}) >= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
