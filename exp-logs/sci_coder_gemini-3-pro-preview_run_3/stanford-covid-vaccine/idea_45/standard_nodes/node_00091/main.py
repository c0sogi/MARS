import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_loader
from library.model import DSDBiGRUModel
from library.train import Trainer


def main():
    # 1. Configuration and Setup
    config = Config()

    # Override paths to meet specific Task requirements
    config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Set fixed seed for reproducibility
    set_seed(config.SEED)

    print(f"Starting execution on device: {config.DEVICE}")

    # 2. Training
    # Initialize Trainer with the config
    trainer = Trainer(config)

    # Run training (handles training loop, validation, early stopping, saving best model)
    print("Starting training phase...")
    trainer.run()

    # 3. Validation & Metrics
    print("\nLoading best model for final validation assessment...")

    # Re-initialize model structure
    model = DSDBiGRUModel(config).to(config.DEVICE)

    # Load weights
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {config.BEST_MODEL_PATH}")
        return

    model.load_state_dict(
        torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    )
    model.eval()

    # Get Validation Loader
    val_loader = get_loader(
        "val",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # Inference on Validation
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(config.DEVICE)
            bpp_indices = batch["bpp_indices"].to(config.DEVICE)
            bpp_mask = batch["bpp_mask"].to(config.DEVICE)
            targets = batch["targets"].to(config.DEVICE)
            ids = batch["id"]

            outputs = model(inputs, bpp_indices, bpp_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Metric
    # mcrmse_metric handles slicing to config.PRED_LEN and selecting specific columns internally
    final_metric = mcrmse_metric(all_preds, all_targets)

    # Print strictly required format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n==== Failure Analysis ====")

    # Calculate error per sample for analysis
    # We focus on the metric-relevant columns (0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C)
    # and the scored sequence length
    scored_cols = [0, 1, 3]
    preds_scored = all_preds[:, : config.PRED_LEN, scored_cols]
    targets_scored = all_targets[:, : config.PRED_LEN, scored_cols]

    # MSE per sample (average over positions and selected columns)
    mse_per_sample = torch.mean((preds_scored - targets_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load Validation Metadata to get features
    val_df = pd.read_parquet(config.VAL_METADATA)

    # Create analysis dataframe
    # We merge on ID to ensure alignment
    error_df = pd.DataFrame({"id": all_ids, "rmse_error": rmse_per_sample})
    analysis_df = val_df.merge(error_df, on="id", how="inner")

    # Feature Engineering for Analysis
    # GC Content: Fraction of G and C in the sequence
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )

    # Select features to correlate
    features_to_analyze = ["signal_to_noise", "SN_filter", "gc_content"]

    print("Correlation between Model Error (RMSE) and Input Features:")
    for feat in features_to_analyze:
        if feat in analysis_df.columns:
            # Calculate Pearson correlation
            corr = analysis_df["rmse_error"].corr(analysis_df[feat])
            print(f"  {feat}: {corr:.8f}")
        else:
            print(f"  {feat}: Not found in metadata")

    # 5. Submission Generation
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Get Test Loader
        test_loader = get_loader(
            "test",
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(config.DEVICE)
                bpp_indices = batch["bpp_indices"].to(config.DEVICE)
                bpp_mask = batch["bpp_mask"].to(config.DEVICE)
                ids = batch["id"]

                outputs = model(inputs, bpp_indices, bpp_mask)

                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        # Concatenate predictions: (N_test, 107, 5)
        test_preds = np.concatenate(test_preds, axis=0)

        # Prepare Submission DataFrame
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_rows = []
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids):
            # For each sample, we have 107 positions
            sample_pred = test_preds[i]  # (107, 5)

            for seqpos in range(config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                values = sample_pred[seqpos].tolist()
                submission_rows.append([row_id] + values)

        submission_df = pd.DataFrame(
            submission_rows, columns=["id_seqpos"] + target_cols
        )

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
