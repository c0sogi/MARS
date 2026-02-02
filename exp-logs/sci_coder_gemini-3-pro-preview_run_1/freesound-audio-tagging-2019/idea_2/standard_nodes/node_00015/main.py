import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from library.config import CFG
from library.utils import set_seed, calculate_overall_lwlrap
from library.trainer import Trainer
from library.dataset import get_dataloader


def main():
    # 1. Configuration Overrides for Fast Baseline
    # We reduce epochs to ensure completion within the time limit while maintaining performance.
    # EfficientNet-B2 converges relatively quickly.
    CFG.epochs = 10

    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Set seeds
    set_seed(CFG.seed)

    print(f"Initializing training with {CFG.epochs} epochs...")

    # 2. Training
    trainer = Trainer()
    trainer.fit(epochs=CFG.epochs)

    # 3. Validation & Metric Calculation
    print("Performing final validation...")

    # Load the best model explicitly for validation analysis
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model not found after training.")

    trainer.model.load_state_dict(torch.load(best_model_path, map_location=CFG.device))
    trainer.model.eval()

    val_loader = get_dataloader("val", debug=CFG.debug)

    all_preds = []
    all_targets = []
    all_fnames = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(CFG.device)
            targets = batch["target"].to(CFG.device)
            fnames = batch["fname"]

            outputs = trainer.model(images)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_fnames.extend(fnames)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Final Metric
    final_metric = calculate_overall_lwlrap(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate per-sample error (Mean Absolute Error across classes)
    # shape: (n_samples,)
    sample_errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Load Validation Metadata to get features
    val_df = pd.read_csv(CFG.val_csv)

    # Ensure alignment: val_loader is not shuffled, so order should match val_df
    # We can double check with fnames if needed, but standard loaders preserve order if shuffle=False
    if len(val_df) != len(sample_errors):
        print(
            "Warning: Validation dataframe length mismatch. Skipping detailed analysis."
        )
    else:
        val_df["error"] = sample_errors

        # Feature 1: Label Count (Cardinality)
        # Identify label columns (exclude metadata)
        exclude_cols = {"fname", "labels", "filepath", "error"}
        label_cols = [c for c in val_df.columns if c not in exclude_cols]
        val_df["label_count"] = val_df[label_cols].sum(axis=1)

        # Feature 2: Audio Duration
        # We need to read this from files as it's not in the minimal metadata csv
        durations = []
        for idx, row in val_df.iterrows():
            try:
                full_path = os.path.join(CFG.input_root, row["filepath"])
                info = sf.info(full_path)
                durations.append(info.duration)
            except:
                durations.append(np.nan)
        val_df["duration"] = durations

        # Calculate Correlations
        # Drop NaNs if any file read failed
        analysis_df = val_df.dropna(subset=["duration", "label_count", "error"])

        corr_duration = analysis_df["duration"].corr(analysis_df["error"])
        corr_labels = analysis_df["label_count"].corr(analysis_df["error"])

        print(f"Correlation between Error and Audio Duration: {corr_duration:.4f}")
        print(f"Correlation between Error and Label Count: {corr_labels:.4f}")

    # 5. Submission
    THRESHOLD = 0.7883450707332457

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
