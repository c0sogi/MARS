import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import CFG
from library.train import train_model
from library.predict import predict_and_submit
from library.dataset import ContrailDataset, get_transforms
from library.model import ConvNeXtUNet
from library.utils import set_seed


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override default config to ensure execution within 2 hours
    # while maintaining enough data to reach the target metric.
    CFG.epochs = 5
    CFG.debug_sample_size = 10000  # Use ~10k samples for training
    CFG.batch_size = 32

    # Set seed for reproducibility
    set_seed(CFG.seed)

    print("Starting Fast Baseline Run...")
    print(
        f"Configuration: Epochs={CFG.epochs}, Training Samples={CFG.debug_sample_size}"
    )

    # ==========================================
    # 2. Train Model
    # ==========================================
    # debug=True triggers the use of CFG.debug_sample_size
    train_model(debug=True)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")

    # Load the FULL validation dataset (no subsampling)
    val_dataset = ContrailDataset(
        metadata_path=CFG.valid_metadata_path,
        split="validation",
        transform=get_transforms("validation"),
        debug=False,
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Load the best model weights
    model = ConvNeXtUNet(
        backbone_name=CFG.backbone,
        in_channels=CFG.in_channels,
        num_classes=CFG.out_channels,
        pretrained=False,
    )

    if os.path.exists(CFG.best_model_path):
        state_dict = torch.load(CFG.best_model_path, map_location=CFG.device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {CFG.best_model_path}")
    else:
        print("Error: Best model weights not found.")
        return

    model.to(CFG.device)
    model.eval()

    # Variables for Global Dice
    val_intersection = 0.0
    val_union = 0.0

    # Variables for Failure Analysis
    sample_errors = []  # List of dicts: {'record_id': str, 'error': float}

    # Load metadata for correlation analysis
    meta_df = pd.read_csv(CFG.valid_metadata_path)
    meta_df["record_id"] = meta_df["record_id"].astype(str)

    # Feature Engineering for Analysis
    if "timestamp" in meta_df.columns:
        meta_df["datetime"] = pd.to_datetime(meta_df["timestamp"], unit="s")
        meta_df["hour"] = meta_df["datetime"].dt.hour

    print("Running Inference on Validation Set...")

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(CFG.device, dtype=torch.float32)
            masks = batch["mask"].to(CFG.device, dtype=torch.float32)
            record_ids = batch["record_id"]

            # Inference with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                preds_bin = (preds > CFG.threshold).float()

            # --- Global Metric Accumulation ---
            p_flat = preds_bin.view(-1)
            t_flat = masks.view(-1)

            intersection = (p_flat * t_flat).sum().item()
            union = p_flat.sum().item() + t_flat.sum().item()

            val_intersection += intersection
            val_union += union

            # --- Per-Sample Failure Analysis ---
            probs_np = preds_bin.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(record_ids)):
                p = probs_np[i].flatten()
                t = masks_np[i].flatten()

                inter = (p * t).sum()
                uni = p.sum() + t.sum()

                # Calculate Dice for this specific image
                if uni == 0:
                    dice = 1.0
                else:
                    dice = (2.0 * inter) / uni

                # Error magnitude is 1 - Dice
                sample_errors.append(
                    {"record_id": str(record_ids[i]), "error": 1.0 - dice}
                )

    # Compute Final Global Dice
    if val_union == 0:
        final_metric = 1.0 if val_intersection == 0 else 0.0
    else:
        final_metric = (2.0 * val_intersection) / val_union

    # REQUIRED PRINT
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis Report ---
    print("\nPerforming Failure Analysis...")
    errors_df = pd.DataFrame(sample_errors)

    # Merge errors with metadata features
    analysis_df = errors_df.merge(meta_df, on="record_id", how="left")

    features_to_analyze = ["hour", "row_min", "col_min"]
    correlations = {}

    for feat in features_to_analyze:
        if feat in analysis_df.columns:
            # Clean data for correlation
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                # Compute Pearson correlation
                corr = np.corrcoef(valid_data[feat], valid_data["error"])[0, 1]
                correlations[feat] = corr
            else:
                correlations[feat] = 0.0

    print("Correlation between Error Magnitude (1-Dice) and Input Features:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = 0.5910660985501295

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # Run inference on test set and generate submission.csv
        predict_and_submit(debug=False)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
