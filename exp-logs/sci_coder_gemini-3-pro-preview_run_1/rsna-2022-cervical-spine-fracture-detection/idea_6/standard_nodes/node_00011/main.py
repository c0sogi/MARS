import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import random

# Ensure library is in path
sys.path.append(os.getcwd())

from library import config

# Override global batch size for inference speed before importing modules that rely on it
config.BATCH_SIZE = 32

from library import (
    train_segmentation,
    train_encoder,
    generate_features,
    train_aggregator,
    inference,
    datasets,
    models,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_validation_and_analysis():
    print("Running Validation and Failure Analysis...")

    # 1. Load Validation Data (Stage 3 format)
    _, val_ds = datasets.get_datasets(stage="stage3")
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,  # Batch size 1 for granular analysis per patient
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # 2. Load Model
    model = inference.load_stage3_model(config.DEVICE)
    criterion = train_aggregator.WeightedMultiLabelLogLoss(config.DEVICE)

    losses = []
    rows = []

    model.eval()
    with torch.no_grad():
        for i, (visual_feats, anat_ids, labels) in enumerate(val_loader):
            visual_feats = visual_feats.to(config.DEVICE)
            anat_ids = anat_ids.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Forward
            logits = model(visual_feats, anat_ids)

            # Calculate Loss (Weighted Log Loss)
            # The criterion returns mean loss over batch (which is 1 here)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            # Metadata
            meta_row = val_ds.metadata.iloc[i]
            rows.append(meta_row)

    # 3. Final Metric
    final_metric = np.mean(losses)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    df_analysis = pd.DataFrame(rows)
    df_analysis["loss"] = losses

    # Feature: Fracture Count (Proxy for severity/complexity)
    target_cols = config.VERTEBRAE_CLASSES + ["patient_overall"]
    df_analysis["fracture_count"] = df_analysis[target_cols].sum(axis=1)

    # Feature: Has Segmentation (Proxy for data quality/source)
    # Handle boolean column
    if "has_segmentation" in df_analysis.columns:
        df_analysis["has_segmentation"] = df_analysis["has_segmentation"].astype(int)
    else:
        df_analysis["has_segmentation"] = 0

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(
        f"Loss Statistics: Mean={np.mean(losses):.4f}, Std={np.std(losses):.4f}, Max={np.max(losses):.4f}"
    )

    # Correlation
    if df_analysis["fracture_count"].std() > 0:
        corr_frac = df_analysis["loss"].corr(df_analysis["fracture_count"])
        print(f"Correlation (Loss vs Fracture Count): {corr_frac:.4f}")
    else:
        print("Correlation (Loss vs Fracture Count): N/A (No variance)")

    if df_analysis["has_segmentation"].std() > 0:
        corr_seg = df_analysis["loss"].corr(df_analysis["has_segmentation"])
        print(f"Correlation (Loss vs Has Segmentation): {corr_seg:.4f}")
    else:
        print("Correlation (Loss vs Has Segmentation): N/A (No variance)")

    return final_metric


def main():
    parser = argparse.ArgumentParser(
        description="Cervical Spine Fracture Detection Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train", "inference"],
        help="Execution mode",
    )
    args, _ = parser.parse_known_args()

    set_seed(config.SEED)

    # Threshold for submission
    METRIC_THRESHOLD = 0.9254394427010018

    if args.mode in ["all", "train"]:
        print("=== Starting Training Pipeline ===")

        # Step 1: Train Localizer (Stage 1)
        # Fast baseline: 1 epoch to learn basic anatomy
        train_segmentation.run_stage1_training(epochs=1, batch_size=16)

        # Step 2: Train Encoder (Stage 2)
        # Fast baseline: 1 epoch to learn basic fracture features
        train_encoder.run_stage2_training(epochs=1, batch_size=32)

        # Step 3: Generate Features
        # This processes Train, Val, and Test sets.
        # We use cached data if available to speed up re-runs.
        generate_features.generate_features(load_cached_data=True)

        # Step 4: Train Aggregator (Stage 3)
        # Fast baseline: 5 epochs (lightweight model)
        train_aggregator.run_stage3_training(epochs=5, batch_size=4)

        # Step 5: Validation & Analysis
        val_metric = run_validation_and_analysis()

        # Step 6: Conditional Submission
        if val_metric < METRIC_THRESHOLD:
            print(
                f"Validation metric {val_metric} < {METRIC_THRESHOLD}. Generating submission..."
            )
            # Run inference (will use features generated in Step 3)
            inference.run_inference(load_cached_data=True)
        else:
            print(
                f"Validation metric {val_metric} >= {METRIC_THRESHOLD}. Skipping submission."
            )

    elif args.mode == "inference":
        print("=== Starting Inference Mode ===")
        # Assumes models are trained.
        # Ensure features are generated
        generate_features.generate_features(load_cached_data=True)
        inference.run_inference(load_cached_data=True)


if __name__ == "__main__":
    main()
