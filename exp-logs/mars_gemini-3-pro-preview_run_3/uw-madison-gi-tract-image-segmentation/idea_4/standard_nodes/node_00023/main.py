import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

# Import from provided libraries
from library.config import (
    SEED,
    CHECKPOINT_DIR,
    DEVICE,
    THR_LARGE_BOWEL,
    THR_SMALL_BOWEL,
    THR_STOMACH,
    NUM_CLASSES,
    EPOCHS,
)
from library.utils import set_seed, keep_largest_component_3d
from library.train import Trainer
from library.inference import predict_and_submit
from library.metrics import get_competition_score


def run_validation_and_analysis(model, val_loader):
    """
    Runs inference on the validation set to compute the competition metric
    and aggregates metadata for failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # 1. Inference Loop
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE, dtype=torch.float32)

            with autocast():
                # Forward pass
                outputs = model(images)
                outputs = torch.sigmoid(outputs)

            # Move to CPU to save GPU memory
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Group by Volume (Case + Day)
    val_df = val_loader.dataset.df
    groups = val_df.groupby(["case", "day"])

    scores = []
    meta_stats = []
    thresholds = [THR_LARGE_BOWEL, THR_SMALL_BOWEL, THR_STOMACH]

    # 3. Process Volumes
    for (case, day), group_df in groups:
        indices = group_df.index.values

        # Sort slices by depth to ensure correct 3D reconstruction
        slice_nums = group_df["slice"].values
        sort_idx = np.argsort(slice_nums)

        vol_preds = all_preds[indices][sort_idx]
        vol_targets = all_targets[indices][sort_idx]

        # Calculate score for this volume (average of 3 classes)
        vol_class_scores = []
        for cls_idx in range(NUM_CLASSES):
            # Threshold
            y_pred = (vol_preds[:, cls_idx] > thresholds[cls_idx]).astype(np.uint8)
            y_true = (vol_targets[:, cls_idx] > 0.5).astype(np.uint8)

            # Post-processing (3D CCA)
            y_pred = keep_largest_component_3d(y_pred)

            # Metric
            s = get_competition_score(y_true, y_pred)
            vol_class_scores.append(s)

        mean_vol_score = np.mean(vol_class_scores)
        scores.append(mean_vol_score)

        # Collect Metadata for Failure Analysis
        # Assuming these properties are constant per volume
        h = group_df["height"].iloc[0]
        w = group_df["width"].iloc[0]
        ps_h = group_df["pixel_spacing_h"].iloc[0]
        depth = len(group_df)

        meta_stats.append(
            {
                "score": mean_vol_score,
                "error": 1.0 - mean_vol_score,  # Error Magnitude
                "height": h,
                "width": w,
                "pixel_spacing": ps_h,
                "depth": depth,
            }
        )

    final_metric = np.mean(scores)
    return final_metric, pd.DataFrame(meta_stats)


def main():
    # Ensure reproducibility
    set_seed(SEED)

    print("=== Starting Fast Baseline Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Training
    # -------------------------------------------------------------------------
    # Initialize Trainer. debug=False ensures we use the full dataset for proper learning.
    # We use EPOCHS from config (7) to allow for convergence.
    trainer = Trainer(debug=False)
    trainer.fit(epochs=EPOCHS)

    # -------------------------------------------------------------------------
    # 2. Validation & Metric
    # -------------------------------------------------------------------------
    print("\n=== Validation & Failure Analysis ===")

    # Load the best model saved during training
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: Best model not found. Using current model state.")

    # Run validation
    val_metric, analysis_df = run_validation_and_analysis(
        trainer.model, trainer.val_loader
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")
    if not analysis_df.empty:
        # Calculate correlation between Error (1 - Score) and features
        # We drop 'score' and 'error' from the features list
        features = ["height", "width", "pixel_spacing", "depth"]

        for feat in features:
            if feat in analysis_df.columns:
                corr = analysis_df["error"].corr(analysis_df[feat])
                print(f"Error vs {feat}: {corr:.4f}")
    else:
        print("No validation data available for analysis.")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    print("\n=== Submission Check ===")
    THRESHOLD = 0.448

    if val_metric > THRESHOLD:
        print(
            f"Metric ({val_metric:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(
            checkpoint_path=best_model_path,
            output_dir="./submission",
            output_filename="submission.csv",
        )
    else:
        print(
            f"Metric ({val_metric:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
