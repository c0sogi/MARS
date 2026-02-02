import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.utils import seed_everything, calc_map, unpad_image
from library.dataset import get_dataloaders
from library.model import ResNet34WideLinkNet
from library.engine import train_and_evaluate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def calculate_iou_per_image(preds, targets, threshold=0.5):
    """
    Calculates IoU for each image in the batch.
    """
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > threshold).astype(np.uint8)

    preds_flat = preds_bin.reshape(preds_bin.shape[0], -1)
    targets_flat = targets_bin.reshape(targets_bin.shape[0], -1)

    intersection = (preds_flat * targets_flat).sum(axis=1)
    union = preds_flat.sum(axis=1) + targets_flat.sum(axis=1) - intersection

    ious = np.zeros(preds.shape[0])
    # Handle division by zero (empty union means perfect match of background)
    union_nonzero = union > 0
    ious[union_nonzero] = intersection[union_nonzero] / union[union_nonzero]
    ious[~union_nonzero] = 1.0

    return ious


def run_failure_analysis(model, dataloader, device, val_metadata_path):
    """
    Runs inference on validation set, calculates metrics, and performs failure analysis.
    """
    print("Running Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, masks, depths, ids in dataloader:
            images = images.to(device)
            depths = depths.to(device)

            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            probs_np = probs.detach().cpu().numpy().squeeze(1)
            masks_np = masks.detach().cpu().numpy().squeeze(1)

            # Unpad to original size
            for i in range(probs_np.shape[0]):
                p = unpad_image(probs_np[i], original_size=101)
                m = unpad_image(masks_np[i], original_size=101)
                all_preds.append(p)
                all_targets.append(m)
                all_ids.append(ids[i])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_ids = np.array(all_ids)

    # 1. Calculate and Print Final Validation Metric (mAP) with Optimization
    best_map = -1.0
    best_thresh = 0.5
    search_space = np.linspace(0.3, 0.7, 9)

    for t in search_space:
        score = calc_map(all_preds, all_targets, binarization_threshold=t)
        if score > best_map:
            best_map = score
            best_thresh = t

    print(f"Final Validation Metric: {best_map} (Threshold: {best_thresh:.2f})")

    # 2. Failure Analysis
    # Calculate IoU per image using the optimized threshold
    ious = calculate_iou_per_image(all_preds, all_targets, threshold=best_thresh)
    errors = 1.0 - ious

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame({"id": all_ids, "iou": ious, "error": errors})

    # Load metadata to get features (depth, coverage)
    df_meta = pd.read_csv(val_metadata_path)

    # Merge
    df_merged = pd.merge(df_analysis, df_meta, on="id", how="left")

    # Calculate Correlations
    # Correlation between Error and Depth (z)
    corr_depth = df_merged["error"].corr(df_merged["z"])

    # Correlation between Error and Salt Coverage
    corr_coverage = df_merged["error"].corr(df_merged["salt_coverage"])

    print("\nFailure Analysis Report:")
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.4f}")

    return best_map, best_thresh


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Train Model
    # Increase epochs to 50 for full convergence (Cite solution_lesson_node_00035)
    print("Starting training pipeline...")
    best_model_path, depth_stats, best_threshold = train_and_evaluate(
        num_epochs=50, batch_size=32, learning_rate=1e-4, weight_decay=1e-2, patience=15
    )

    # 3. Validation & Failure Analysis
    # Reload best model
    model = ResNet34WideLinkNet(pretrained=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)

    # Get Validation Loader
    _, val_loader, _ = get_dataloaders(batch_size=32, load_cached_data=True)

    # Run Analysis (re-optimizes threshold on validation set)
    val_map, val_thresh = run_failure_analysis(
        model, val_loader, device, "./metadata/val.csv"
    )

    # 4. Submission
    # Threshold check
    TARGET_THRESHOLD = 0.7985

    if val_map > TARGET_THRESHOLD:
        print(f"\nValidation metric {val_map} exceeds threshold {TARGET_THRESHOLD}.")
        # Generate submission using Marginalized Depth-Scan with optimized threshold
        # We use the threshold found during failure analysis (which is the same as training if data is same)
        generate_submission(
            best_model_path, depth_stats, batch_size=32, threshold=val_thresh
        )
    else:
        print(
            f"\nValidation metric {val_map} does not meet threshold {TARGET_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
