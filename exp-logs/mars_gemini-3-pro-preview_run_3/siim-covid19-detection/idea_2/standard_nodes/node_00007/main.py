import os
import sys
import torch
import numpy as np
import pandas as pd
from torchvision.ops import box_iou
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config, seed_everything
from library.dataset import ChestXRayDataset
from library.engine import fit, evaluate
from library.model import get_one_stage_detector, predict_and_submit
from library.utils import collate_fn, load_dataset_dataframe


def perform_failure_analysis(model, device):
    """
    Analyzes the model's performance on the validation set to find correlations
    between error magnitude and input features.
    """
    print("\n--- Performing Failure Analysis ---")

    # Load validation metadata to get features
    df_val = load_dataset_dataframe(split="val", load_cached_data=True)
    # Ensure aspect ratio is calculated
    if "width" in df_val.columns and "height" in df_val.columns:
        df_val["aspect_ratio"] = df_val["width"] / df_val["height"]

    # Create a map for quick feature lookup
    feature_map = df_val.set_index("image_id")[
        ["width", "height", "aspect_ratio"]
    ].to_dict("index")

    # Load validation loader
    val_dataset = ChestXRayDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model.eval()

    errors = []
    widths = []
    heights = []
    aspect_ratios = []

    with torch.no_grad():
        for images, targets, image_ids in val_loader:
            images = images.to(device)

            # Get predictions
            detections = model(images)  # List of dicts

            for i, img_id in enumerate(image_ids):
                # Get GT info
                gt_boxes = targets[i]["boxes"].to(device)

                # Get Pred info
                pred_boxes = detections[i]["boxes"]
                pred_scores = detections[i]["scores"]

                # Filter by threshold for analysis consistency
                mask = pred_scores > Config.CONF_THRESHOLD
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]

                error_magnitude = 0.0

                if len(gt_boxes) > 0:
                    # Case: Objects present in GT
                    if len(pred_boxes) > 0:
                        # Calculate IoU
                        ious = box_iou(gt_boxes, pred_boxes)
                        # Max IoU for each GT box
                        max_ious_per_gt, _ = ious.max(dim=1)
                        # We define error as 1 - mean(max_iou) across objects, or simply 1 - max(all_ious)
                        # Let's use 1 - average best overlap
                        avg_best_iou = max_ious_per_gt.mean().item()
                        error_magnitude = 1.0 - avg_best_iou
                    else:
                        # Missed detection
                        error_magnitude = 1.0
                else:
                    # Case: Background image (Negative)
                    if len(pred_boxes) > 0:
                        # False Positive
                        # Error is proportional to the confidence of the false detection
                        error_magnitude = pred_scores.max().item()
                    else:
                        # Correct Rejection
                        error_magnitude = 0.0

                # Retrieve features
                if img_id in feature_map:
                    feats = feature_map[img_id]
                    errors.append(error_magnitude)
                    widths.append(feats.get("width", 0))
                    heights.append(feats.get("height", 0))
                    aspect_ratios.append(feats.get("aspect_ratio", 0))

    # Calculate Correlations
    if len(errors) > 0:
        df_analysis = pd.DataFrame(
            {
                "error": errors,
                "width": widths,
                "height": heights,
                "aspect_ratio": aspect_ratios,
            }
        )

        print("\nCorrelation between Error Magnitude and Input Features:")
        corr = df_analysis.corr()["error"].drop("error")
        print(corr)
    else:
        print("No data collected for failure analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training
    # Using 10 epochs for a fast baseline within 2 hours
    fit(
        epochs=10,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 3. Validation Assessment
    # Load the best model saved during training
    model = get_one_stage_detector()
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Error: Model file not found after training.")
        return

    model.to(Config.DEVICE)
    model.eval()

    # Create validation loader
    val_dataset = ChestXRayDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Compute metrics
    metrics = evaluate(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metrics['map']}")

    # 4. Failure Analysis
    perform_failure_analysis(model, Config.DEVICE)

    # 5. Submission
    # This function handles test loading, inference, and saving submission.csv
    predict_and_submit(load_cached_data=True)


if __name__ == "__main__":
    main()
