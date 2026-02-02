import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import (
    VAL_METADATA_PATH,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE,
    DOWN_RATIO,
    CONF_THRESHOLD,
    MAX_DETECTIONS,
    WORKING_DIR,
)
from library.utils import (
    seed_everything,
    load_and_parse_metadata,
    post_process_coords,
    collate_fn,
)
from library.dataset import KuzushijiDataset, get_class_mapping
from library.model import ConvNextCenterNet
from library.engine import run_training, predict, _decode
from library.loss import CenterNetLoss


def main():
    # Set seeds for reproducibility
    seed_everything()

    # 1. Train the model
    # We limit epochs to 12 to ensure the "fast baseline" completes within the time limit (approx 2.5 hours remaining).
    # ConvNeXt-Tiny converges relatively fast.
    print("Starting Training...")
    best_model_path = run_training(debug=False, epochs=12)

    # 2. Validation & Failure Analysis
    print("Starting Validation and Failure Analysis...")

    # Load the best model
    model = ConvNextCenterNet().to(DEVICE)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Error: Best model not found.")
        return

    model.eval()

    # Prepare Validation Data
    val_dataset = KuzushijiDataset(VAL_METADATA_PATH, split="val", debug=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load Ground Truth for Metric Calculation
    gt_data = load_and_parse_metadata(VAL_METADATA_PATH, load_cached_data=True)
    gt_map = {item["image_id"]: item["annotations"] for item in gt_data}
    _, id_to_char = get_class_mapping()

    # Variables for Global Metric
    tp_global = 0
    fp_global = 0
    fn_global = 0

    # Variables for Failure Analysis
    image_stats = []

    # Inference Loop (No Gradients)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            image_ids = batch["image_id"]
            original_shapes = batch["original_shape"]

            # Forward pass
            outputs = model(images)

            # Decode outputs
            scores, xs, ys, cls_ids = _decode(
                outputs["hm"],
                outputs["wh"],
                outputs["reg"],
                outputs["cls_logits"],
                k=MAX_DETECTIONS,
            )

            batch_size_curr = images.size(0)

            for i in range(batch_size_curr):
                img_id = image_ids[i]
                orig_shape = original_shapes[i]

                # Filter by confidence threshold
                mask = scores[i] > CONF_THRESHOLD

                pred_points = []
                pred_classes = []
                pred_scores = []

                if mask.sum() > 0:
                    valid_scores = scores[i][mask].cpu().numpy()
                    valid_xs = xs[i][mask].cpu().numpy()
                    valid_ys = ys[i][mask].cpu().numpy()
                    valid_cls = cls_ids[i][mask].cpu().numpy()

                    for j in range(len(valid_scores)):
                        # Map from feature map to input image
                        ix = valid_xs[j] * DOWN_RATIO
                        iy = valid_ys[j] * DOWN_RATIO

                        # Map from input image to original image
                        ox, oy = post_process_coords(ix, iy, orig_shape, IMG_SIZE)

                        pred_points.append((ox, oy))
                        pred_classes.append(id_to_char[valid_cls[j]])
                        pred_scores.append(valid_scores[j])

                # Retrieve Ground Truth
                gt_anns = gt_map.get(img_id, [])

                # Matching Logic (Greedy by Score)
                preds = sorted(
                    zip(pred_points, pred_classes, pred_scores),
                    key=lambda x: x[2],
                    reverse=True,
                )

                matched_gt = set()
                tp = 0
                fp = 0

                for (px, py), p_label, p_score in preds:
                    match_found = False
                    # Check against unmatched GTs
                    for gt_idx, ann in enumerate(gt_anns):
                        if gt_idx in matched_gt:
                            continue

                        if ann["label"] != p_label:
                            continue

                        gx, gy, gw, gh = ann["bbox"]

                        # Point inside BBox check
                        if (gx <= px <= gx + gw) and (gy <= py <= gy + gh):
                            matched_gt.add(gt_idx)
                            match_found = True
                            break

                    if match_found:
                        tp += 1
                    else:
                        fp += 1

                fn = len(gt_anns) - len(matched_gt)

                # Update Global Counts
                tp_global += tp
                fp_global += fp
                fn_global += fn

                # Per-Image F1 for Failure Analysis
                prec_img = tp / (tp + fp + 1e-8)
                rec_img = tp / (tp + fn + 1e-8)
                f1_img = 2 * (prec_img * rec_img) / (prec_img + rec_img + 1e-8)

                # Error Magnitude (1 - F1)
                error_mag = 1.0 - f1_img

                image_stats.append(
                    {
                        "image_id": img_id,
                        "error_magnitude": error_mag,
                        "img_height": orig_shape[0],
                        "img_width": orig_shape[1],
                        "num_anns": len(gt_anns),
                    }
                )

    # Calculate Final Global Metric
    precision = tp_global / (tp_global + fp_global + 1e-8)
    recall = tp_global / (tp_global + fn_global + 1e-8)
    f1_global = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"Final Validation Metric: {f1_global}")

    # Failure Analysis
    df_stats = pd.DataFrame(image_stats)
    if not df_stats.empty:
        corr_width = df_stats["error_magnitude"].corr(df_stats["img_width"])
        corr_height = df_stats["error_magnitude"].corr(df_stats["img_height"])
        corr_anns = df_stats["error_magnitude"].corr(df_stats["num_anns"])

        print("\nFailure Analysis (Correlation with Error Magnitude):")
        print(f"Image Width: {corr_width}")
        print(f"Image Height: {corr_height}")
        print(f"Number of Annotations: {corr_anns}")

    # Submission Logic
    THRESHOLD = 0.7679033467456621
    if f1_global > THRESHOLD:
        print(
            f"\nMetric ({f1_global}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict(model_path=best_model_path, debug=False)
    else:
        print(
            f"\nMetric ({f1_global}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
