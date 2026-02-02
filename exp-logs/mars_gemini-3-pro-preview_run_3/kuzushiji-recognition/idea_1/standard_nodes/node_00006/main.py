import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import warnings

# Import provided library modules
from library import config, trainer, inference, utils, dataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def calculate_iou_point_box(point, box):
    """
    Checks if a point is inside a bounding box.
    point: [x, y]
    box: [x, y, w, h]
    """
    px, py = point
    bx, by, bw, bh = box
    return (bx <= px <= bx + bw) and (by <= py <= by + bh)


def evaluate_and_analyze(model, val_dataset, device):
    """
    Evaluates the model on the validation set using the competition metric
    and performs failure analysis.
    """
    print("Starting validation evaluation...")
    model.eval()

    # Create DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Map indices back to Unicode characters
    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}

    # Pre-process Ground Truth for fast lookup
    # val_dataset.data is a list of dicts with 'image_id' and 'labels'
    gt_map = {}
    for item in val_dataset.data:
        gt_map[item["image_id"]] = item["labels"]

    tp_global = 0
    fp_global = 0
    fn_global = 0

    image_stats = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)

            # Forward Pass
            hm, reg, emb = model(images)

            # Decode predictions
            scores, cls_ids, xs, ys = inference.decode(hm, reg, emb, model.classifier)

            B = images.shape[0]
            output_h, output_w = hm.shape[2], hm.shape[3]

            for b in range(B):
                img_id = batch["image_id"][b]
                c = batch["center"][b].cpu().numpy()
                s = batch["scale"][b].item()
                orig_w = batch["orig_size"][b][0].item()
                orig_h = batch["orig_size"][b][1].item()

                # Filter by confidence threshold
                mask = scores[b] > config.CONF_THRESHOLD
                valid_xs = xs[b][mask]
                valid_ys = ys[b][mask]
                valid_cls_ids = cls_ids[b][mask]

                # Transform predictions to original image space
                if len(valid_xs) > 0:
                    coords = torch.stack([valid_xs, valid_ys], dim=1).cpu().numpy()
                    trans_coords = utils.transform_preds(
                        coords, c, s, (output_w, output_h)
                    )
                else:
                    trans_coords = np.empty((0, 2))

                valid_cls_ids = valid_cls_ids.cpu().numpy()

                # Retrieve Ground Truth
                gt_items = gt_map.get(img_id, [])

                # Matching Logic
                matched_gt_indices = set()
                img_tp = 0
                img_fp = 0

                # Iterate through all predictions
                for i in range(len(trans_coords)):
                    pred_point = trans_coords[i]
                    pred_cls_idx = valid_cls_ids[i]

                    if pred_cls_idx not in idx_to_class:
                        img_fp += 1
                        continue

                    pred_char = idx_to_class[pred_cls_idx]

                    match_found = False

                    # Greedy match against available GTs
                    for gt_idx, gt_item in enumerate(gt_items):
                        if gt_idx in matched_gt_indices:
                            continue

                        # Check Label Match
                        if gt_item["char"] == pred_char:
                            # Check Spatial Match (Point inside Box)
                            if calculate_iou_point_box(pred_point, gt_item["bbox"]):
                                matched_gt_indices.add(gt_idx)
                                match_found = True
                                break

                    if match_found:
                        img_tp += 1
                    else:
                        img_fp += 1

                img_fn = len(gt_items) - len(matched_gt_indices)

                # Update Global Counters
                tp_global += img_tp
                fp_global += img_fp
                fn_global += img_fn

                # Per-image stats for failure analysis
                precision = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0
                recall = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                image_stats.append(
                    {
                        "image_id": img_id,
                        "num_gt": len(gt_items),
                        "num_pred": len(trans_coords),
                        "tp": img_tp,
                        "fp": img_fp,
                        "fn": img_fn,
                        "f1": f1,
                        "width": orig_w,
                        "height": orig_h,
                    }
                )

    # Calculate Global F1 Score
    precision_global = (
        tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0
    )
    recall_global = (
        tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0
    )
    f1_global = (
        2 * precision_global * recall_global / (precision_global + recall_global)
        if (precision_global + recall_global) > 0
        else 0
    )

    print(f"Final Validation Metric: {f1_global}")

    return pd.DataFrame(image_stats), f1_global


def run_failure_analysis(df_stats):
    """
    Analyzes the validation results to find correlations between error and input features.
    """
    print("\n--- Failure Analysis ---")
    if len(df_stats) == 0:
        print("No stats available for failure analysis.")
        return

    # Calculate Error Magnitude (1 - F1)
    df_stats["error_magnitude"] = 1.0 - df_stats["f1"]

    # Features to correlate
    features = ["num_gt", "width", "height"]

    print("Correlation between Error Magnitude (1 - F1) and features:")
    for feat in features:
        if df_stats[feat].std() == 0:
            print(f"  {feat}: N/A (No variance)")
            continue

        corr, _ = pearsonr(df_stats["error_magnitude"], df_stats[feat])
        print(f"  {feat}: {corr:.4f}")

    # Additional Insights
    print("\nSummary Statistics:")
    print(f"  Average F1: {df_stats['f1'].mean():.4f}")
    print(
        f"  Worst Performing Image ID: {df_stats.loc[df_stats['f1'].idxmin()]['image_id']}"
    )
    print(f"  Images with 0 F1 Score: {(df_stats['f1'] == 0).sum()}")


def main():
    # 1. Setup
    print("Setting up environment...")
    utils.setup_directories()
    utils.seed_everything(config.SEED)

    # 2. Training
    # Initialize Trainer
    # We use the provided Trainer class which handles DataLoaders and Model init
    print("Initializing Trainer...")
    t = trainer.Trainer(debug=False, load_cached_data=True)

    # Run Training
    # Using config.NUM_EPOCHS (30) to allow convergence for the larger model
    print("Starting training loop...")
    t.fit(num_epochs=config.NUM_EPOCHS)

    # 3. Validation & Metric
    # We use the trained model from the trainer instance
    df_stats, val_f1 = evaluate_and_analyze(t.model, t.val_dataset, t.device)

    # 4. Failure Analysis
    run_failure_analysis(df_stats)

    # 5. Submission
    # Generate submission for the test set
    if val_f1 > 0.3771343611017969:
        print(
            f"\nValidation F1 ({val_f1:.4f}) exceeds threshold. Generating submission file..."
        )
        # Passing the save path ensures we use the best checkpoint saved during training
        inference.generate_submission(checkpoint_path=t.save_path)
    else:
        print(
            f"\nValidation F1 ({val_f1:.4f}) did not exceed threshold. Skipping submission."
        )

    print("Process completed successfully.")


if __name__ == "__main__":
    main()
