import sys
import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config, seed_everything
from library.engine import Engine
from library.dataset import KuzushijiDataset
from library.utils import decode_outputs, get_affine_transform


def validate_and_analyze(engine, val_loader, val_ds):
    engine.model.eval()
    device = engine.device

    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Statistics for failure analysis
    # List of dicts: {'area': float, 'detected': int}
    gt_stats = []

    # Map image_id to dataset item for quick retrieval of GT and paths
    ds_map = {item["image_id"]: item for item in val_ds.data}

    print("Starting Validation...")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs = batch["input"].to(device)
            image_ids = batch["image_id"]

            # Forward pass
            outputs = engine.model(inputs)

            # Decode outputs (returns coordinates in 256x256 feature space)
            xs, ys, scores, cls_ids = decode_outputs(
                outputs["hm"], outputs["cls"], outputs["reg"], K=Config.TOP_K
            )

            batch_size = inputs.size(0)
            for i in range(batch_size):
                img_id = image_ids[i]
                item = ds_map[img_id]

                # Retrieve original image dimensions to compute inverse transform
                img_path = os.path.join(Config.INPUT_DIR, item["file_path"])

                # Read image to get shape (needed for correct coordinate mapping)
                img_temp = cv2.imread(img_path)
                if img_temp is None:
                    # Fallback if image read fails (should not happen)
                    h_orig, w_orig = Config.IMG_SIZE, Config.IMG_SIZE
                else:
                    h_orig, w_orig = img_temp.shape[:2]

                # Compute Inverse Affine Transform parameters
                c = np.array([w_orig / 2.0, h_orig / 2.0], dtype=np.float32)
                s = max(h_orig, w_orig) * 1.0

                # Filter predictions by confidence threshold
                valid_mask = scores[i] > Config.CONF_THRESHOLD

                pred_points = []
                pred_classes = []

                if valid_mask.sum() > 0:
                    v_xs = xs[i][valid_mask].cpu().numpy()
                    v_ys = ys[i][valid_mask].cpu().numpy()
                    v_cls = cls_ids[i][valid_mask].cpu().numpy()

                    # Map from 256x256 feature map to 1024x1024 input space
                    pts_input = np.stack([v_xs, v_ys], axis=1) * 4.0

                    # Map from 1024x1024 input space to Original Image space
                    trans_inv = get_affine_transform(
                        c, s, 0, [Config.IMG_SIZE, Config.IMG_SIZE], inv=True
                    )

                    # Apply affine transform
                    pts_homo = np.concatenate(
                        [pts_input, np.ones((pts_input.shape[0], 1))], axis=1
                    )
                    pts_orig = (trans_inv @ pts_homo.T).T  # (N, 2)

                    pred_points = pts_orig
                    pred_classes = v_cls

                # Retrieve Ground Truth
                # item['annotations'] format: [class_id, x, y, w, h]
                gt_anns = item["annotations"]

                # --- Matching Logic ---
                matched_gt_indices = set()

                # Iterate through predictions
                # Note: Predictions are already sorted by score from decode_outputs
                for p_idx in range(len(pred_points)):
                    p_x, p_y = pred_points[p_idx]
                    p_cls = pred_classes[p_idx]

                    match_found = False
                    # Check against all unmatched GTs
                    for g_idx, gt in enumerate(gt_anns):
                        if g_idx in matched_gt_indices:
                            continue

                        g_cls, g_x, g_y, g_w, g_h = gt

                        # Check Class Match
                        if int(g_cls) == int(p_cls):
                            # Check Point inside BBox
                            if (g_x <= p_x <= g_x + g_w) and (g_y <= p_y <= g_y + g_h):
                                matched_gt_indices.add(g_idx)
                                match_found = True
                                break

                    if match_found:
                        tp_total += 1
                    else:
                        fp_total += 1

                # Calculate False Negatives
                fn_count = len(gt_anns) - len(matched_gt_indices)
                fn_total += fn_count

                # --- Failure Analysis Data Collection ---
                for g_idx, gt in enumerate(gt_anns):
                    g_cls, g_x, g_y, g_w, g_h = gt
                    is_detected = 1 if g_idx in matched_gt_indices else 0
                    gt_stats.append(
                        {"area": float(g_w * g_h), "detected": int(is_detected)}
                    )

    # Compute F1 Score
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, pd.DataFrame(gt_stats)


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Configure Hyperparameters for Fast Baseline
    # A100 allows larger batch size. 15 epochs should be sufficient for convergence.
    Config.NUM_EPOCHS = 15
    Config.BATCH_SIZE = 32

    print(f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Initialize Engine
    engine = Engine()

    # 3. Train
    # Passing epochs explicitly to override default argument binding
    engine.fit(epochs=Config.NUM_EPOCHS)

    # 4. Validation & Metric
    # Initialize Validation Dataset and Loader
    val_ds = KuzushijiDataset(split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    f1, gt_analysis_df = validate_and_analyze(engine, val_loader, val_ds)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {f1}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    if not gt_analysis_df.empty and gt_analysis_df["area"].std() > 0:
        # Correlation between Box Area and Detection Success
        # Using Pearson correlation (equivalent to Point-Biserial for Binary-Continuous)
        corr = gt_analysis_df["detected"].corr(gt_analysis_df["area"])
        print(f"Correlation between Box Area and Detection Status: {corr:.6f}")

        # Additional stats
        detection_rate = gt_analysis_df["detected"].mean()
        print(f"Overall Detection Rate (Recall): {detection_rate:.4f}")
    else:
        print("Insufficient data for failure analysis.")

    # 6. Submission
    # Generates submission.csv in ./submission/
    engine.predict()


if __name__ == "__main__":
    main()
