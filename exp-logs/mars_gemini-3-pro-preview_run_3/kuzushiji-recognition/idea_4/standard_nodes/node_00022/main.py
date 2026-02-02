import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed
from library.trainer import DetectorTrainer, ClassifierTrainer
from library.inference import InferencePipeline


def parse_labels_gt(label_str):
    """Parses GT label string into list of dicts."""
    if not label_str or not isinstance(label_str, str):
        return []
    parts = label_str.split()
    items = []
    for i in range(0, len(parts), 5):
        try:
            label = parts[i]
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            w = int(parts[i + 3])
            h = int(parts[i + 4])
            items.append({"label": label, "box": [x, y, w, h], "matched": False})
        except (ValueError, IndexError):
            continue
    return items


def parse_labels_pred(label_str):
    """Parses Prediction label string into list of dicts."""
    if not label_str or not isinstance(label_str, str):
        return []
    parts = label_str.split()
    items = []
    for i in range(0, len(parts), 3):
        try:
            label = parts[i]
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            items.append({"label": label, "point": [x, y]})
        except (ValueError, IndexError):
            continue
    return items


def is_point_in_box(point, box):
    px, py = point
    bx, by, bw, bh = box
    return (bx <= px <= bx + bw) and (by <= py <= by + bh)


def calculate_metric(gt_df, pred_df):
    """
    Calculates the modified F1 score.
    Returns global F1 and per-image stats.
    """
    # Create dictionary for fast lookup
    pred_map = dict(zip(pred_df["image_id"], pred_df["labels"]))

    tp_global = 0
    fp_global = 0
    fn_global = 0

    image_stats = []

    for idx, row in gt_df.iterrows():
        img_id = row["image_id"]
        gt_str = row["labels"]
        pred_str = pred_map.get(img_id, "")

        gt_items = parse_labels_gt(gt_str)
        pred_items = parse_labels_pred(pred_str)

        tp = 0
        fp = 0

        # Greedy matching
        for p in pred_items:
            match_found = False
            for g in gt_items:
                if not g["matched"] and g["label"] == p["label"]:
                    if is_point_in_box(p["point"], g["box"]):
                        g["matched"] = True
                        match_found = True
                        break
            if match_found:
                tp += 1
            else:
                fp += 1

        fn = sum(1 for g in gt_items if not g["matched"])

        tp_global += tp
        fp_global += fp
        fn_global += fn

        # Per image F1
        denom = 2 * tp + fp + fn
        f1_img = (2 * tp) / denom if denom > 0 else 0.0
        if denom == 0 and len(gt_items) == 0 and len(pred_items) == 0:
            f1_img = 1.0  # Perfect empty prediction for empty page

        image_stats.append(
            {
                "image_id": img_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": f1_img,
                "num_chars": len(gt_items),
            }
        )

    denom_global = 2 * tp_global + fp_global + fn_global
    f1_global = (2 * tp_global) / denom_global if denom_global > 0 else 0.0

    return f1_global, pd.DataFrame(image_stats)


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Configure epochs for optimized performance within 4 hour limit
    # Cite solution_lesson_node_00012: Extended training for convergence
    Config.DETECTOR_EPOCHS = 20
    Config.CLASSIFIER_EPOCHS = 10

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print("=== Starting Run Pipeline ===")

    # 2. Train Detector
    print("\n[Step 1/5] Training Detector...")
    det_trainer = DetectorTrainer()
    det_trainer.fit()

    # Clean up to save memory
    del det_trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Train Classifier
    print("\n[Step 2/5] Training Classifier...")
    cls_trainer = ClassifierTrainer()
    cls_trainer.fit()

    # Clean up
    del cls_trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Validation Inference
    print("\n[Step 3/5] Running Validation Inference...")
    val_pred_path = os.path.join(Config.WORKING_DIR, "val_predictions.csv")

    # We reuse InferencePipeline but point it to validation metadata
    # Note: InferencePipeline loads the best checkpoints saved during training
    pipeline = InferencePipeline()
    pipeline.run(test_metadata_path=Config.VAL_METADATA_PATH, output_path=val_pred_path)

    # 5. Metric Calculation & Failure Analysis
    print("\n[Step 4/5] Evaluating Performance...")
    val_gt_df = pd.read_csv(Config.VAL_METADATA_PATH, keep_default_na=False)
    val_pred_df = pd.read_csv(val_pred_path, keep_default_na=False)

    f1_score, stats_df = calculate_metric(val_gt_df, val_pred_df)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {f1_score}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")

    stats_df["error_magnitude"] = 1.0 - stats_df["f1"]

    # Correlation with Number of Characters (Density/Complexity)
    if len(stats_df) > 1 and stats_df["num_chars"].std() > 0:
        corr_chars, _ = pearsonr(stats_df["num_chars"], stats_df["error_magnitude"])
        print(f"Correlation (Num Characters vs Error): {corr_chars:.4f}")

    # Correlation with False Positives (Hallucination tendency)
    if len(stats_df) > 1 and stats_df["fp"].std() > 0:
        corr_fp, _ = pearsonr(stats_df["fp"], stats_df["error_magnitude"])
        print(f"Correlation (False Positives vs Error): {corr_fp:.4f}")

    # Correlation with False Negatives (Miss rate)
    if len(stats_df) > 1 and stats_df["fn"].std() > 0:
        corr_fn, _ = pearsonr(stats_df["fn"], stats_df["error_magnitude"])
        print(f"Correlation (False Negatives vs Error): {corr_fn:.4f}")

    # 6. Submission
    print("\n[Step 5/5] Checking Submission Criteria...")
    threshold = 0.8455090517492287

    if f1_score > threshold:
        print(f"Validation score {f1_score} > {threshold}. Generating submission...")
        # Clear pipeline to free memory (though it's same model, good practice)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        # Re-init pipeline for test set
        pipeline = InferencePipeline()
        pipeline.run(
            test_metadata_path=Config.TEST_METADATA_PATH,
            output_path=Config.SUBMISSION_PATH,
        )
        print("Submission generation complete.")
    else:
        print(f"Validation score {f1_score} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
