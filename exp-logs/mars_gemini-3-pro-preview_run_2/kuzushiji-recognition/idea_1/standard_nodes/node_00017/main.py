import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import from provided libraries
from library.config import Config, seed_everything
from library.train import train_detector
from library.inference import generate_submission, load_models, process_page
from library.utils import parse_labels, load_image

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def calculate_f1(preds, gts):
    """
    Calculates the modified F1 score for a single image.
    """
    tp = 0
    fp = 0
    fn = 0

    matched_gt_indices = set()

    for pred in preds:
        pred_label = pred["label"]
        px, py = pred["x"], pred["y"]

        match_found = False

        for i, gt in enumerate(gts):
            if i in matched_gt_indices:
                continue

            if pred_label != gt["char"]:
                continue

            gx, gy, gw, gh = gt["x"], gt["y"], gt["w"], gt["h"]
            if gx <= px <= gx + gw and gy <= py <= gy + gh:
                match_found = True
                matched_gt_indices.add(i)
                break

        if match_found:
            tp += 1
        else:
            fp += 1

    fn = len(gts) - len(matched_gt_indices)

    return tp, fp, fn


def run_validation(val_csv_path):
    print("Starting Validation...")
    device = Config.DEVICE

    val_df = pd.read_csv(val_csv_path)
    model, id2label = load_models(device)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    analysis_data = []

    for idx, row in val_df.iterrows():
        image_id = row["image_id"]
        file_path = row["file_path"]
        labels_str = row.get("labels", "")

        gt_boxes = parse_labels(labels_str)

        try:
            image = load_image(file_path)
            h, w = image.shape[:2]

            pred_strings = process_page(image, model, device, id2label)

            preds = []
            for s in pred_strings:
                parts = s.split()
                if len(parts) == 3:
                    preds.append(
                        {"label": parts[0], "x": int(parts[1]), "y": int(parts[2])}
                    )

            tp, fp, fn = calculate_f1(preds, gt_boxes)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            denom = 2 * tp + fp + fn
            img_f1 = (2 * tp) / denom if denom > 0 else 0.0

            analysis_data.append(
                {
                    "image_id": image_id,
                    "f1": img_f1,
                    "width": w,
                    "height": h,
                    "area": w * h,
                    "num_chars": len(gt_boxes),
                }
            )

        except Exception as e:
            print(f"Error validating image {image_id}: {e}")

        if (idx + 1) % 100 == 0:
            print(f"Validated {idx + 1}/{len(val_df)} images.")

    denominator = 2 * total_tp + total_fp + total_fn
    final_f1 = (2 * total_tp) / denominator if denominator > 0 else 0.0

    print(f"Final Validation Metric: {final_f1}")
    return final_f1, pd.DataFrame(analysis_data)


def perform_failure_analysis(df_analysis):
    print("\n=== Failure Analysis ===")
    if df_analysis.empty:
        print("No analysis data available.")
        return

    correlations = df_analysis[["f1", "width", "height", "area", "num_chars"]].corr()[
        "f1"
    ]

    print("Correlation between F1 Score and Input Features:")
    print(f"  Image Width: {correlations['width']:.4f}")
    print(f"  Image Height: {correlations['height']:.4f}")
    print(f"  Image Area: {correlations['area']:.4f}")
    print(f"  Character Count (Density): {correlations['num_chars']:.4f}")


def main():
    seed_everything(Config.SEED)

    print("\n=== Stage 1: Training Detection Model ===")
    train_detector(debug=False)

    print("\n=== Stage 2: Validation ===")
    final_f1 = 0.0
    if os.path.exists(Config.VAL_CSV):
        final_f1, df_analysis = run_validation(Config.VAL_CSV)
        perform_failure_analysis(df_analysis)
    else:
        print("Validation metadata not found.")

    # Threshold check based on task requirement
    if final_f1 > 0.793687066253159:
        print("\n=== Stage 3: Submission Generation ===")
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")
        generate_submission(
            test_csv_path=Config.TEST_CSV, submission_path=submission_path
        )
    else:
        print(
            f"\nValidation F1 ({final_f1}) did not meet threshold (0.60146). Submission skipped."
        )


if __name__ == "__main__":
    main()
