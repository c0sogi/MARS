import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats

# Import library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.engine as engine

# ==========================================
# 1. Configuration Patching & Setup
# ==========================================
# Patch config for fast baseline execution
config.NUM_EPOCHS = 5
config.WORKING_DIR = "./working/idea_5"
os.makedirs(config.WORKING_DIR, exist_ok=True)

# Set seeds
utils.seed_everything(config.SEED)


# ==========================================
# 2. Metric Calculation Utilities
# ==========================================
def compute_iou(box1, box2):
    """Computes IoU between two sets of boxes."""
    # box1: (N, 4), box2: (M, 4)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def calculate_ap_voc2010(rec, prec):
    """Calculates AP using VOC 2010 method (All-point interpolation)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta Recall) * Prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_map(model, data_loader, device):
    """
    Runs inference on validation set and computes mAP@0.5 for 'opacity'.
    Returns:
        mAP (float)
        per_image_stats (list): List of dicts with 'ap', 'num_gt', 'avg_area' for failure analysis.
    """
    model.eval()

    gt_boxes_list = []
    pred_boxes_list = []
    pred_scores_list = []

    # For Failure Analysis
    image_stats = []

    print("Running validation inference for metric calculation...")
    with torch.no_grad():
        for images, targets, image_ids in tqdm(data_loader, disable=True):
            images = list(img.to(device) for img in images)

            # Get predictions
            # Model returns list of dicts: [{'boxes':..., 'scores':..., 'labels':..., 'study_logits':...}]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Predictions
                p_boxes = output["boxes"].cpu()
                p_scores = output["scores"].cpu()

                # Ground Truth
                t_boxes = targets[i]["boxes"]

                # Store for global mAP
                gt_boxes_list.append(t_boxes)
                pred_boxes_list.append(p_boxes)
                pred_scores_list.append(p_scores)

                # --- Per Image AP Calculation for Failure Analysis ---
                # Calculate AP for this single image
                ap_img = 0.0
                if len(t_boxes) > 0:
                    if len(p_boxes) > 0:
                        iou = compute_iou(p_boxes, t_boxes)
                        # Greedily match predictions to GT
                        # Sort by score
                        sorted_indices = torch.argsort(p_scores, descending=True)
                        iou_sorted = iou[sorted_indices]

                        tp = np.zeros(len(p_boxes))
                        fp = np.zeros(len(p_boxes))
                        gt_matched = np.zeros(len(t_boxes), dtype=bool)

                        for k in range(len(p_boxes)):
                            max_iou, max_idx = torch.max(iou_sorted[k], dim=0)
                            if max_iou > 0.5:
                                if not gt_matched[max_idx]:
                                    tp[k] = 1.0
                                    gt_matched[max_idx] = True
                                else:
                                    fp[k] = 1.0
                            else:
                                fp[k] = 1.0

                        tp_cumsum = np.cumsum(tp)
                        fp_cumsum = np.cumsum(fp)
                        rec = tp_cumsum / len(t_boxes)
                        prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                        ap_img = calculate_ap_voc2010(rec, prec)
                    else:
                        ap_img = 0.0
                else:
                    # No GT boxes.
                    # If predictions exist -> FP -> AP=0.
                    # If no predictions -> TN -> AP=1 (Perfectly predicted empty).
                    if len(p_boxes) > 0:
                        ap_img = 0.0
                    else:
                        ap_img = 1.0

                # Metadata for correlation
                num_gt = len(t_boxes)
                avg_area = 0.0
                if num_gt > 0:
                    areas = (t_boxes[:, 2] - t_boxes[:, 0]) * (
                        t_boxes[:, 3] - t_boxes[:, 1]
                    )
                    avg_area = areas.mean().item()

                image_stats.append(
                    {
                        "image_id": image_ids[i],
                        "ap": ap_img,
                        "num_gt": num_gt,
                        "avg_area": avg_area,
                    }
                )

    # --- Global mAP Calculation ---
    # We treat all samples as one large batch for mAP calculation
    # Flatten everything
    all_scores = []
    all_tp = []
    all_fp = []
    total_gt = 0

    for i in range(len(gt_boxes_list)):
        t_boxes = gt_boxes_list[i]
        p_boxes = pred_boxes_list[i]
        p_scores = pred_scores_list[i]

        total_gt += len(t_boxes)

        if len(p_boxes) == 0:
            continue

        # Sort predictions by score
        sorted_ind = torch.argsort(p_scores, descending=True)
        p_boxes = p_boxes[sorted_ind]
        p_scores = p_scores[sorted_ind]

        all_scores.append(p_scores.numpy())

        tp = np.zeros(len(p_boxes))
        fp = np.zeros(len(p_boxes))

        if len(t_boxes) > 0:
            iou = compute_iou(p_boxes, t_boxes)
            gt_matched = np.zeros(len(t_boxes), dtype=bool)

            for k in range(len(p_boxes)):
                max_iou, max_idx = torch.max(iou[k], dim=0)
                if max_iou > 0.5:
                    if not gt_matched[max_idx]:
                        tp[k] = 1.0
                        gt_matched[max_idx] = True
                    else:
                        fp[k] = 1.0
                else:
                    fp[k] = 1.0
        else:
            fp[:] = 1.0

        all_tp.append(tp)
        all_fp.append(fp)

    if total_gt == 0:
        return 0.0, image_stats

    if len(all_scores) == 0:
        return 0.0, image_stats

    all_scores = np.concatenate(all_scores)
    all_tp = np.concatenate(all_tp)
    all_fp = np.concatenate(all_fp)

    # Sort globally by score
    sorted_ind = np.argsort(-all_scores)
    all_tp = all_tp[sorted_ind]
    all_fp = all_fp[sorted_ind]

    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(all_fp)

    rec = tp_cumsum / total_gt
    prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    mAP = calculate_ap_voc2010(rec, prec)

    return mAP, image_stats


# ==========================================
# 3. Main Execution Flow
# ==========================================
def main():
    # --- Step 1: Training ---
    print("=== Starting Training ===")
    # Engine.fit handles dataset loading, training loop, and saving best model
    engine.fit(load_cached_data=True, debug=config.DEBUG)

    # --- Step 2: Validation & Metric ---
    print("\n=== Starting Validation & Metric Calculation ===")
    device = config.DEVICE

    # Load Validation Data
    val_dataset = dataset.CovidDataset(
        mode="val",
        transforms=utils.get_valid_transforms(),
        load_cached_data=True,
        debug=config.DEBUG,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=utils.collate_fn,
    )

    # Load Best Model
    model = model_lib.get_model()
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded best model from {model_path}")
    else:
        print("Error: Best model not found. Using initialized weights.")

    model.to(device)

    # Compute Metric
    final_metric, image_stats = evaluate_map(model, val_loader, device)

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric}")

    # --- Step 3: Failure Analysis ---
    print("\n=== Failure Analysis ===")
    stats_df = pd.DataFrame(image_stats)
    stats_df["error_magnitude"] = 1.0 - stats_df["ap"]

    # Correlations
    if len(stats_df) > 1:
        corr_num_boxes, _ = scipy.stats.spearmanr(
            stats_df["error_magnitude"], stats_df["num_gt"]
        )
        corr_avg_area, _ = scipy.stats.spearmanr(
            stats_df["error_magnitude"], stats_df["avg_area"]
        )

        print(f"Correlation (Error vs Num Boxes): {corr_num_boxes:.4f}")
        print(f"Correlation (Error vs Avg Box Area): {corr_avg_area:.4f}")
    else:
        print("Insufficient data for failure analysis.")

    # --- Step 4: Submission ---
    THRESHOLD = 0.43290277912681663

    if final_metric > THRESHOLD:
        print("\n=== Generating Submission ===")

        # Load Test Data
        test_dataset = dataset.CovidDataset(
            mode="test", transforms=utils.get_valid_transforms(), load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=utils.collate_fn,
        )

        submission_rows = []

        model.eval()
        with torch.no_grad():
            for images, _, image_ids in tqdm(test_loader, disable=True):
                images = list(img.to(device) for img in images)
                outputs = model(images)

                for i, output in enumerate(outputs):
                    image_id = image_ids[i]
                    study_id = (
                        image_id.replace("_image", "") + "_study"
                    )  # Assuming mapping logic or ID structure
                    # Actually, test dataset has StudyInstanceUID in metadata, but dataset returns image_id.
                    # The sample submission expects:
                    # id, PredictionString
                    # study_id, ...
                    # image_id, ...

                    # We need to reconstruct study ID from image ID or use metadata.
                    # The test.csv metadata has image_id and StudyInstanceUID.
                    # Let's rely on the fact that we need to output rows for both.
                    # Wait, the submission format requires specific IDs.
                    # We will output rows for 'image_id' + '_image' (if not already suffixed) and 'study_id' + '_study'.
                    # The dataset returns 'image_id' which is the filename stem.

                    # Get predictions
                    p_boxes = output["boxes"].cpu().numpy()
                    p_scores = output["scores"].cpu().numpy()
                    p_labels = output["labels"].cpu().numpy()

                    # Study Prediction
                    # output['study_probs'] is (4,) tensor
                    study_probs = output["study_probs"].cpu().numpy()
                    study_label_idx = np.argmax(study_probs)
                    study_conf = study_probs[study_label_idx]
                    study_label_name = config.STUDY_ID_TO_LABEL[study_label_idx]

                    # Consistency Override
                    # If study is Negative, image should be None
                    if study_label_idx == 0:  # Negative
                        image_pred_str = "none 1 0 0 1 1"
                    else:
                        image_pred_str = utils.format_prediction_string(
                            p_boxes, p_scores, p_labels
                        )

                    # Format Study String
                    study_pred_str = utils.format_study_prediction_string(
                        study_label_name, study_conf
                    )

                    # We need to map image_id to study_id to output the study row correctly.
                    # Since we iterate by image, and multiple images might belong to one study,
                    # we should ideally aggregate. But for this competition, usually 1 image = 1 study in test?
                    # Or we just output the study prediction for *this* image's study ID.
                    # If multiple images per study, we might overwrite.
                    # For simplicity in this baseline, we assume 1-to-1 or just output.
                    # We need the study ID.
                    # Let's look up the study ID from the dataframe.
                    row = test_dataset.df[test_dataset.df["image_id"] == image_id].iloc[
                        0
                    ]
                    study_uid = row["StudyInstanceUID"]

                    # Append Study Row
                    submission_rows.append(
                        {"id": f"{study_uid}_study", "PredictionString": study_pred_str}
                    )

                    # Append Image Row
                    submission_rows.append(
                        {"id": f"{image_id}_image", "PredictionString": image_pred_str}
                    )

        # Create DataFrame
        sub_df = pd.DataFrame(submission_rows)
        # Remove duplicates (if multiple images per study produced multiple study rows)
        # We keep the first one or aggregate. Keeping first for baseline speed.
        sub_df = sub_df.drop_duplicates(subset=["id"])

        # Save
        os.makedirs("./submission", exist_ok=True)
        sub_df.to_csv("./submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")

    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
