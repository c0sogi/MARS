import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
import scipy.stats

# Import from provided libraries
from library.config import Config, seed_everything
from library.train import run_training
from library.predict import predict
from library.dataset import SIIMDataset, get_transforms
from library.model import MultiTaskUNet
from library.utils import mask2boxes, STUDY_CLASSES
from library.loss import MultiTaskLoss


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_ap_opacity(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compute Average Precision for Opacity class using IoU matching.
    pred_boxes: list of [conf, x1, y1, x2, y2]
    gt_boxes: list of [x1, y1, x2, y2]
    """
    if not pred_boxes and not gt_boxes:
        return 1.0
    if not pred_boxes:
        return 0.0
    if not gt_boxes:
        return 0.0

    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = [False] * len(gt_boxes)

    for i, p_box in enumerate(pred_boxes):
        p_coords = p_box[1:]
        best_iou = 0.0
        best_gt_idx = -1

        for j, g_box in enumerate(gt_boxes):
            iou = calculate_iou(p_coords, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou > iou_threshold:
            if not gt_matched[best_gt_idx]:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # VOC-style AP (Area under PR curve)
    # Append sentinels
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def evaluate_validation(model, val_loader, device):
    """
    Runs inference on validation set and computes mAP and per-sample losses.
    """
    model.eval()
    criterion = MultiTaskLoss()

    # Storage for AP calculation
    all_study_preds = []
    all_study_targets = []

    all_opacity_preds = []  # List of list of boxes per image
    all_opacity_targets = []  # List of list of boxes per image

    # Storage for Failure Analysis
    sample_losses = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            cls_targets = batch["label"].to(device)
            mask_targets = batch["mask"].to(device)

            # Forward
            cls_logits, mask_logits = model(images)

            # Loss for failure analysis
            loss, metrics = criterion(
                cls_logits, mask_logits, cls_targets, mask_targets
            )
            # We approximate per-sample loss by averaging batch loss (limitation of provided loss fn)
            # Ideally we'd want reduction='none', but we use the provided library.
            # We will assign the batch average loss to all samples in batch for correlation analysis.
            batch_loss = metrics["loss_total"]

            # Predictions
            cls_probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
            mask_probs = torch.sigmoid(mask_logits).cpu().numpy()

            # Targets
            cls_targets_np = cls_targets.cpu().numpy()
            mask_targets_np = mask_targets.cpu().numpy()

            bs = images.size(0)
            for i in range(bs):
                # Study
                all_study_preds.append(cls_probs[i])
                all_study_targets.append(cls_targets_np[i])

                # Opacity
                # Extract pred boxes
                p_boxes = mask2boxes(
                    mask_probs[i, 0], threshold=0.5
                )  # [conf, x1, y1, x2, y2]
                all_opacity_preds.append(p_boxes)

                # Extract GT boxes from mask
                g_boxes = mask2boxes(mask_targets_np[i, 0], threshold=0.5)
                # mask2boxes returns [conf, ...], for GT conf is 1.0. We just need coords.
                g_boxes_coords = [b[1:] for b in g_boxes]
                all_opacity_targets.append(g_boxes_coords)

                # Metadata for failure analysis
                sample_losses.append(batch_loss)

    # 1. Study Level AP (Classification)
    all_study_preds = np.array(all_study_preds)
    all_study_targets = np.array(all_study_targets)

    study_aps = []
    for i, class_name in enumerate(STUDY_CLASSES):
        ap = average_precision_score(all_study_targets[:, i], all_study_preds[:, i])
        study_aps.append(ap)

    # 2. Opacity Level AP (Detection)
    # We compute AP over the entire dataset
    # Flatten list of boxes for global sorting? No, AP is computed per class over dataset.
    # We need to collect all pred boxes and all gt boxes with image IDs to match.
    # Simplified: We use the function defined above which expects lists.
    # Actually, standard mAP requires sorting ALL predictions across the dataset.

    # Flatten preds: [conf, x1, y1, x2, y2, image_idx]
    flat_preds = []
    for idx, boxes in enumerate(all_opacity_preds):
        for b in boxes:
            flat_preds.append(b + [idx])  # Append image index

    # Flatten GT: dict mapping image_idx -> list of boxes
    gt_map = {idx: boxes for idx, boxes in enumerate(all_opacity_targets)}

    # Compute Opacity AP
    # Sort all preds by confidence
    flat_preds.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(flat_preds))
    fp = np.zeros(len(flat_preds))

    # Track which GT boxes have been matched
    gt_matched_map = {idx: [False] * len(boxes) for idx, boxes in gt_map.items()}
    total_gt_boxes = sum(len(b) for b in gt_map.values())

    if total_gt_boxes == 0:
        opacity_ap = 0.0 if len(flat_preds) > 0 else 1.0
    else:
        for i, p_arr in enumerate(flat_preds):
            conf = p_arr[0]
            p_box = p_arr[1:5]
            img_idx = p_arr[5]

            gt_boxes = gt_map[img_idx]

            best_iou = 0.0
            best_gt_idx = -1

            for j, g_box in enumerate(gt_boxes):
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou > 0.5:
                if not gt_matched_map[img_idx][best_gt_idx]:
                    tp[i] = 1
                    gt_matched_map[img_idx][best_gt_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / total_gt_boxes
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        opacity_ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    # Final Metric: Mean of (Study APs + Opacity AP)
    # Note: This gives equal weight to each study class and the opacity class.
    all_aps = study_aps + [opacity_ap]
    final_map = np.mean(all_aps)

    return final_map, sample_losses


def perform_failure_analysis(val_df, sample_losses):
    """
    Correlates loss with metadata.
    """
    print("\n==== Failure Analysis ====")

    # We need original dimensions from file paths to correlate with image size
    # Since we don't have them in val_df directly (only file paths), we can try to read them
    # or just use the fact that we might have them in metadata if we generated it fully.
    # The provided metadata script doesn't save width/height in csv.
    # However, we can check if there's a correlation with 'number of boxes' which is in 'boxes' column.

    import ast

    def get_num_boxes(x):
        try:
            return len(ast.literal_eval(x))
        except:
            return 0

    val_df["num_boxes"] = val_df["boxes"].apply(get_num_boxes)

    # Ensure lengths match (drop last batch if dropped in loader, but we used drop_last=False for val)
    # The val_loader in 'run_training' uses drop_last=False.
    # However, we re-created the loader here.

    if len(sample_losses) != len(val_df):
        print(
            f"Warning: Loss count ({len(sample_losses)}) != DF count ({len(val_df)}). Truncating to min."
        )
        n = min(len(sample_losses), len(val_df))
        sample_losses = sample_losses[:n]
        val_df = val_df.iloc[:n]

    # Correlation with Num Boxes
    corr_boxes, _ = scipy.stats.pearsonr(val_df["num_boxes"], sample_losses)
    print(f"Correlation (Loss vs Num Boxes): {corr_boxes:.4f}")

    # We can't easily get width/height without reading files, which takes time.
    # We will skip width/height correlation to keep it fast, or check if 'Typical' class has higher loss.

    for col in STUDY_CLASSES:
        # Map column name
        col_name = {
            "negative": "Negative for Pneumonia",
            "typical": "Typical Appearance",
            "indeterminate": "Indeterminate Appearance",
            "atypical": "Atypical Appearance",
        }.get(col)

        if col_name in val_df.columns:
            corr_cls, _ = scipy.stats.pearsonr(val_df[col_name], sample_losses)
            print(f"Correlation (Loss vs {col_name}): {corr_cls:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Train
    # Extended training: 15 epochs (Cite solution_lesson_node_00002)
    print("Starting Training...")
    run_training(debug=False, epochs=15, batch_size=16)

    # 3. Validation & Metric
    print("\nStarting Validation & Metric Calculation...")

    # Load Data
    val_df = pd.read_csv(Config.VAL_CSV)
    val_dataset = SIIMDataset(
        df=val_df, split="val", transform=get_transforms("val"), load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load Model
    model = MultiTaskUNet(pretrained=False)
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        )
        print("Loaded best model for validation.")
    else:
        print("Error: Model checkpoint not found!")
        return

    model.to(device)

    # Evaluate
    final_map, sample_losses = evaluate_validation(model, val_loader, device)

    # Print Required Metric
    print(f"Final Validation Metric: {final_map:.10f}")

    # 4. Failure Analysis
    perform_failure_analysis(val_df, sample_losses)

    # 5. Prediction (Test Set)
    if final_map > 0.4729475001:
        print("\nStarting Inference on Test Set...")
        predict(debug=False, batch_size=16, load_cached_data=True)
    else:
        print(
            f"\nValidation metric {final_map:.10f} did not beat threshold 0.4729475001. Skipping inference."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
