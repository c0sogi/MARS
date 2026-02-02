import os
import sys
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.dataset import ChestXrayDataset, get_transforms
from library.model import AntiAliasedResNetUNet
from library.utils import seed_everything, calculate_map, calculate_classification_ap

# Suppress warnings
warnings.filterwarnings("ignore")


def run_fast_baseline():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # Override Config attributes directly before they are used by other classes
    Config.NUM_EPOCHS = 6
    Config.BATCH_SIZE = 32  # Increase batch size for A100
    Config.DEBUG = False  # Use full dataset to ensure we beat the threshold

    print(f"Starting Fast Baseline Run...")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    trainer = Trainer()
    trainer.fit()

    # -------------------------------------------------------------------------
    # 3. Final Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Final Validation and Failure Analysis...")

    # Load Best Model
    device = torch.device(Config.DEVICE)
    model = AntiAliasedResNetUNet().to(device)

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    # Load EMA state dict if available, else standard state dict
    if checkpoint.get("ema_state_dict"):
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("Loaded Best EMA Model.")
    else:
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded Best Model.")

    model.eval()

    # Setup Validation Loader
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_ds = ChestXrayDataset(df_val, mode="val", transform=get_transforms("val"))

    # Custom collate is needed, but we can reuse the one from trainer logic or simple list
    # We'll re-instantiate the loader to be safe
    from library.trainer import collate_fn

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Storage for Metrics and Analysis
    all_study_targets = []
    all_study_probs = []

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    # For Failure Analysis
    sample_errors = []
    gt_num_boxes = []
    gt_box_areas = []

    criterion_cls = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            study_labels = batch["study_label"].to(device)
            gt_boxes_batch = batch["boxes"]
            gt_labels_batch = batch["box_labels"]

            # Forward
            cls_logits, seg_logits = model(images)

            # --- Study Metrics ---
            probs = torch.softmax(cls_logits, dim=1)
            all_study_targets.append(study_labels.cpu())
            all_study_probs.append(probs.cpu())

            # --- Failure Analysis Data Collection ---
            # Calculate CrossEntropy error per sample
            targets_idx = torch.argmax(study_labels, dim=1)
            losses = criterion_cls(cls_logits, targets_idx)
            sample_errors.extend(losses.cpu().numpy())

            # Collect GT features
            for i in range(len(images)):
                boxes = gt_boxes_batch[i]
                gt_num_boxes.append(len(boxes))
                if len(boxes) > 0:
                    # Area = (x2-x1)*(y2-y1)
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    gt_box_areas.append(areas.sum().item())
                else:
                    gt_box_areas.append(0.0)

            # --- Image Metrics (Box Extraction) ---
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
            pred_classes = torch.argmax(probs, dim=1).cpu()

            for i in range(len(images)):
                all_gt_boxes.append(gt_boxes_batch[i])
                all_gt_labels.append(gt_labels_batch[i])

                # Gating: If Negative (class 0), force none
                if Config.GATED_PREDICTION and pred_classes[i].item() == 0:
                    all_pred_boxes.append(torch.tensor([], dtype=torch.float32))
                    all_pred_scores.append(torch.tensor([], dtype=torch.float32))
                    all_pred_labels.append(torch.tensor([], dtype=torch.int64))
                    continue

                # Extract boxes
                mask = seg_probs[i, 0]
                binary_mask = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                boxes = []
                scores = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h < 50:
                        continue
                    boxes.append([x, y, x + w, y + h])
                    scores.append(np.mean(mask[y : y + h, x : x + w]))

                if boxes:
                    all_pred_boxes.append(torch.tensor(boxes, dtype=torch.float32))
                    all_pred_scores.append(torch.tensor(scores, dtype=torch.float32))
                    all_pred_labels.append(torch.zeros(len(boxes), dtype=torch.int64))
                else:
                    all_pred_boxes.append(torch.tensor([], dtype=torch.float32))
                    all_pred_scores.append(torch.tensor([], dtype=torch.float32))
                    all_pred_labels.append(torch.tensor([], dtype=torch.int64))

    # Calculate Metrics
    all_study_targets_np = torch.cat(all_study_targets, dim=0).numpy()
    all_study_probs_np = torch.cat(all_study_probs, dim=0).numpy()

    study_map = np.mean(
        calculate_classification_ap(all_study_targets_np, all_study_probs_np)
    )
    image_map = calculate_map(
        all_pred_boxes,
        all_pred_scores,
        all_pred_labels,
        all_gt_boxes,
        all_gt_labels,
        num_classes=1,
        iou_threshold=0.5,
    )

    composite_score = (study_map + image_map) / 2.0

    # REQUIRED PRINT
    print(f"Final Validation Metric: {composite_score}")

    # Failure Analysis Output
    print("\n--- Failure Analysis ---")
    corr_boxes, _ = pearsonr(sample_errors, gt_num_boxes)
    corr_area, _ = pearsonr(sample_errors, gt_box_areas)
    print(f"Correlation (Error vs Num Boxes): {corr_boxes:.4f}")
    print(f"Correlation (Error vs Opacity Area): {corr_area:.4f}")

    # -------------------------------------------------------------------------
    # 4. Inference & Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.49944536565378

    if composite_score > THRESHOLD:
        print(
            f"\nMetric ({composite_score:.6f}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nMetric ({composite_score:.6f}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


def generate_submission(model, device):
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    test_ds = ChestXrayDataset(df_test, mode="test", transform=get_transforms("val"))

    # No collate needed for test as we process one by one or standard batch (no boxes)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    # TTA Transform (Horizontal Flip)
    # We'll implement TTA manually on the tensor

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            study_ids = batch["study_id"]
            image_ids = batch["image_id"]

            # 1. Forward Pass (Original)
            cls_logits, seg_logits = model(images)
            probs = torch.softmax(cls_logits, dim=1)
            seg_probs = torch.sigmoid(seg_logits)

            if Config.TTA_FLIP:
                # 2. Forward Pass (Flipped)
                images_flipped = torch.flip(images, dims=[3])
                cls_logits_f, seg_logits_f = model(images_flipped)
                probs_f = torch.softmax(cls_logits_f, dim=1)
                seg_probs_f = torch.sigmoid(seg_logits_f)

                # Flip mask back
                seg_probs_f = torch.flip(seg_probs_f, dims=[3])

                # Average
                probs = (probs + probs_f) / 2.0
                seg_probs = (seg_probs + seg_probs_f) / 2.0

            # Process Batch
            probs_np = probs.cpu().numpy()
            seg_probs_np = seg_probs.cpu().numpy()

            for i in range(len(images)):
                s_id = study_ids[i]
                i_id = image_ids[i]

                # --- Study Prediction ---
                # Format: "class conf 0 0 1 1" for all classes or just argmax?
                # Task: "For each study... predict at least one... format... class ID... confidence... 0 0 1 1"
                # We will output all 4 classes to be safe and maximize AP
                study_pred_strs = []
                for idx, label in enumerate(Config.STUDY_LABELS):
                    # Clean label string for submission (e.g. "Negative for Pneumonia" -> "negative")
                    # The sample submission uses: 'negative', 'typical', 'indeterminate', 'atypical'
                    # The dataset columns are: 'Negative for Pneumonia', 'Typical Appearance', ...
                    # Mapping based on first word usually works or specific mapping
                    short_label = label.split()[0].lower()
                    if short_label == "negative":
                        short_label = "negative"
                    elif short_label == "typical":
                        short_label = "typical"
                    elif short_label == "indeterminate":
                        short_label = "indeterminate"
                    elif short_label == "atypical":
                        short_label = "atypical"

                    conf = probs_np[i, idx]
                    study_pred_strs.append(f"{short_label} {conf:.6f} 0 0 1 1")

                study_str = " ".join(study_pred_strs)
                results.append({"id": f"{s_id}_study", "PredictionString": study_str})

                # --- Image Prediction ---
                # Gating
                pred_class_idx = np.argmax(probs_np[i])

                if Config.GATED_PREDICTION and pred_class_idx == 0:
                    # Negative
                    image_str = "none 1 0 0 1 1"
                else:
                    # Extract Boxes
                    mask = seg_probs_np[i, 0]
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    box_preds = []
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w * h < 50:
                            continue

                        # Scale boxes back to original size?
                        # The competition usually expects predictions in original image coordinates.
                        # However, the provided metadata/dataset logic handles resizing.
                        # We need to check if we have original dims.
                        # For the test set, we didn't load original dims in the loop.
                        # But wait, the sample submission format implies we just output.
                        # If we resized input to 512, we must resize boxes back to original.

                        # We need to read the original dimensions from the test metadata/files or cache.
                        # The Dataset class caches dims in `test_dims.parquet`.
                        # We can access it via the dataset object if we stored it, or read the parquet.

                        # Let's rely on the fact that `ChestXrayDataset` saves `test_dims.parquet`.
                        # We should load it once.
                        pass

                    # To properly scale, we need access to the scale factors.
                    # Since we are inside a loop, reading parquet every time is slow.
                    # We will assume we output in 512x512 and the evaluator handles it OR
                    # (Correct way): We must rescale.

                    # Let's get the scale factors for this image.
                    # We can get them from the dataset if we modify it, or read the parquet file into a DF before loop.
                    pass

    # Re-implementing loop with scaling support
    # Load dimensions
    dims_path = os.path.join(Config.WORKING_DIR, "test_dims.parquet")
    if os.path.exists(dims_path):
        dims_df = pd.read_parquet(dims_path)
        # We assume the order matches the dataset (dataset uses metadata_df which matches dims_df rows)
    else:
        # Fallback (should not happen if dataset initialized)
        dims_df = None

    results = []

    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            images = batch["image"].to(device)
            study_ids = batch["study_id"]
            image_ids = batch["image_id"]

            # TTA Forward
            cls_logits, seg_logits = model(images)
            probs = torch.softmax(cls_logits, dim=1)
            seg_probs = torch.sigmoid(seg_logits)

            if Config.TTA_FLIP:
                images_f = torch.flip(images, dims=[3])
                cls_f, seg_f = model(images_f)
                probs_f = torch.softmax(cls_f, dim=1)
                seg_f = torch.flip(torch.sigmoid(seg_f), dims=[3])
                probs = (probs + probs_f) / 2
                seg_probs = (seg_probs + seg_f) / 2

            probs_np = probs.cpu().numpy()
            seg_probs_np = seg_probs.cpu().numpy()

            for i in range(len(images)):
                global_idx = batch_idx * Config.BATCH_SIZE + i
                s_id = study_ids[i]
                i_id = image_ids[i]

                # Scaling info
                scale_w = 1.0
                scale_h = 1.0
                if dims_df is not None:
                    row = dims_df.iloc[global_idx]
                    scale_w = row["scale_w"]
                    scale_h = row["scale_h"]

                # 1. Study String
                study_parts = []
                for idx, label in enumerate(Config.STUDY_LABELS):
                    lbl = label.split()[0].lower()
                    conf = probs_np[i, idx]
                    study_parts.append(f"{lbl} {conf:.6f} 0 0 1 1")
                results.append(
                    {"id": f"{s_id}_study", "PredictionString": " ".join(study_parts)}
                )

                # 2. Image String
                pred_cls = np.argmax(probs_np[i])
                if Config.GATED_PREDICTION and pred_cls == 0:
                    results.append(
                        {"id": f"{i_id}_image", "PredictionString": "none 1 0 0 1 1"}
                    )
                else:
                    mask = seg_probs_np[i, 0]
                    binary = (mask > 0.5).astype(np.uint8)
                    cnts, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    box_parts = []
                    for c in cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        if w * h < 50:
                            continue

                        conf = np.mean(mask[y : y + h, x : x + w])

                        # Rescale to original
                        x_orig = x / scale_w
                        y_orig = y / scale_h
                        w_orig = w / scale_w
                        h_orig = h / scale_h

                        # Format: opacity conf xmin ymin xmax ymax
                        box_parts.append(
                            f"opacity {conf:.4f} {x_orig:.2f} {y_orig:.2f} {x_orig+w_orig:.2f} {y_orig+h_orig:.2f}"
                        )

                    if not box_parts:
                        results.append(
                            {
                                "id": f"{i_id}_image",
                                "PredictionString": "none 1 0 0 1 1",
                            }
                        )
                    else:
                        results.append(
                            {
                                "id": f"{i_id}_image",
                                "PredictionString": " ".join(box_parts),
                            }
                        )

            batch_idx += 1

    # Save
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    seed_everything(Config.SEED)
    run_fast_baseline()
