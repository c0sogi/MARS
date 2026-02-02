import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import (
    LOSS_WEIGHTS,
    CLASS_LABELS,
    WORKING_DIR,
    SUBMISSION_PATH,
    TEST_META,
    DEVICE,
    NUM_CLASSES,
)
from library.utils import get_box_from_mask, calculate_map, format_prediction_string


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    Computes weighted loss: 1.0 * Class_Loss + 10.0 * Seg_Loss.
    """
    model.train()

    loss_meter = AverageMeter()
    class_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    # Define losses
    criterion_class = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Forward pass
        class_logits, seg_logits = model(images)

        # Calculate losses
        loss_class = criterion_class(class_logits, labels)
        loss_seg = criterion_seg(seg_logits, masks)

        # Weighted sum
        total_loss = (LOSS_WEIGHTS["class"] * loss_class) + (
            LOSS_WEIGHTS["seg"] * loss_seg
        )

        # Backward
        total_loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        class_loss_meter.update(loss_class.item(), batch_size)
        seg_loss_meter.update(loss_seg.item(), batch_size)

        pbar.set_postfix(
            {
                "Loss": f"{loss_meter.avg:.4f}",
                "Cls": f"{class_loss_meter.avg:.4f}",
                "Seg": f"{seg_loss_meter.avg:.4f}",
            }
        )

    return loss_meter.avg


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Calculates Loss and mAP (with study-level gating).
    """
    model.eval()

    loss_meter = AverageMeter()

    criterion_class = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    # Containers for mAP calculation
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []

    all_gt_boxes = []
    all_gt_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            # Forward
            class_logits, seg_logits = model(images)

            # Loss
            loss_class = criterion_class(class_logits, labels)
            loss_seg = criterion_seg(seg_logits, masks)
            total_loss = (LOSS_WEIGHTS["class"] * loss_class) + (
                LOSS_WEIGHTS["seg"] * loss_seg
            )
            loss_meter.update(total_loss.item(), images.size(0))

            # --- Process for mAP ---

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, [3])
            class_logits_flip, seg_logits_flip = model(images_flip)

            # 1. Study Prediction (Gating) - Average with TTA
            class_probs_orig = torch.softmax(class_logits, dim=1)
            class_probs_flip = torch.softmax(class_logits_flip, dim=1)
            class_probs = (class_probs_orig + class_probs_flip) / 2.0

            pred_classes = torch.argmax(class_probs, dim=1).cpu().numpy()

            # 2. Segmentation Prediction - Average with TTA
            seg_probs_orig = torch.sigmoid(seg_logits)
            seg_probs_flip = torch.sigmoid(seg_logits_flip)
            seg_probs_flip = torch.flip(seg_probs_flip, [3])  # Flip back
            seg_probs = (seg_probs_orig + seg_probs_flip) / 2.0

            # Iterate batch
            batch_size = images.size(0)
            for i in range(batch_size):
                # --- Ground Truth ---
                img_gt_boxes = []
                img_gt_labels = []

                # 1. Study Label GT (Classes 0-3)
                # Add 1-pixel box for the correct study label
                gt_label_idx = int(labels[i].item())
                img_gt_boxes.append([0, 0, 1, 1])
                img_gt_labels.append(gt_label_idx)

                # 2. Opacity GT (Class 4)
                gt_mask = masks[i].cpu().numpy()  # (1, H, W)
                gt_opacity_boxes = get_box_from_mask(gt_mask)

                for box in gt_opacity_boxes:
                    img_gt_boxes.append(box)
                    img_gt_labels.append(NUM_CLASSES)  # Class 4 is Opacity

                all_gt_boxes.append(img_gt_boxes)
                all_gt_labels.append(img_gt_labels)

                # --- Predictions ---
                img_pred_boxes = []
                img_pred_scores = []
                img_pred_labels = []

                # 1. Study Label Predictions (Classes 0-3)
                # Add 1-pixel box for each study class with its probability
                # Cite solution_lesson_node_00017: Composite metric alignment
                for c in range(NUM_CLASSES):
                    score = float(class_probs[i, c].item())
                    img_pred_boxes.append([0, 0, 1, 1])
                    img_pred_scores.append(score)
                    img_pred_labels.append(c)

                # 2. Opacity Predictions (Class 4)
                pred_cls = pred_classes[i]

                # Gating: If predicted "Negative for Pneumonia" (index 0), suppress opacity boxes
                if pred_cls != 0:
                    pred_mask = seg_probs[i].cpu().numpy()
                    pred_opacity_boxes = get_box_from_mask(pred_mask, threshold=0.5)

                    for box in pred_opacity_boxes:
                        xmin, ymin, xmax, ymax = box
                        region = pred_mask[0, ymin:ymax, xmin:xmax]
                        score = float(np.mean(region)) if region.size > 0 else 0.0

                        img_pred_boxes.append(box)
                        img_pred_scores.append(score)
                        img_pred_labels.append(NUM_CLASSES)  # Class 4 is Opacity

                all_pred_boxes.append(img_pred_boxes)
                all_pred_scores.append(img_pred_scores)
                all_pred_labels.append(img_pred_labels)

    # Calculate mAP at IoU 0.5
    map_score = calculate_map(
        all_pred_boxes,
        all_pred_scores,
        all_pred_labels,
        all_gt_boxes,
        all_gt_labels,
        iou_threshold=0.5,
    )

    return loss_meter.avg, map_score


def inference(model, dataloader, device):
    """
    Runs inference on the test set and generates the submission file.
    Aggregates study-level predictions and formats strings.
    """
    model.eval()

    # Load test metadata to map image_id to study_id
    if not os.path.exists(TEST_META):
        print("Test metadata not found. Skipping inference.")
        return

    test_df = pd.read_csv(TEST_META)
    # Map image_id -> study_id
    img_to_study = dict(zip(test_df["image_id"], test_df["study_id"]))

    results = []
    study_logits_map = {}  # study_id -> list of logits (to average if multiple images)

    print("Running Inference...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]

            # Original Forward
            class_logits, seg_logits = model(images)

            # TTA Forward (Horizontal Flip)
            images_flip = torch.flip(images, [3])
            class_logits_flip, seg_logits_flip = model(images_flip)

            # Average Predictions
            class_probs_orig = torch.softmax(class_logits, dim=1)
            class_probs_flip = torch.softmax(class_logits_flip, dim=1)
            class_probs = (class_probs_orig + class_probs_flip) / 2.0

            seg_probs_orig = torch.sigmoid(seg_logits)
            seg_probs_flip = torch.sigmoid(seg_logits_flip)
            seg_probs_flip = torch.flip(seg_probs_flip, [3])  # Flip back
            seg_probs = (seg_probs_orig + seg_probs_flip) / 2.0

            preds_cls = torch.argmax(class_probs, dim=1).cpu().numpy()

            for i, img_id in enumerate(image_ids):
                study_id = img_to_study.get(img_id, None)

                # --- Collect Study Logits ---
                if study_id:
                    if study_id not in study_logits_map:
                        study_logits_map[study_id] = []
                    study_logits_map[study_id].append(class_probs[i].cpu().numpy())

                # --- Generate Image Prediction ---
                # Gating
                pred_label_idx = preds_cls[i]

                if pred_label_idx == 0:
                    # Negative -> No findings
                    pred_string = "none 1 0 0 1 1"
                else:
                    # Positive -> Extract boxes
                    pred_mask = seg_probs[i].cpu().numpy()
                    boxes = get_box_from_mask(pred_mask, threshold=0.5)

                    if len(boxes) == 0:
                        pred_string = "none 1 0 0 1 1"
                    else:
                        # Calculate scores
                        scores = []
                        for box in boxes:
                            xmin, ymin, xmax, ymax = box
                            region = pred_mask[0, ymin:ymax, xmin:xmax]
                            scores.append(
                                float(np.mean(region)) if region.size > 0 else 0.0
                            )

                        pred_string = format_prediction_string(
                            ["opacity"] * len(boxes), scores, boxes
                        )

                results.append(
                    {"id": f"{img_id}_image", "PredictionString": pred_string}
                )

    # --- Generate Study Predictions ---
    for study_id, probs_list in study_logits_map.items():
        # Average probabilities across images in the study
        avg_probs = np.mean(probs_list, axis=0)
        best_idx = np.argmax(avg_probs)
        confidence = avg_probs[best_idx]

        # Map index to label string (e.g., "Negative for Pneumonia" -> "negative")
        label_raw = CLASS_LABELS[best_idx]
        label_name = label_raw.split(" ")[0].lower()

        pred_string = f"{label_name} {confidence:.6f} 0 0 1 1"

        results.append({"id": f"{study_id}_study", "PredictionString": pred_string})

    # Save submission
    submission_df = pd.DataFrame(results)
    # Ensure column order
    submission_df = submission_df[["id", "PredictionString"]]
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    device,
    num_epochs,
    patience=5,
):
    """
    Main training loop with early stopping and model checkpointing.
    """
    best_map = 0.0
    patience_counter = 0
    save_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_map = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val mAP: {val_map:.6f}"
        )

        # Checkpointing
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Best mAP! Model saved.")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val mAP: {best_map:.6f}")

    # Load best model for inference
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Loaded best model for inference.")

    # Run Inference
    inference(model, test_loader, device)
