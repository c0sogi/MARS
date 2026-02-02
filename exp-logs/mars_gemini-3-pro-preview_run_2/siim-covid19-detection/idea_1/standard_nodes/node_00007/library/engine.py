import torch
import math
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config, CLASS_ID_TO_LABEL
from library.utils import format_prediction_string


def calculate_iou(box1, box2):
    """
    Calculates IoU between two boxes [xmin, ymin, xmax, ymax].
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


def calculate_ap(recalls, precisions):
    """
    Computes Average Precision using the VOC 2010 method (Area Under Curve).
    """
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Integrate area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


class Engine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        final_loss = 0.0
        count = 0

        # Iterate over data
        # Note: We avoid tqdm here to keep logs clean as per instructions,
        # but simple print statements for epoch progress are allowed.
        for images, targets, image_ids in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            final_loss += loss_value
            count += 1

        if self.scheduler:
            self.scheduler.step()

        avg_loss = final_loss / count if count > 0 else 0
        return avg_loss

    @torch.no_grad()
    def evaluate_loss(self, data_loader):
        # To get loss from torchvision detection models, we must be in train mode
        # even during validation, but with no_grad.
        self.model.train()
        total_loss = 0.0
        count = 0

        for images, targets, image_ids in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            count += 1

        return total_loss / count if count > 0 else 0

    def evaluate_map(self, data_loader):
        """
        Evaluates mAP @ IoU > 0.5 on the validation set.
        """
        self.model.eval()
        num_classes = Config.NUM_CLASSES

        # Store all GT and Predictions
        # structure: {class_id: [{'box': [], 'score': float, 'img_id': int}]}
        gt_data = {c: [] for c in range(1, num_classes)}
        pred_data = {c: [] for c in range(1, num_classes)}

        with torch.no_grad():
            for images, targets, image_ids in data_loader:
                images = list(img.to(self.device) for img in images)

                # Get Predictions
                outputs = self.model(images)
                outputs = [{k: v.cpu().numpy() for k, v in t.items()} for t in outputs]

                # Process Batch
                for i, output in enumerate(outputs):
                    img_id = image_ids[i]
                    target = targets[i]

                    # Ground Truth
                    gt_boxes = target["boxes"].numpy()
                    gt_labels = target["labels"].numpy()

                    for box, label in zip(gt_boxes, gt_labels):
                        if label in gt_data:
                            gt_data[label].append(
                                {"box": box, "used": False, "img_id": img_id}
                            )

                    # Predictions
                    pred_boxes = output["boxes"]
                    pred_scores = output["scores"]
                    pred_labels = output["labels"]

                    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                        if label in pred_data:
                            pred_data[label].append(
                                {"box": box, "score": score, "img_id": img_id}
                            )

        # Calculate AP per class
        aps = []

        for c in range(1, num_classes):
            gts = gt_data[c]
            preds = pred_data[c]

            # Sort predictions by confidence descending
            preds.sort(key=lambda x: x["score"], reverse=True)

            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            # Total ground truths for this class
            n_pos = len(gts)

            if n_pos == 0:
                aps.append(0.0)
                continue

            # Group GT by image for faster lookup
            gts_by_image = {}
            for idx, gt in enumerate(gts):
                iid = gt["img_id"]
                if iid not in gts_by_image:
                    gts_by_image[iid] = []
                gts_by_image[iid].append(idx)

            for i, pred in enumerate(preds):
                img_id = pred["img_id"]
                pred_box = pred["box"]

                best_iou = 0.0
                best_gt_idx = -1

                if img_id in gts_by_image:
                    for gt_idx in gts_by_image[img_id]:
                        iou = calculate_iou(pred_box, gts[gt_idx]["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                if best_iou > 0.5:
                    if not gts[best_gt_idx]["used"]:
                        tp[i] = 1.0
                        gts[best_gt_idx]["used"] = True
                    else:
                        fp[i] = 1.0
                else:
                    fp[i] = 1.0

            # Compute Precision and Recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / n_pos
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            ap = calculate_ap(recalls, precisions)
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def fit_model(self, train_loader, val_loader, epochs, patience=3):
        best_map = 0.0
        patience_counter = 0

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.evaluate_loss(val_loader)
            val_map = self.evaluate_map(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val mAP: {val_map:.6f}"
            )

            # Early Stopping and Checkpointing
            # Cite solution_lesson_node_00006: Checkpoint on mAP, not Loss
            if val_map > best_map:
                best_map = val_map
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"Validation mAP improved. Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")
        return save_path

    @torch.no_grad()
    def inference(self, test_loader, model_path):
        print("Starting inference...")

        # Load best model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(
                f"Warning: Model path {model_path} does not exist. Using current model weights."
            )

        self.model.eval()
        self.model.to(self.device)

        results = []
        study_preds = {}  # Store list of (score, pred_string) for each study_id

        # Threshold for keeping a box
        detection_threshold = Config.DETECTION_THRESHOLD

        for images, image_ids, study_ids in test_loader:
            images = list(img.to(self.device) for img in images)

            # Forward pass returns list of dicts: [{'boxes':..., 'labels':..., 'scores':...}, ...]
            outputs = self.model(images)

            # Move to CPU for processing
            outputs = [{k: v.cpu().numpy() for k, v in t.items()} for t in outputs]

            for i, output in enumerate(outputs):
                image_id = image_ids[i]
                study_id = study_ids[i]

                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                # Filter by threshold
                mask = scores >= detection_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                # --- 1. Study Level Prediction Accumulation ---
                # We collect predictions from all images in a study and aggregate later.
                if len(boxes) == 0:
                    # Negative prediction with 0 score (placeholder to not override positives)
                    study_preds.setdefault(study_id, []).append(
                        (0.0, "negative 1 0 0 1 1")
                    )
                else:
                    # Find box with max score
                    max_idx = np.argmax(scores)
                    max_score = scores[max_idx]
                    max_label_id = labels[max_idx]

                    # Map ID to string (1: typical, 2: indeterminate, 3: atypical)
                    label_str = CLASS_ID_TO_LABEL.get(max_label_id, "negative")

                    if label_str == "negative":
                        study_preds.setdefault(study_id, []).append(
                            (0.0, "negative 1 0 0 1 1")
                        )
                    else:
                        pred_str = f"{label_str} {max_score:.6f} 0 0 1 1"
                        study_preds.setdefault(study_id, []).append(
                            (max_score, pred_str)
                        )

                # --- 2. Image Level Prediction ---
                # Logic:
                # If no boxes -> "none 1 0 0 1 1"
                # Else -> All boxes are "opacity".
                # Format: "opacity {score} {xmin} {ymin} {xmax} {ymax} ..."

                img_pred_id = f"{image_id}_image"

                if len(boxes) == 0:
                    image_pred = "none 1 0 0 1 1"
                else:
                    # Create lists for formatter
                    # Class is always "opacity" for image level
                    pred_labels = ["opacity"] * len(boxes)
                    pred_scores = scores.tolist()
                    pred_boxes = boxes.tolist()  # [xmin, ymin, xmax, ymax]

                    image_pred = format_prediction_string(
                        pred_labels, pred_boxes, pred_scores
                    )

                results.append({"id": img_pred_id, "PredictionString": image_pred})

        # --- Aggregate Study Predictions ---
        for study_id, preds in study_preds.items():
            # Sort by score descending.
            # If any positive prediction exists (score > 0), it will be at the top.
            # If all are negative (score 0), the first one is picked.
            preds.sort(key=lambda x: x[0], reverse=True)
            best_pred = preds[0][1]

            results.append({"id": f"{study_id}_study", "PredictionString": best_pred})

        # Save Submission
        submission_df = pd.DataFrame(results)

        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        return submission_df
