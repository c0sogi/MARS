import math
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, format_prediction_string
from library.dataset import get_processed_metadata


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Trains the model for one epoch.
    Includes linear warmup for the first epoch and gradient clipping.
    """
    model.train()
    metric_logger = {
        "loss": AverageMeter(),
        "loss_study": AverageMeter(),
        "lr": AverageMeter(),
    }

    header = f"Epoch: [{epoch}]"

    # Linear Warmup for the first epoch
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = len(data_loader) - 1
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets, _) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # Sum all losses (Study loss is already weighted in model.py)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # Gradient Clipping (Stability Strategy)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update metrics
        metric_logger["loss"].update(loss_value)
        if "loss_study" in loss_dict:
            metric_logger["loss_study"].update(loss_dict["loss_study"].item())
        metric_logger["lr"].update(optimizer.param_groups[0]["lr"])

        if i % print_freq == 0:
            print(
                f"{header} [{i}/{len(data_loader)}] "
                f"Loss: {metric_logger['loss'].val:.4f} ({metric_logger['loss'].avg:.4f}) "
                f"Study Loss: {metric_logger['loss_study'].val:.4f} ({metric_logger['loss_study'].avg:.4f}) "
                f"LR: {metric_logger['lr'].val:.6f}"
            )

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Computes Validation Loss (criterion for checkpointing) and Study Accuracy.
    """
    # 1. Compute Validation Loss
    # We set model to train() to compute losses, but disable gradients
    model.train()
    loss_meter = AverageMeter()

    print("Evaluating Validation Loss...")
    for images, targets, _ in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_meter.update(losses.item())

    val_loss = loss_meter.avg
    print(f"Validation Loss: {val_loss}")

    # 2. Compute Study Accuracy
    model.eval()
    correct_study = 0
    total_study = 0

    print("Evaluating Study Accuracy...")
    for images, targets, _ in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward in eval mode returns (detections, study_probs)
        _, study_probs = model(images)

        gt_study_labels = torch.stack([t["study_label"] for t in targets])
        preds = torch.argmax(study_probs, dim=1)

        correct_study += (preds == gt_study_labels).sum().item()
        total_study += len(gt_study_labels)

    study_acc = correct_study / total_study if total_study > 0 else 0.0
    print(f"Study Accuracy: {study_acc}")

    return val_loss


@torch.no_grad()
def generate_submission(model, data_loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    Applies consistency rules: Negative Study -> No Opacity Boxes.
    """
    model.eval()

    # Load test metadata to map image_id to study_id
    test_df = get_processed_metadata("test", load_cached_data=True)
    image_to_study = dict(zip(test_df["image_id"], test_df["StudyInstanceUID"]))

    results = []

    print("Generating submission predictions...")
    for images, image_ids in data_loader:
        images = list(image.to(device) for image in images)

        # Forward pass
        detections, study_probs = model(images)

        for i, img_id in enumerate(image_ids):
            # 1. Study Prediction
            # Get class with highest probability
            study_prob = study_probs[i]
            study_class_id = torch.argmax(study_prob).item()
            study_conf = study_prob[study_class_id].item()
            study_label = Config.ID_TO_STUDY_CLASS[study_class_id]

            study_id = image_to_study.get(img_id, "unknown")

            # Format Study Prediction: "class conf 0 0 1 1"
            study_pred_str = f"{study_label} {study_conf} 0 0 1 1"

            # Add Study Row (Note: This might create duplicates if multiple images per study,
            # but we'll handle unique rows at the end)
            results.append(
                {"id": f"{study_id}_study", "PredictionString": study_pred_str}
            )

            # 2. Image Prediction
            # Consistency Rule: If Study is Negative, Image must be "none"
            if study_label == "Negative for Pneumonia":
                image_pred_str = "none 1 0 0 1 1"
            else:
                # Process Detections
                det = detections[i]
                boxes = det["boxes"].cpu().numpy()
                scores = det["scores"].cpu().numpy()
                labels = det["labels"].cpu().numpy()

                # Filter by threshold (already done by model config, but being safe)
                # Model config ROI_HEADS_SCORE_THRESH is 0.05

                if len(boxes) > 0:
                    # Map class IDs to "opacity"
                    # Config.DETECTION_CLASS_MAP maps 1,2,3 -> "opacity"
                    label_strs = [
                        Config.DETECTION_CLASS_MAP.get(l, "opacity") for l in labels
                    ]

                    image_pred_str = format_prediction_string(label_strs, boxes, scores)
                else:
                    image_pred_str = "none 1 0 0 1 1"

            # Add Image Row
            results.append(
                {"id": f"{img_id}_image", "PredictionString": image_pred_str}
            )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Deduplicate: For studies with multiple images, we might have multiple rows.
    # We'll simply take the first one (or could average, but taking first is safe for now).
    submission_df = submission_df.drop_duplicates(subset=["id"])

    # Save
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
