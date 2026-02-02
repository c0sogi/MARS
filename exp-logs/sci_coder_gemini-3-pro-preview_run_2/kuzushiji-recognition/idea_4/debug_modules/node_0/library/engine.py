import torch
import math
import sys
import os
import numpy as np
import pandas as pd
import time
from library.config import Config
from library.utils import calculate_f1_score
from library.dataset import get_label_map


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = []
    header = f"Epoch: [{epoch}]"

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

        optimizer.step()

        metric_logger.append(loss_value)

        if i % print_freq == 0:
            print(f"{header} Iter: [{i}/{len(data_loader)}] Loss: {loss_value:.6f}")

    avg_loss = np.mean(metric_logger)
    print(f"{header} Average Loss: {avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # Store all predictions and targets for global metric calculation
    all_preds = []
    all_targets = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        # Forward pass
        outputs = model(images)

        # Move to CPU and Rescale
        for i, output in enumerate(outputs):
            scale = targets[i]["scale_factor"].item()

            # Rescale predictions to original image size
            # Output boxes are (x1, y1, x2, y2)
            pred_boxes = output["boxes"].detach().cpu() / scale
            pred_labels = output["labels"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()

            all_preds.append(
                {"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores}
            )

            # Rescale targets to original image size for fair comparison
            tgt_boxes = targets[i]["boxes"].detach().cpu() / scale
            tgt_labels = targets[i]["labels"].detach().cpu()

            all_targets.append({"boxes": tgt_boxes, "labels": tgt_labels})

    # Calculate metrics
    metrics = calculate_f1_score(
        all_preds, all_targets, score_threshold=Config.SCORE_THRESH
    )

    print(f"Validation F1: {metrics['f1']:.10f}")
    print(f"Validation Precision: {metrics['precision']:.10f}")
    print(f"Validation Recall: {metrics['recall']:.10f}")

    return metrics


def train_eval_loop(
    model, optimizer, train_loader, val_loader, device, epochs, patience=3
):
    """
    Runs the training and evaluation loop with Early Stopping.
    """
    best_f1 = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=Config.LR_STEPS, gamma=Config.LR_GAMMA
    )

    for epoch in range(epochs):
        # Train
        train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Step scheduler
        lr_scheduler.step()

        # Evaluate
        metrics = evaluate(model, val_loader, device)
        val_f1 = metrics["f1"]

        # Early Stopping and Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            print(f"New best F1: {best_f1:.10f}. Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            print(
                f"No improvement for {epochs_no_improve} epochs. Best F1: {best_f1:.10f}"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("Training finished.")
    return best_f1


@torch.no_grad()
def inference(model, data_loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Starting inference...")
    model.eval()

    # Load label map and create inverse map (ID -> Unicode)
    # Background is 0, so we don't include it in the map for lookup
    label_map = get_label_map()
    id_to_char = {v: k for k, v in label_map.items()}

    results = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id_str"]
            scale = targets[i]["scale_factor"].item()

            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()

            # Filter by score threshold
            mask = scores >= Config.SCORE_THRESH
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            # Cap detections
            if len(boxes) > Config.DETECTIONS_PER_IMG:
                indices = np.argsort(scores)[::-1][: Config.DETECTIONS_PER_IMG]
                boxes = boxes[indices]
                labels = labels[indices]
                scores = scores[indices]

            # Rescale to original coordinates
            boxes = boxes / scale

            # Calculate centers
            centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
            centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

            # Format label string
            label_strs = []
            for j in range(len(labels)):
                char_id = labels[j]
                if char_id in id_to_char:
                    char = id_to_char[char_id]
                    cx = int(centers_x[j])
                    cy = int(centers_y[j])
                    label_strs.append(f"{char} {cx} {cy}")

            label_str = " ".join(label_strs)
            results.append({"image_id": image_id, "labels": label_str})

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
