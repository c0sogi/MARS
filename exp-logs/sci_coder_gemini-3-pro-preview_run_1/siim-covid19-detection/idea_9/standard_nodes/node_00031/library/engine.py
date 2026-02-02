import torch
import numpy as np
import pandas as pd
import sys
import os
from library.config import Config
from library.utils import mixup_data, mask2bbox, get_map_score
from library.loss import MultiTaskLoss


class AverageMeter:
    """Computes and stores the average and current value"""

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


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch using MixUp regularization.
    """
    model.train()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    criterion = MultiTaskLoss()

    for i, (images, labels, masks) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Apply MixUp
        mixed_images, indices, lam = mixup_data(
            images, alpha=Config.MIXUP_ALPHA, device=device
        )

        # Mix Targets
        # labels: (B, 4), masks: (B, 1, H, W)
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        mixed_masks = lam * masks + (1 - lam) * masks[indices]

        optimizer.zero_grad()

        # Forward pass
        cls_logits, seg_logits = model(mixed_images)

        # Calculate Loss
        loss_dict = criterion(cls_logits, seg_logits, mixed_labels, mixed_masks)
        loss = loss_dict["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        # Logging
        loss_meter.update(loss.item(), images.size(0))
        cls_loss_meter.update(loss_dict["cls_loss"].item(), images.size(0))
        seg_loss_meter.update(loss_dict["seg_loss"].item(), images.size(0))

    # Step Scheduler (Cosine Annealing is typically stepped per epoch)
    if scheduler is not None:
        scheduler.step()

    print(
        f"Epoch [{epoch+1}/{Config.EPOCHS}] Train Loss: {loss_meter.avg:.5f} "
        f"(Cls: {cls_loss_meter.avg:.5f}, Seg: {seg_loss_meter.avg:.5f})"
    )

    return loss_meter.avg


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set and computes mAP.
    """
    model.eval()
    loss_meter = AverageMeter()

    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []

    criterion = MultiTaskLoss()

    with torch.no_grad():
        for images, labels, masks in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            cls_logits, seg_logits = model(images)

            # Validation Loss
            loss_dict = criterion(cls_logits, seg_logits, labels, masks)
            loss_meter.update(loss_dict["loss"].item(), images.size(0))

            # Predictions
            cls_probs = torch.softmax(cls_logits, dim=1)
            seg_probs = torch.sigmoid(seg_logits)

            # Convert to numpy for processing
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy()
            cls_probs_np = cls_probs.cpu().numpy()
            seg_probs_np = seg_probs.cpu().numpy()

            batch_size = images.size(0)

            for i in range(batch_size):
                # --- Ground Truth ---
                # Get class index (argmax of one-hot)
                gt_cls = np.argmax(labels_np[i])
                # Extract boxes from GT mask
                gt_bbox = mask2bbox(masks_np[i, 0], threshold=0.5)

                gt_boxes_list.append(gt_bbox)
                # Assign the study label to all boxes in this image
                gt_labels_list.append([gt_cls] * len(gt_bbox))

                # --- Predictions ---
                # Get predicted class
                pred_cls = np.argmax(cls_probs_np[i])
                pred_conf = cls_probs_np[i, pred_cls]

                # Extract boxes from predicted mask
                pred_bbox = mask2bbox(seg_probs_np[i, 0], threshold=0.5)

                pred_boxes_list.append(pred_bbox)
                pred_labels_list.append([pred_cls] * len(pred_bbox))
                pred_scores_list.append([pred_conf] * len(pred_bbox))

    # Calculate mAP
    map_score = get_map_score(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        iou_threshold=0.5,
        num_classes=Config.NUM_CLASSES,
    )

    print(f"Val Loss: {loss_meter.avg:.5f} | mAP: {map_score:.5f}")
    return loss_meter.avg, map_score


def fit(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience=5
):
    """
    Main training loop with Early Stopping.
    """
    best_map = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )
        val_loss, val_map = evaluate(model, val_loader, device)

        # Checkpoint Strategy: Maximize mAP
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best mAP: {best_map:.5f}. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def inference_and_submit(
    model, test_loader, device, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    results = []

    # Class mapping
    class_names = ["negative", "typical", "indeterminate", "atypical"]

    print("Running inference on test set...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            cls_logits, seg_logits = model(images)

            cls_probs = torch.softmax(cls_logits, dim=1)
            seg_probs = torch.sigmoid(seg_logits)

            cls_probs_np = cls_probs.cpu().numpy()
            seg_probs_np = seg_probs.cpu().numpy()

            for i in range(len(image_ids)):
                img_id = image_ids[i]

                # Prediction
                pred_cls_idx = np.argmax(cls_probs_np[i])
                pred_cls_name = class_names[pred_cls_idx]
                pred_cls_conf = cls_probs_np[i, pred_cls_idx]

                # --- Study Level Prediction String ---
                # Format: class_id confidence 0 0 1 1
                study_pred_string = f"{pred_cls_name} {pred_cls_conf:.6f} 0 0 1 1"

                # --- Image Level Prediction String ---
                # Gated Logic: If negative, predict 'none'. Else, predict boxes.
                if pred_cls_name == "negative":
                    image_pred_string = "none 1 0 0 1 1"
                else:
                    boxes = mask2bbox(seg_probs_np[i, 0], threshold=0.5)

                    if len(boxes) == 0:
                        image_pred_string = "none 1 0 0 1 1"
                    else:
                        box_strs = []
                        for box in boxes:
                            # box: [xmin, ymin, xmax, ymax]
                            # Submission format: opacity confidence xmin ymin xmax ymax
                            b_str = f"opacity {pred_cls_conf:.6f} {box[0]} {box[1]} {box[2]} {box[3]}"
                            box_strs.append(b_str)
                        image_pred_string = " ".join(box_strs)

                results.append(
                    {
                        "image_id": img_id,
                        "study_pred": study_pred_string,
                        "image_pred": image_pred_string,
                    }
                )

    # --- Generate Submission CSV ---
    # Load test metadata to map image_id to study_id
    test_df = pd.read_csv(Config.TEST_METADATA)
    img_to_study = dict(zip(test_df["image_id"], test_df["study_id"]))

    submission_rows = []
    processed_studies = set()

    for res in results:
        img_id = res["image_id"]
        study_id = img_to_study.get(img_id, f"{img_id}_study")

        # Image Row
        submission_rows.append(
            {"id": f"{img_id}_image", "PredictionString": res["image_pred"]}
        )

        # Study Row (Add only once per study)
        if study_id not in processed_studies:
            submission_rows.append(
                {"id": f"{study_id}_study", "PredictionString": res["study_pred"]}
            )
            processed_studies.add(study_id)

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
