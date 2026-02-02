import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import AverageMeter, get_score, mask2bbox


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        data_loader: The training data loader.
        device: The device to train on.
        epoch: The current epoch number.
        scheduler: Optional learning rate scheduler.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, labels, masks) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Forward pass
        logits, pred_masks = model(images)

        # Calculate losses
        # Study-level loss: CrossEntropy (requires class indices)
        # labels are one-hot encoded float tensors, convert to indices
        targets_idx = torch.argmax(labels, dim=1)
        loss_cls = F.cross_entropy(logits, targets_idx)

        # Image-level loss: BCEWithLogits
        loss_seg = F.binary_cross_entropy_with_logits(pred_masks, masks)

        # Composite loss
        loss = (Config.lambda_cls * loss_cls) + (Config.lambda_seg * loss_seg)

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg}")
    return loss_meter.avg


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Computes loss and mAP.

    Args:
        model: The PyTorch model.
        data_loader: The validation data loader.
        device: The device to evaluate on.

    Returns:
        tuple: (Average Loss, mAP Score)
    """
    model.eval()
    loss_meter = AverageMeter()

    pred_rows = []
    gt_rows = []

    with torch.no_grad():
        for images, labels, masks in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward pass
            logits, pred_masks = model(images)

            # Calculate Loss
            targets_idx = torch.argmax(labels, dim=1)
            loss_cls = F.cross_entropy(logits, targets_idx)
            loss_seg = F.binary_cross_entropy_with_logits(pred_masks, masks)
            loss = (Config.lambda_cls * loss_cls) + (Config.lambda_seg * loss_seg)

            loss_meter.update(loss.item(), images.size(0))

            # --- Prepare for mAP Calculation ---

            # 1. Study Predictions
            study_probs = torch.softmax(logits, dim=1)
            study_preds = torch.argmax(study_probs, dim=1)

            # 2. Mask Predictions
            mask_probs = torch.sigmoid(pred_masks)

            # Convert to numpy for post-processing
            mask_probs_np = mask_probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            study_preds_np = study_preds.cpu().numpy()

            batch_size = images.size(0)

            for b in range(batch_size):
                # Ground Truth Boxes
                # masks are float32 (0.0 or 1.0), convert to uint8 for cv2
                gt_mask_b = masks_np[b, 0]
                gt_boxes = mask2bbox((gt_mask_b > 0.5).astype(np.uint8))
                gt_rows.append({"boxes": gt_boxes})

                # Prediction
                # Logic: If study is "Negative for Pneumonia" (index 0), predict no boxes.
                if study_preds_np[b] == 0:
                    pred_rows.append({"boxes": [], "scores": []})
                else:
                    # Threshold mask to get binary prediction
                    pred_mask_b = (mask_probs_np[b, 0] > 0.5).astype(np.uint8)
                    pred_boxes = mask2bbox(pred_mask_b)

                    # Calculate confidence scores for each box
                    # Using mean probability within the box
                    scores = []
                    for box in pred_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        # Clip to image bounds to avoid indexing errors
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(Config.img_size, x2)
                        y2 = min(Config.img_size, y2)

                        if x2 > x1 and y2 > y1:
                            box_prob = mask_probs_np[b, 0, y1:y2, x1:x2].mean()
                            scores.append(float(box_prob))
                        else:
                            scores.append(0.0)

                    pred_rows.append({"boxes": pred_boxes, "scores": scores})

    # Calculate mAP
    map_score = get_score(pred_rows, gt_rows, iou_threshold=0.5)

    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation mAP: {map_score}")

    return loss_meter.avg, map_score
