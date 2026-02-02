import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config

# =============================================================================
# Helper Functions for Decoding (CenterNet)
# =============================================================================


def _gather_feat(feat, ind, mask=None):
    """Gathers values from a feature map at specific indices."""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """Transposes feature map and gathers values."""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    feat = feat.gather(1, ind)
    return feat


def _topk(scores, K=40):
    """Selects top K scores from the heatmap."""
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).float()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _nms(heat, kernel=3):
    """Performs Non-Maximum Suppression using Max Pooling."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_predictions(hm, wh, reg, stride=4, K=100):
    """
    Decodes model outputs into bounding boxes.

    Args:
        hm: Heatmap logits [B, C, H, W]
        wh: Width/Height predictions [B, 2, H, W]
        reg: Offset predictions [B, 2, H, W]
        stride: Downsampling factor of the feature map (default 4)
        K: Number of top objects to select

    Returns:
        detections: Tensor of shape [B, K, 6] -> (x1, y1, x2, y2, score, class)
    """
    batch_size, cat, height, width = hm.size()

    # Apply sigmoid to heatmap to get probabilities
    heat = torch.sigmoid(hm)

    # Perform NMS
    heat = _nms(heat)

    # Get top K peaks
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    # Gather regression offsets
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    # Gather width/height
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch_size, K, 2)

    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # Convert from center (xs, ys) and size (wh) to bounding box (x1, y1, x2, y2)
    # Note: xs, ys, wh are in feature map scale
    x1 = xs - wh[..., 0:1] / 2
    y1 = ys - wh[..., 1:2] / 2
    x2 = xs + wh[..., 0:1] / 2
    y2 = ys + wh[..., 1:2] / 2

    bboxes = torch.cat([x1, y1, x2, y2], dim=2)

    # Scale up to input image size
    bboxes *= stride

    # Concatenate: [bboxes, scores, clses]
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


# =============================================================================
# Engine Logic
# =============================================================================


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Runs one epoch of training."""
    model.train()
    running_loss = 0.0
    stats = {"hm_loss": 0.0, "wh_loss": 0.0, "off_loss": 0.0, "global_loss": 0.0}

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # Move targets to device
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss, loss_stats = criterion(outputs, targets)

        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        for k, v in loss_stats.items():
            if k in stats:
                stats[k] += v.item()

    n = len(dataloader)
    return running_loss / n, {k: v / n for k, v in stats.items()}


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    stats = {"hm_loss": 0.0, "wh_loss": 0.0, "off_loss": 0.0, "global_loss": 0.0}

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(device)

            outputs = model(images)
            loss, loss_stats = criterion(outputs, targets)

            running_loss += loss.item()
            for k, v in loss_stats.items():
                if k in stats:
                    stats[k] += v.item()

    n = len(dataloader)
    return running_loss / n, {k: v / n for k, v in stats.items()}


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    num_epochs=Config.NUM_EPOCHS,
    patience=5,
):
    """Orchestrates the training process with Early Stopping."""
    best_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        train_loss, train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_stats = evaluate(model, val_loader, criterion, device)

        # Cite solution_lesson_node_00011: Step scheduler
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"  Train Loss: {train_loss:.6f} (HM: {train_stats['hm_loss']:.6f}, WH: {train_stats['wh_loss']:.6f}, Off: {train_stats['off_loss']:.6f}, Glob: {train_stats['global_loss']:.6f})"
        )
        print(
            f"  Val Loss:   {val_loss:.6f} (HM: {val_stats['hm_loss']:.6f}, WH: {val_stats['wh_loss']:.6f}, Off: {val_stats['off_loss']:.6f}, Glob: {val_stats['global_loss']:.6f})"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(
                f"  Validation loss improved. Model saved to {Config.MODEL_SAVE_PATH}"
            )
        else:
            epochs_no_improve += 1
            print(f"  No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


def generate_submission(model, test_loader, device):
    """Runs inference on the test set and generates the submission CSV."""
    print("Generating submission...")
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)

            outputs = model(images)

            # 1. Global Head Prediction
            # Model outputs logits for "Finding" (1) vs "No Finding" (0)
            # We convert to probability of "Finding"
            global_probs = torch.sigmoid(outputs["global_logits"]).view(-1)

            # 2. Decode Boxes
            # detections: [B, K, 6] (x1, y1, x2, y2, score, class)
            detections = decode_predictions(
                outputs["hm"], outputs["wh"], outputs["reg"]
            )

            batch_size = images.size(0)

            for i in range(batch_size):
                img_id = targets["image_id"][i]
                orig_shape = targets["original_shape"][i]  # [H, W]
                orig_h, orig_w = orig_shape[0].item(), orig_shape[1].item()

                # Probability of finding
                p_finding = global_probs[i].item()
                p_no_finding = 1.0 - p_finding

                prediction_string = ""

                # 3. Apply Global Gate
                # If the model is confident there is NO finding, output Class 14
                if p_no_finding > Config.NO_FINDING_THRESH:
                    prediction_string = "14 1 0 0 1 1"
                else:
                    # Process detections
                    dets = detections[i]  # [K, 6]

                    # Filter by confidence threshold
                    mask = dets[:, 4] > Config.CONF_THRESHOLD
                    valid_dets = dets[mask]

                    if len(valid_dets) == 0:
                        # Fallback if global head said finding but no boxes passed threshold
                        prediction_string = "14 1 0 0 1 1"
                    else:
                        # 4. Rescale boxes from 512x512 to Original Size
                        scale_x = orig_w / Config.IMG_SIZE
                        scale_y = orig_h / Config.IMG_SIZE

                        box_strs = []
                        for det in valid_dets:
                            x1, y1, x2, y2, score, cls_id = det.tolist()

                            # Rescale
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y

                            # Clip to image boundaries
                            x1 = max(0, min(x1, orig_w))
                            y1 = max(0, min(y1, orig_h))
                            x2 = max(0, min(x2, orig_w))
                            y2 = max(0, min(y2, orig_h))

                            cls_id = int(cls_id)

                            box_strs.append(
                                f"{cls_id} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
                            )

                        prediction_string = " ".join(box_strs)

                results.append(
                    {"image_id": img_id, "PredictionString": prediction_string}
                )

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
