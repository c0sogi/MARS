import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config
from library.utils import unpad_image, calc_map_score, create_submission


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        criterion: Loss function (MultiTaskLoss).
        optimizer: Optimizer.
        device: Torch device.
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
        dict: Average metrics (components of loss).
    """
    model.train()
    running_loss = 0.0
    running_metrics = {}
    dataset_size = 0

    for batch in dataloader:
        # Unpack batch: dataset.py returns (image, mask, depth) for train
        images, masks, depths = batch

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass with Depth Injection (Cite solution_lesson_node_00009)
        logits = model(images, depths)

        # Calculate loss
        loss, metrics = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate statistics
        running_loss += loss.item() * batch_size

        for k, v in metrics.items():
            if k not in running_metrics:
                running_metrics[k] = 0.0
            running_metrics[k] += v * batch_size

        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_metrics = {k: v / dataset_size for k, v in running_metrics.items()}

    print(f"Epoch {epoch} Train Loss: {epoch_loss}")
    for k, v in epoch_metrics.items():
        print(f"  {k}: {v}")

    return epoch_loss, epoch_metrics


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates Loss and mAP on unpadded (original size) images.
    Uses Adaptive Thresholding (Cite solution_lesson_node_00033).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            images, masks, depths = batch

            images = images.to(device)
            masks_gpu = masks.to(device)
            depths = depths.to(device)

            batch_size = images.size(0)

            # Forward pass with Depth Injection
            logits = model(images, depths)

            # Calculate loss
            loss, _ = criterion(logits, masks_gpu)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Prepare for mAP calculation
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()

            for i in range(batch_size):
                p = probs_np[i, 0, :, :]
                m = masks_np[i, 0, :, :]
                p_unpad = unpad_image(p, Config.ORIG_H, Config.ORIG_W)
                m_unpad = unpad_image(m, Config.ORIG_H, Config.ORIG_W)
                all_preds.append(p_unpad)
                all_masks.append(m_unpad)

    avg_loss = running_loss / dataset_size

    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    # Adaptive Thresholding (Cite solution_lesson_node_00033)
    best_map = 0.0
    best_thresh = 0.5

    thresholds = np.linspace(0.3, 0.7, 41)  # 0.30, 0.31, ..., 0.70

    for t in thresholds:
        binary_preds = (all_preds > t).astype(np.uint8)
        binary_masks = (all_masks > 0.5).astype(np.uint8)
        score = calc_map_score(binary_preds, binary_masks)

        if score > best_map:
            best_map = score
            best_thresh = t

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation mAP: {best_map:.4f} (Best Threshold: {best_thresh:.2f})")

    return avg_loss, best_map, best_thresh


def predict(model, dataloader, device):
    """
    Runs inference on a dataset (e.g., test set) using TTA.
    Returns unpadded probability maps and IDs.
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3 and isinstance(batch[2][0], str):
                images, depths, ids = batch
            elif len(batch) == 3:
                images, _, depths = batch
                ids = [None] * images.size(0)
            else:
                raise ValueError("Unexpected batch format in predict")

            images = images.to(device)
            depths = depths.to(device)

            # --- TTA: Horizontal Flip ---
            # 1. Original
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # 2. Flipped
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip_back = torch.flip(probs_flip, dims=[3])

            # Average
            avg_probs = (probs + probs_flip_back) / 2.0

            # Unpad
            avg_probs_np = avg_probs.cpu().numpy()  # (N, 1, H, W)

            for i in range(len(images)):
                p = avg_probs_np[i, 0, :, :]
                p_unpad = unpad_image(p, Config.ORIG_H, Config.ORIG_W)
                all_probs.append(p_unpad)
                if ids[i] is not None:
                    all_ids.append(ids[i])

    return all_ids, np.array(all_probs)


def generate_submission(
    model, dataloader, device, output_path=Config.SUBMISSION_PATH, threshold=0.5
):
    """
    Generates predictions for the test set and saves to CSV.
    Uses the optimal threshold found during validation.
    """
    print(f"Generating submission predictions with TTA (Threshold: {threshold:.4f})...")
    ids, probs = predict(model, dataloader, device)

    # Binarize for RLE encoding
    binary_masks = (probs > threshold).astype(np.uint8)

    print(f"Saving submission to {output_path}...")
    create_submission(ids, binary_masks, output_path)
