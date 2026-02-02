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

        # Forward pass
        # Model returns (logits, depth_pred)
        logits, pred_depths = model(images)

        # Calculate loss
        loss, metrics = criterion(logits, pred_depths, masks, depths)

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

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        float: Average loss.
        float: mAP score.
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

            # Forward pass
            logits, pred_depths = model(images)

            # Calculate loss (on padded data, as per training)
            loss, _ = criterion(logits, pred_depths, masks_gpu, depths)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Prepare for mAP calculation (Unpadding)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Convert to numpy for unpadding
            probs_np = probs.cpu().numpy()  # (N, 1, H, W)
            masks_np = masks.numpy()  # (N, 1, H, W)

            # Iterate through batch to unpad individually
            for i in range(batch_size):
                # Squeeze channel dim: (1, H, W) -> (H, W)
                p = probs_np[i, 0, :, :]
                m = masks_np[i, 0, :, :]

                # Unpad back to 101x101
                p_unpad = unpad_image(p, Config.ORIG_H, Config.ORIG_W)
                m_unpad = unpad_image(m, Config.ORIG_H, Config.ORIG_W)

                all_preds.append(p_unpad)
                all_masks.append(m_unpad)

    avg_loss = running_loss / dataset_size

    # Calculate mAP
    # Stack into arrays (N, 101, 101)
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    # Threshold predictions at 0.5 for mAP calculation input
    # Note: calc_map_score sweeps thresholds, but requires binary input for IoU calc at each step?
    # Looking at utils.py: "pred_masks = pred_masks > 0".
    # So we should pass binary masks based on a 0.5 cutoff.
    binary_preds = (all_preds > 0.5).astype(np.uint8)
    binary_masks = (all_masks > 0.5).astype(np.uint8)

    map_score = calc_map_score(binary_preds, binary_masks)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation mAP: {map_score}")

    return avg_loss, map_score


def predict(model, dataloader, device):
    """
    Runs inference on a dataset (e.g., test set) using TTA.
    Returns unpadded probability maps and IDs.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader (Test mode expected).
        device: Torch device.

    Returns:
        list: List of image IDs.
        np.ndarray: Array of predicted probability maps (N, 101, 101).
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Test loader returns: image, depth, id
            # Val loader returns: image, mask, depth

            if len(batch) == 3 and isinstance(batch[2][0], str):
                # Test mode
                images, depths, ids = batch
            elif len(batch) == 3:
                # Val/Train mode (just ignore masks)
                images, _, depths = batch
                ids = [None] * images.size(0)  # No IDs provided
            else:
                raise ValueError("Unexpected batch format in predict")

            images = images.to(device)

            # --- TTA: Horizontal Flip ---
            # 1. Original
            logits, _ = model(images)
            probs = torch.sigmoid(logits)

            # 2. Flipped
            images_flip = torch.flip(images, dims=[3])
            logits_flip, _ = model(images_flip)
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


def generate_submission(model, dataloader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: Trained model.
        dataloader: Test DataLoader.
        device: Torch device.
        output_path: Path to save submission CSV.
    """
    print("Generating submission predictions with TTA...")
    ids, probs = predict(model, dataloader, device)

    # Binarize for RLE encoding
    # Using 0.5 threshold as standard
    binary_masks = (probs > 0.5).astype(np.uint8)

    print(f"Saving submission to {output_path}...")
    create_submission(ids, binary_masks, output_path)
