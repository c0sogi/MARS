import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import rle_encode


class BCEDiceLoss(nn.Module):
    """
    Balanced Loss combining Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)

        # Flatten for Dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return bce_loss + dice_loss


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    criterion = BCEDiceLoss()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (volumes, labels) in enumerate(loader):
        volumes = volumes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(volumes)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Train Loss: {avg_loss}")

    return avg_loss


def evaluate(model, loader, device, threshold=0.5):
    """
    Evaluates the model on the validation set using global F0.5 score.
    """
    model.eval()
    criterion = BCEDiceLoss()
    total_loss = 0.0
    num_batches = 0

    # Global counters for F0.5 calculation
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    beta = 0.5
    smooth = 1e-6

    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device)
            labels = labels.to(device)

            logits = model(volumes)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Calculate metrics
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Flatten
            p_flat = preds.view(-1)
            t_flat = labels.view(-1)

            tp = (p_flat * t_flat).sum().item()
            fp = (p_flat * (1 - t_flat)).sum().item()
            fn = ((1 - p_flat) * t_flat).sum().item()

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Calculate Global F0.5
    beta_sq = beta**2
    precision = tp_sum / (tp_sum + fp_sum + smooth)
    recall = tp_sum / (tp_sum + fn_sum + smooth)

    score = (
        (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + smooth)
    )

    print(f"Val Loss: {avg_loss}")
    print(f"Val F0.5: {score}")

    return avg_loss, score


def predict_and_submit(
    model, loader, dataset, device, threshold=0.5, save_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set, reconstructs full images,
    encodes them with RLE, and saves the submission file.
    """
    model.eval()

    # Initialize buffers for reconstruction
    # We need to map fragment_id to its probability map and count map
    fragment_probs = {}
    fragment_counts = {}

    # Pre-allocate buffers based on dataset fragments
    for frag in dataset.fragments:
        fid = frag["id"]
        h, w = frag["mask"].shape
        fragment_probs[fid] = np.zeros((h, w), dtype=np.float32)
        fragment_counts[fid] = np.zeros((h, w), dtype=np.float32)

    # Iterate through the loader
    # We need to track the global index to retrieve coordinates from dataset.grid
    global_idx = 0

    with torch.no_grad():
        for volumes, _ in loader:
            volumes = volumes.to(device)
            logits = model(volumes)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()  # (B, 1, H, W)

            batch_size = probs_np.shape[0]

            for b in range(batch_size):
                # Get coordinates for this patch
                frag_idx, y, x = dataset.grid[global_idx]
                frag_id = dataset.fragments[frag_idx]["id"]

                patch_prob = probs_np[b, 0]  # (H, W)
                h_patch, w_patch = patch_prob.shape

                # Accumulate
                fragment_probs[frag_id][y : y + h_patch, x : x + w_patch] += patch_prob
                fragment_counts[frag_id][y : y + h_patch, x : x + w_patch] += 1.0

                global_idx += 1

    # Process each fragment to generate RLE
    submission_data = []

    for frag in dataset.fragments:
        fid = frag["id"]
        mask = frag["mask"]  # Binary mask of valid area

        # Average the probabilities
        prob_map = fragment_probs[fid]
        count_map = fragment_counts[fid]

        # Avoid division by zero
        count_map[count_map == 0] = 1.0
        prob_map /= count_map

        # Apply threshold
        binary_map = (prob_map > threshold).astype(np.uint8)

        # Apply the original valid-pixel mask to remove padding artifacts
        binary_map = binary_map * mask

        # RLE Encode
        rle = rle_encode(binary_map)
        submission_data.append({"Id": fid, "Predicted": rle})

    # Create DataFrame and Save
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
