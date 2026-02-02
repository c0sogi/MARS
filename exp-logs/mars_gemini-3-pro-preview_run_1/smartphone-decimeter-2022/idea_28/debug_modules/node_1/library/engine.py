import torch
import torch.nn.functional as F
import numpy as np
from library import config
from library.utils import enu_to_geodetic, haversine_distance


def apply_mask_and_compute_loss(pred, target, mask):
    """
    Computes MAE loss only on valid time steps defined by the mask.

    Args:
        pred: (Batch, Channels, Length)
        target: (Batch, Channels, Length)
        mask: (Batch, Length) - Boolean tensor where True indicates valid data

    Returns:
        Scalar loss
    """
    # Permute to (Batch, Length, Channels) to index with mask (Batch, Length)
    pred = pred.permute(0, 2, 1)
    target = target.permute(0, 2, 1)

    # Select valid elements
    # resulting shape: (Num_Valid_Elements, Channels)
    pred_valid = pred[mask]
    target_valid = target[mask]

    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return F.l1_loss(pred_valid, target_valid)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Decimated Deep Supervision.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        features = batch["features"].to(device)  # (B, C, L)
        targets = batch["targets"].to(device)  # (B, 2, L)
        mask = batch["mask"].to(device)  # (B, L)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)

        # 1. Final Head Loss (Full Resolution)
        loss_final = apply_mask_and_compute_loss(outputs["final"], targets, mask)

        # 2. Auxiliary Heads Loss (Decimated Resolution)
        loss_aux = 0.0
        for scale in config.AUXILIARY_SCALES:
            key = f"aux_{scale}"
            if key in outputs:
                # Decimate targets and mask to match auxiliary output resolution
                # Slicing with step = scale
                target_decimated = targets[:, :, ::scale]
                mask_decimated = mask[:, ::scale]

                # Ensure shapes match (handle potential off-by-one due to padding logic if any)
                # The collate_fn ensures L is multiple of 16, so slicing should be exact for scales 2, 4, 8
                pred_aux = outputs[key]

                # Truncate to min length just in case
                min_len = min(pred_aux.shape[2], target_decimated.shape[2])
                pred_aux = pred_aux[:, :, :min_len]
                target_decimated = target_decimated[:, :, :min_len]
                mask_decimated = mask_decimated[:, :min_len]

                l = apply_mask_and_compute_loss(
                    pred_aux, target_decimated, mask_decimated
                )
                loss_aux += l

        # Combined Loss
        total_loss = loss_final + (config.AUX_LOSS_WEIGHT * loss_aux)

        # Backward pass
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        optimizer.step()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    Metric: Mean of (50th percentile error + 95th percentile error) / 2 per phone.
    """
    model.eval()

    # Store errors per phone (trip)
    # Key: (drive_id, phone_name), Value: list of errors
    trip_errors = {}

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            mask = batch[
                "mask"
            ]  # Keep on CPU for indexing numpy arrays later if needed, or move to device for unpadding

            # Forward pass (Inference only needs final head)
            outputs = model(features)
            preds_enu = outputs["final"].cpu().numpy()  # (B, 2, L)

            # Process each sequence in the batch
            for i in range(features.shape[0]):
                # Identify trip
                drive_id = batch["drive_ids"][i]
                phone_name = batch["phone_names"][i]
                trip_key = f"{drive_id}-{phone_name}"

                # Get valid length
                valid_len = mask[i].sum().item()

                # Extract valid predictions (East, North)
                # Shape: (2, Valid_L) -> Transpose to (Valid_L, 2)
                pred_enu_valid = preds_enu[i, :, :valid_len].T
                d_east = pred_enu_valid[:, 0]
                d_north = pred_enu_valid[:, 1]

                # Get Metadata for reconstruction
                wls = batch["wls"][i]  # (Valid_L, 2) [Lat, Lon]
                # Note: Dataset returns wls as numpy array of shape (Original_L, 2)
                # The mask handles the padding, so we just take the first valid_len
                wls_lat = wls[:valid_len, 0]
                wls_lon = wls[:valid_len, 1]

                # Reconstruct predicted Geodetic Coordinates
                pred_lat, pred_lon = enu_to_geodetic(d_east, d_north, wls_lat, wls_lon)

                # Get Ground Truth for evaluation
                # Targets in batch are ENU offsets, but we need Lat/Lon for Haversine.
                # We can reconstruct GT Lat/Lon from GT ENU + WLS, OR we can't easily because
                # the dataset __getitem__ transforms GT to ENU.
                # However, for validation, we really want the exact distance.
                # Reconstructing GT from ENU + WLS is an approximation.
                # Ideally, we should pass GT Lat/Lon through the dataloader.
                # But since we don't have it in the batch dict, we will reconstruct GT from
                # the target ENU offsets provided in the batch.

                target_enu = batch["targets"][i, :, :valid_len].numpy().T
                gt_east = target_enu[:, 0]
                gt_north = target_enu[:, 1]
                gt_lat, gt_lon = enu_to_geodetic(gt_east, gt_north, wls_lat, wls_lon)

                # Calculate Haversine Distance
                distances = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

                if trip_key not in trip_errors:
                    trip_errors[trip_key] = []
                trip_errors[trip_key].extend(distances)

    # Compute Metric
    # Mean of (50th + 95th) / 2
    scores = []
    for trip, errors in trip_errors.items():
        if len(errors) == 0:
            continue
        errors = np.array(errors)
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score = (p50 + p95) / 2.0
        scores.append(score)

    final_metric = np.mean(scores) if scores else 0.0
    print(f"Validation Metric: {final_metric:.9f}")

    return final_metric
