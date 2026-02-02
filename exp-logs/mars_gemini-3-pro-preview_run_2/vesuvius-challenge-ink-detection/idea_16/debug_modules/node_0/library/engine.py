import torch
import numpy as np
import pandas as pd
import gc
import math
from library.config import Config
from library.utils import process_fragment_mips, rle_encode
from library.metrics import fbeta_score_numpy
from library.dataset import get_transforms


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    epoch_loss = running_loss / count
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")
    return epoch_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Executes one validation epoch and calculates F0.5 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Apply sigmoid for metric calculation
            preds_prob = torch.sigmoid(outputs)

            # Move to CPU for metric calculation
            all_preds.append(preds_prob.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / count

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate global F0.5 score
    # Flattening happens inside fbeta_score_numpy
    val_score = fbeta_score_numpy(all_preds, all_labels, beta=0.5, threshold=0.5)

    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation F0.5: {val_score}")

    return epoch_loss, val_score


def predict_tiled(model, image, device, batch_size=16):
    """
    Performs sliding-window inference on a large image.

    Args:
        model: Trained PyTorch model.
        image: Numpy array of shape (3, H, W), values in [0, 1].
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        prob_map: Numpy array of shape (H, W) with probabilities.
    """
    model.eval()
    c, h, w = image.shape
    tile_size = Config.TILE_SIZE
    stride = Config.STRIDE

    # Initialize probability map and count map for averaging
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Prepare transforms
    transforms = get_transforms(split="test")

    # Generate coordinates
    y_steps = list(range(0, h - tile_size + 1, stride))
    if (h - tile_size) % stride != 0:
        y_steps.append(h - tile_size)

    x_steps = list(range(0, w - tile_size + 1, stride))
    if (w - tile_size) % stride != 0:
        x_steps.append(w - tile_size)

    # Collect tiles
    coords = []
    tiles = []

    # Since image is (3, H, W), we need (H, W, 3) for albumentations
    image_hwc = np.transpose(image, (1, 2, 0))

    for y in y_steps:
        for x in x_steps:
            # Crop
            tile = image_hwc[y : y + tile_size, x : x + tile_size, :]

            # Transform
            augmented = transforms(image=tile, mask=None)
            tile_tensor = augmented["image"]  # (3, H, W)

            tiles.append(tile_tensor)
            coords.append((y, x))

    # Batch inference
    with torch.no_grad():
        num_tiles = len(tiles)
        num_batches = math.ceil(num_tiles / batch_size)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_tiles)

            batch_tiles = torch.stack(tiles[start_idx:end_idx]).to(device)

            # Forward pass
            outputs = model(batch_tiles)
            probs = torch.sigmoid(outputs)  # (B, 1, H, W)
            probs = probs.squeeze(1).cpu().numpy()  # (B, H, W)

            # Place back
            for j in range(end_idx - start_idx):
                y, x = coords[start_idx + j]
                prob_tile = probs[j]

                prob_map[y : y + tile_size, x : x + tile_size] += prob_tile
                count_map[y : y + tile_size, x : x + tile_size] += 1.0

    # Average overlaps
    # Avoid division by zero (should not happen with valid tiling)
    count_map[count_map == 0] = 1.0
    prob_map /= count_map

    return prob_map


def predict_with_z_scanning(model, test_df, device):
    """
    Generates predictions for the test set using Decoupled Volumetric Z-Scanning.

    Args:
        model: Trained PyTorch model.
        test_df: DataFrame containing test metadata.
        device: Torch device.

    Returns:
        List of dicts [{'Id': frag_id, 'Predicted': rle_string}, ...]
    """
    results = []
    z_starts = Config.INFERENCE_Z_STARTS

    print(f"Starting Inference with Z-Scanning: {z_starts}")

    for _, row in test_df.iterrows():
        frag_id = str(row["fragment_id"])
        volume_path = row["volume_path"]

        print(f"Processing Fragment {frag_id}...")

        # Container for the multi-depth probability maps
        scan_prob_maps = []

        for z in z_starts:
            # 1. Generate/Load MIPs for this depth
            # Returns (3, H, W) float32 [0, 1]
            try:
                image = process_fragment_mips(
                    fragment_id=frag_id,
                    volume_path=volume_path,
                    z_start=z,
                    load_cached_data=True,
                )
            except Exception as e:
                print(f"Error loading Z={z} for fragment {frag_id}: {e}")
                continue

            # 2. Run tiled inference
            prob_map = predict_tiled(model, image, device, batch_size=Config.BATCH_SIZE)
            scan_prob_maps.append(prob_map)

            # Explicitly free memory for the image
            del image
            gc.collect()

        if not scan_prob_maps:
            print(
                f"Warning: No valid predictions for fragment {frag_id}. Returning empty mask."
            )
            # Get shape from metadata or assume based on last load?
            # Ideally we should have shape. We'll handle this by skipping or empty string.
            results.append({"Id": frag_id, "Predicted": ""})
            continue

        # 3. Max-Fusion
        # Stack maps: (Num_Scans, H, W)
        stacked_maps = np.stack(scan_prob_maps, axis=0)
        fused_map = np.max(stacked_maps, axis=0)

        # Free memory
        del scan_prob_maps
        del stacked_maps
        gc.collect()

        # 4. Threshold and Encode
        binary_mask = (fused_map > 0.5).astype(np.uint8)
        rle_string = rle_encode(binary_mask)

        results.append({"Id": frag_id, "Predicted": rle_string})

        # Free memory
        del fused_map
        del binary_mask
        gc.collect()

    return results
