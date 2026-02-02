import os
import time
import random
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    PATIENCE,
    THRESHOLD,
    BASELINE_SCORE,
    SEED,
    TILE_SIZE,
    STRIDE,
    MAX_TRAIN_SAMPLES,
)
from library.utils import rle_encode, optimize_threshold, fbeta_score
from library.losses import BCEDiceLoss
from library.data import get_loaders
from library.models import build_model


def set_seed(seed=SEED):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Performs validation and finds the optimal threshold.
    """
    model.eval()
    running_loss = 0.0

    # Store all predictions and targets for global metric calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            preds_prob = torch.sigmoid(outputs)

            # Move to CPU and flatten
            all_preds.append(preds_prob.cpu().numpy().flatten())
            all_targets.append(labels.cpu().numpy().flatten())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Find optimal threshold
    best_thresh, best_score = optimize_threshold(all_preds, all_targets, beta=0.5)

    return avg_loss, best_score, best_thresh


def tiled_inference(model, image, device, tile_size=TILE_SIZE, overlap=0):
    """
    Performs inference on a large image using tiling.

    Args:
        model: The trained model.
        image: Input tensor of shape (1, C, H, W).
        device: Torch device.
        tile_size: Size of the tiles.
        overlap: Overlap between tiles (default 0 for simplicity).

    Returns:
        np.ndarray: Probability map of shape (H, W).
    """
    model.eval()

    # Get image dimensions
    _, _, h, w = image.shape

    # Pad image to be divisible by tile_size (or at least 32 for UNet)
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    # Pad input
    # padding format: (left, right, top, bottom)
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)

    h_padded, w_padded = image_padded.shape[2], image_padded.shape[3]
    output_padded = torch.zeros(
        (h_padded, w_padded), device=device, dtype=torch.float32
    )

    # Iterate tiles
    # Using simple non-overlapping tiles as STRIDE=TILE_SIZE in config
    stride = tile_size

    with torch.no_grad():
        for y in range(0, h_padded, stride):
            for x in range(0, w_padded, stride):
                # Extract tile
                y_end = min(y + tile_size, h_padded)
                x_end = min(x + tile_size, w_padded)

                # If we are at the edge and the tile is smaller than tile_size,
                # we need to handle it. However, we padded the image, so we can just take fixed size
                # except the loop logic might go slightly over if not careful.
                # Since we padded to multiple of tile_size, y+tile_size should be exactly valid or end.

                tile = image_padded[:, :, y : y + tile_size, x : x + tile_size]

                # Move to device
                tile = tile.to(device, dtype=torch.float32)

                with autocast():
                    # Forward pass
                    output_tile = model(tile)
                    output_tile = torch.sigmoid(output_tile)

                # Squeeze batch and channel dims: (1, 1, H, W) -> (H, W)
                output_tile = output_tile.squeeze()

                # Place in output
                output_padded[y : y + tile_size, x : x + tile_size] = output_tile

    # Crop back to original size
    output = output_padded[:h, :w]

    return output.cpu().numpy()


def generate_submission(model, test_loader, device, threshold):
    """
    Generates the submission.csv file using the best model.
    """
    print(f"Generating submission with threshold: {threshold}")

    submission_data = []
    model.eval()

    for image, mask, fragment_id in test_loader:
        # image is (1, 3, H, W)
        # mask is (1, H, W) - valid area mask
        # fragment_id is tuple (id,)

        fid = fragment_id[0]
        valid_mask = mask.squeeze().numpy()  # (H, W)

        # Run inference
        pred_prob = tiled_inference(model, image, device)

        # Binarize
        pred_binary = (pred_prob > threshold).astype(np.uint8)

        # Apply valid area mask (only predict inside the scroll fragment)
        if valid_mask is not None:
            pred_binary = pred_binary * valid_mask

        # Encode
        rle = rle_encode(pred_binary)

        submission_data.append({"Id": fid, "Predicted": rle})

        # Clean up memory
        del pred_prob, pred_binary, image
        gc.collect()

    # Create DataFrame and save
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def train_model(
    max_train_samples=MAX_TRAIN_SAMPLES,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    baseline_score=BASELINE_SCORE,
):
    """
    Main training loop with validation gating and submission generation.
    """
    set_seed(SEED)

    # 1. Data Loaders
    train_loader, val_loader, test_loader = get_loaders(
        max_train_samples=max_train_samples,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # 2. Model & Optimization
    model = build_model()
    model = model.to(DEVICE)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = GradScaler()

    # 3. Training Loop
    best_val_score = 0.0
    best_val_threshold = 0.5
    epochs_no_improve = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler
        )

        # Validate
        val_loss, val_score, val_threshold = valid_one_epoch(
            model, val_loader, criterion, DEVICE
        )

        # Scheduler Step
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F0.5: {val_score} | "  # Printing full precision
            f"Best Thresh: {val_threshold:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing & Early Stopping
        if val_score > best_val_score:
            print(f"Score Improved ({best_val_score} -> {val_score}). Saving model...")
            best_val_score = val_score
            best_val_threshold = val_threshold
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training finished. Best Validation F0.5 Score: {best_val_score}")

    # 4. Validation Gating & Submission
    if best_val_score > baseline_score:
        print(f"Validation score {best_val_score} exceeds baseline {baseline_score}.")
        print("Loading best model for inference...")

        # Load best weights
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        # Generate Submission
        generate_submission(model, test_loader, DEVICE, best_val_threshold)
    else:
        print(
            f"Validation score {best_val_score} did not exceed baseline {baseline_score}."
        )
        print("Skipping submission generation.")

    return best_val_score
