import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import seed_everything, fbeta_score, load_volume, rle_encoding
from library.losses import BCEDiceLoss
from library.model import SGDN
from library.data import get_loaders, InkDataset
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for volumes, labels, _ in loader:
        volumes = volumes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(volumes)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set by reconstructing the full fragment.
    Returns flattened probabilities and targets for metric computation.
    """
    model.eval()

    # 1. Identify validation fragments from metadata
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_frag_ids = df_val["fragment_id"].astype(str).unique()

    # Pre-load ground truth and initialize reconstruction buffers
    gt_data = {}
    reconstruction = {}

    for fid in val_frag_ids:
        # Load mask and label (cached)
        _, mask, label = load_volume(fid, split="val", load_cached_data=True)

        gt_data[fid] = {"mask": mask, "label": label}

        reconstruction[fid] = {
            "prob_sum": np.zeros(mask.shape, dtype=np.float32),
            "count": np.zeros(mask.shape, dtype=np.float32),
        }

    # 2. Inference Loop over tiles
    with torch.no_grad():
        for volumes, _, coords in loader:
            volumes = volumes.to(device)
            logits = model(volumes)
            probs = torch.sigmoid(logits).cpu().numpy()
            coords = coords.numpy()

            batch_size = probs.shape[0]
            half_size = Config.PATCH_SIZE // 2

            for i in range(batch_size):
                # Parse coordinate (frag_idx, center_y, center_x)
                fid_idx = int(coords[i, 0])
                cy = int(coords[i, 1])
                cx = int(coords[i, 2])

                fid = val_frag_ids[fid_idx]

                # Calculate patch bounds in original image space
                # coords are centers in original image
                y_start = cy - half_size
                x_start = cx - half_size

                # Handle boundary clipping
                H, W = reconstruction[fid]["prob_sum"].shape

                y_start_clamped = max(0, y_start)
                x_start_clamped = max(0, x_start)
                y_end_clamped = min(H, y_start + Config.PATCH_SIZE)
                x_end_clamped = min(W, x_start + Config.PATCH_SIZE)

                if y_start_clamped >= y_end_clamped or x_start_clamped >= x_end_clamped:
                    continue

                # Calculate offsets within the patch
                py_start = y_start_clamped - y_start
                px_start = x_start_clamped - x_start
                py_end = py_start + (y_end_clamped - y_start_clamped)
                px_end = px_start + (x_end_clamped - x_start_clamped)

                # Accumulate predictions
                reconstruction[fid]["prob_sum"][
                    y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped
                ] += probs[i, 0, py_start:py_end, px_start:px_end]
                reconstruction[fid]["count"][
                    y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped
                ] += 1.0

    # 3. Aggregate results
    all_preds = []
    all_targets = []

    for fid in val_frag_ids:
        prob_sum = reconstruction[fid]["prob_sum"]
        count = reconstruction[fid]["count"]
        mask = gt_data[fid]["mask"]
        label = gt_data[fid]["label"]

        # Normalize by overlap count
        # Avoid division by zero
        count[count == 0] = 1.0
        prob_map = prob_sum / count

        # Only evaluate on valid mask pixels
        valid_mask = mask > 0

        all_preds.append(prob_map[valid_mask])
        all_targets.append(label[valid_mask])

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
    else:
        all_preds = np.array([])
        all_targets = np.array([])

    return all_preds, all_targets


def optimize_threshold(preds, targets):
    """
    Finds the best threshold for F0.5 score.
    """
    best_score = 0.0
    best_thresh = 0.5

    for thresh in Config.THRESHOLD_SEARCH_RANGE:
        score = fbeta_score(preds, targets, beta=0.5, threshold=thresh)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_score, best_thresh


def train():
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Setup
    train_loader, val_loader = get_loaders()
    model = SGDN().to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    best_val_score = 0.0
    best_threshold = 0.5
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_preds, val_targets = validate(model, val_loader, device)
        val_score, val_thresh = optimize_threshold(val_preds, val_targets)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val F0.5: {val_score:.10f} (Thresh: {val_thresh:.2f})"
        )

        # Checkpointing
        if val_score > best_val_score:
            best_val_score = val_score
            best_threshold = val_thresh
            patience_counter = 0

            # Save model
            torch.save(model.state_dict(), Config.WORKING_DIR / "best_model.pth")

            # Save threshold
            with open(Config.WORKING_DIR / "threshold.txt", "w") as f:
                f.write(str(best_threshold))
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(
        f"Training finished. Best F0.5: {best_val_score:.10f} at Threshold: {best_threshold:.2f}"
    )


def inference():
    """
    Generates submission file using the best trained model.
    """
    device = torch.device(Config.DEVICE)
    model = SGDN().to(device)

    # Load model weights
    model_path = Config.WORKING_DIR / "best_model.pth"
    if not model_path.exists():
        print("Error: best_model.pth not found. Cannot run inference.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load best threshold
    threshold_path = Config.WORKING_DIR / "threshold.txt"
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            best_threshold = float(f.read().strip())
    else:
        best_threshold = 0.5
    print(f"Running inference with threshold: {best_threshold}")

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    test_ids = df_test["fragment_id"].astype(str).unique()

    submission_rows = []

    for fid in test_ids:
        # Create Dataset and Loader for this fragment
        ds = InkDataset(split="test", fragment_ids=[fid], samples_per_epoch=None)
        loader = DataLoader(
            ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Load mask for shape info
        _, mask, _ = load_volume(fid, split="test", load_cached_data=True)
        H, W = mask.shape

        # Reconstruction buffers
        prob_sum = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        with torch.no_grad():
            for volumes, _, coords in loader:
                volumes = volumes.to(device)
                logits = model(volumes)
                probs = torch.sigmoid(logits).cpu().numpy()
                coords = coords.numpy()

                batch_size = probs.shape[0]
                half_size = Config.PATCH_SIZE // 2

                for i in range(batch_size):
                    # coords[i] = [frag_idx, cy, cx]
                    cy = int(coords[i, 1])
                    cx = int(coords[i, 2])

                    y_start = cy - half_size
                    x_start = cx - half_size

                    y_start_clamped = max(0, y_start)
                    x_start_clamped = max(0, x_start)
                    y_end_clamped = min(H, y_start + Config.PATCH_SIZE)
                    x_end_clamped = min(W, x_start + Config.PATCH_SIZE)

                    if (
                        y_start_clamped >= y_end_clamped
                        or x_start_clamped >= x_end_clamped
                    ):
                        continue

                    py_start = y_start_clamped - y_start
                    px_start = x_start_clamped - x_start
                    py_end = py_start + (y_end_clamped - y_start_clamped)
                    px_end = px_start + (x_end_clamped - x_start_clamped)

                    prob_sum[
                        y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped
                    ] += probs[i, 0, py_start:py_end, px_start:px_end]
                    count_map[
                        y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped
                    ] += 1.0

        # Normalize
        count_map[count_map == 0] = 1.0
        final_probs = prob_sum / count_map

        # Apply Mask
        final_probs = final_probs * (mask > 0)

        # Binarize
        binary_pred = (final_probs > best_threshold).astype(np.uint8)

        # Encode
        rle = rle_encoding(binary_pred)
        submission_rows.append({"Id": fid, "Predicted": rle})

    # Save submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
