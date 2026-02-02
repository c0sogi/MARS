import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.ndimage

from library.utils import set_seed, get_device
from library.loss import DeepSupervisionLoss
from library.model import GLT_CRCN
from library.data_loader import get_data, GestureDataset, collate_fn, DataLoaderConfig

# ==================================================================================================
# METRICS & POST-PROCESSING
# ==================================================================================================


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of integers.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return matrix[size_x - 1, size_y - 1]


def post_process_sequence(frame_preds, median_window=7):
    """
    Applies Median Filter with Nearest-Neighbor padding, then collapses repeats
    and removes background (class 0).
    """
    # 1. Median Filter with Nearest Padding
    # scipy.ndimage.median_filter supports mode='nearest'
    refined = scipy.ndimage.median_filter(
        frame_preds, size=median_window, mode="nearest"
    )

    # 2. Decode (Collapse repeats and remove background)
    decoded = []
    last = -1
    for p in refined:
        if p != last:
            if p != 0:  # Ignore background
                decoded.append(int(p))
            last = p

    return decoded


def evaluate_levenshtein(model, val_loader, device, median_window=7):
    """
    Computes the average Levenshtein error rate on the validation set.
    Metric = Sum(Levenshtein Distances) / Total Ground Truth Gestures
    """
    model.eval()
    total_dist = 0
    total_gt_gestures = 0

    # We need to map sample_ids to ground truth sequences
    # The loader provides batch data, but reconstructing full sequences requires care
    # We will process batch by batch.

    # Load GT metadata to get true sequences
    val_meta_path = os.path.join(DataLoaderConfig.METADATA_DIR, "val.csv")
    df_val = pd.read_csv(val_meta_path)
    # Parse labels
    df_val["labels"] = df_val["labels"].apply(
        lambda x: (
            [int(i) for i in str(x).split()]
            if pd.notna(x) and str(x).strip() != ""
            else []
        )
    )
    gt_map = dict(zip(df_val["sample_id"], df_val["labels"]))

    with torch.no_grad():
        for feats, _, mask, ids in val_loader:
            feats, mask = feats.to(device), mask.to(device)

            # Forward
            outputs = model(feats, mask)
            # Use Stage 3 output for prediction
            logits = outputs[-1]  # (B, 21, T)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # (B, T)

            # Iterate through batch
            for i, sample_id in enumerate(ids):
                # Get valid length
                valid_len = int(mask[i].sum().item())
                seq_preds = preds[i][:valid_len]

                # Post-process
                pred_seq = post_process_sequence(seq_preds, median_window=median_window)

                # Get GT
                gt_seq = gt_map.get(sample_id, [])

                # Compute Distance
                dist = levenshtein_distance(pred_seq, gt_seq)

                total_dist += dist
                total_gt_gestures += len(gt_seq)

    if total_gt_gestures == 0:
        return 0.0

    return total_dist / total_gt_gestures


# ==================================================================================================
# TRAINING
# ==================================================================================================


def train_model(
    epochs=40,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    smoothing_weight=0.15,
    patience=5,
    median_window=7,
    augment=True,
    working_dir="./working/idea_14",
    load_cached_data=True,
):
    set_seed(42)
    device = get_device()
    os.makedirs(working_dir, exist_ok=True)

    print(f"Initializing Training on {device}...")

    # 1. Data Loading
    train_data = get_data("train", load_cached_data=load_cached_data)
    val_data = get_data("val", load_cached_data=load_cached_data)

    train_dataset = GestureDataset(train_data, augment=augment)
    val_dataset = GestureDataset(val_data, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 2. Model Setup
    model = GLT_CRCN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = DeepSupervisionLoss(num_classes=21, smoothing_weight=smoothing_weight)

    # 3. Training Loop
    best_score = float("inf")
    counter = 0
    best_model_path = os.path.join(working_dir, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for feats, targets, mask, _ in train_loader:
            feats, targets, mask = feats.to(device), targets.to(device), mask.to(device)

            optimizer.zero_grad()
            outputs = model(feats, mask)
            loss = criterion(outputs, targets, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, targets, mask, _ in val_loader:
                feats, targets, mask = (
                    feats.to(device),
                    targets.to(device),
                    mask.to(device),
                )
                outputs = model(feats, mask)
                loss = criterion(outputs, targets, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Compute Metric
        val_lev_score = evaluate_levenshtein(model, val_loader, device, median_window)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Levenshtein: {val_lev_score:.6f}"
        )

        # Early Stopping based on Levenshtein Score
        if val_lev_score < best_score:
            best_score = val_lev_score
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> New best model saved.")
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training finished. Best Levenshtein Score: {best_score:.6f}")
    return best_model_path


# ==================================================================================================
# INFERENCE
# ==================================================================================================


def generate_submission(
    model_path, median_window=7, submission_dir="./submission", load_cached_data=True
):
    set_seed(42)
    device = get_device()

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Load Model
    model = GLT_CRCN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Test Data
    test_data = get_data("test", load_cached_data=load_cached_data)
    test_dataset = GestureDataset(test_data, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    results = []
    print("Generating predictions on test set...")

    with torch.no_grad():
        for feats, _, mask, ids in test_loader:
            feats, mask = feats.to(device), mask.to(device)

            outputs = model(feats, mask)
            logits = outputs[-1]  # Stage 3
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            for i in range(len(ids)):
                sample_id = ids[i]
                valid_len = int(mask[i].sum().item())
                seq_preds = preds[i][:valid_len]

                # Post-process
                decoded_seq = post_process_sequence(
                    seq_preds, median_window=median_window
                )

                # Format string
                pred_str = ",".join(map(str, decoded_seq))
                results.append(f"{sample_id},{pred_str}")

    # Save Submission
    os.makedirs(submission_dir, exist_ok=True)
    sub_file = os.path.join(submission_dir, "submission.csv")

    with open(sub_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {sub_file}")
