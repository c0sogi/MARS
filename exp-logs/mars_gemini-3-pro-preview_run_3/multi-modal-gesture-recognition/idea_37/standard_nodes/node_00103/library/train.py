import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.utils import (
    set_seed,
    levenshtein_distance,
    run_length_encoding,
    TruncatedMSELoss,
)
from library.data_loader import get_dataloaders
from library.model import RHCKN


def get_loss_criterion(device):
    """
    Constructs the composite loss function components.
    """
    # 1. Weighted Cross Entropy
    # Class 0 (Background) gets weight 0.2, others 1.0
    weights = torch.ones(Config.NUM_CLASSES, device=device)
    weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT
    ce_criterion = nn.NLLLoss(weight=weights, reduction="mean")

    # 2. Truncated MSE for Smoothing
    smooth_criterion = TruncatedMSELoss(threshold=Config.SMOOTHING_THRESHOLD)

    return ce_criterion, smooth_criterion


def compute_loss(outputs, targets, ce_criterion, smooth_criterion):
    """
    Computes the Deep Supervision loss:
    L = L_CE(s1) + L_CE(s2) + L_CE(s3) + L_Smooth(s2) + L_Smooth(s3)
    """
    # Targets shape: (Batch, Time)
    # Outputs: Dict of (Batch, Time, Classes) log_probs

    loss = 0.0

    # --- Stage 1 ---
    # Permute for NLLLoss: (Batch, Classes, Time)
    s1_log_probs = outputs["stage1"].transpose(1, 2)
    loss += ce_criterion(s1_log_probs, targets)

    # --- Stage 2 ---
    s2_out = outputs["stage2"]
    s2_log_probs = s2_out.transpose(1, 2)
    loss += ce_criterion(s2_log_probs, targets)
    loss += Config.SMOOTHING_LOSS_WEIGHT * smooth_criterion(s2_out)

    # --- Stage 3 ---
    s3_out = outputs["stage3"]
    s3_log_probs = s3_out.transpose(1, 2)
    loss += ce_criterion(s3_log_probs, targets)
    loss += Config.SMOOTHING_LOSS_WEIGHT * smooth_criterion(s3_out)

    return loss


def train_one_epoch(model, loader, optimizer, ce_criterion, smooth_criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        features = batch["feature"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)

        # Compute loss
        loss = compute_loss(outputs, labels, ce_criterion, smooth_criterion)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def aggregate_predictions(loader, model, device):
    """
    Runs inference on the loader and aggregates sliding window predictions
    into full-sequence probabilities.
    Returns:
        dict: {sample_idx: (aggregated_probs_tensor, sample_id_str)}
    """
    model.eval()

    # Storage for accumulating probabilities: sample_idx -> (sum_probs, count_counts)
    # We need to handle variable lengths.
    sample_data = {}

    # Access dataset to get window info
    dataset = loader.dataset

    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            window_indices = batch["window_idx"]

            # Forward pass
            outputs = model(features)
            # Use Stage 3 for final prediction
            # Exp to get probabilities from log_softmax
            probs = torch.exp(outputs["stage3"]).cpu().numpy()  # (B, T, C)

            for i, w_idx in enumerate(window_indices):
                w_idx = w_idx.item()
                # Retrieve window metadata
                s_idx, start, end = dataset.windows[w_idx]

                # Get sample info if not initialized
                if s_idx not in sample_data:
                    # We need the total length of the sample to initialize the buffer.
                    # We can get it from the dataset samples list.
                    # Note: dataset.samples[s_idx]['skeleton'] shape is (TotalFrames, ...)
                    total_frames = dataset.samples[s_idx]["skeleton"].shape[0]
                    # Get sample ID string
                    # We need to look up the sample_id from the dataframe
                    sample_id = dataset.df.iloc[s_idx]["sample_id"]

                    # Initialize buffers: (Time, Classes)
                    buffer_probs = np.zeros(
                        (total_frames, Config.NUM_CLASSES), dtype=np.float32
                    )
                    buffer_counts = np.zeros((total_frames, 1), dtype=np.float32)

                    sample_data[s_idx] = {
                        "probs": buffer_probs,
                        "counts": buffer_counts,
                        "id": sample_id,
                        "gt_labels": dataset.samples[s_idx][
                            "labels"
                        ],  # Keep GT for validation
                    }

                # Add predictions
                # The window output might be padded or cut, but here it corresponds to [start:end]
                # The model output is fixed size Config.WINDOW_SIZE.
                # If the actual window was shorter (at the end), we need to be careful.
                # In data_loader.py, windows are padded or handled.
                # The __getitem__ copies available data into a fixed size buffer.
                # We should only aggregate the valid part.

                valid_len = min(
                    Config.WINDOW_SIZE, sample_data[s_idx]["probs"].shape[0] - start
                )

                current_probs = probs[i, :valid_len, :]

                sample_data[s_idx]["probs"][start : start + valid_len] += current_probs
                sample_data[s_idx]["counts"][start : start + valid_len] += 1.0

    # Normalize
    results = {}
    for s_idx, data in sample_data.items():
        # Avoid division by zero (though counts should be >= 1 for visited frames)
        counts = data["counts"]
        counts[counts == 0] = 1.0
        avg_probs = data["probs"] / counts
        results[s_idx] = {
            "probs": avg_probs,
            "id": data["id"],
            "gt_labels": data["gt_labels"],
        }

    return results


def validate(model, loader, device):
    """
    Validates the model by computing the Levenshtein distance on full sequences.
    """
    aggregated = aggregate_predictions(loader, model, device)

    total_dist = 0
    total_gestures = 0

    for s_idx, data in aggregated.items():
        # Get frame-wise predictions
        frame_preds = np.argmax(data["probs"], axis=1)

        # Decode to sequence
        pred_seq = run_length_encoding(
            frame_preds,
            min_duration=Config.MIN_GESTURE_DURATION,
            background_class=Config.BACKGROUND_CLASS_ID,
        )

        # Get Ground Truth sequence
        # GT labels are frame-wise. Convert to sequence.
        # Note: The raw labels in dataset are 0 for background, 1-20 for gestures.
        # We can use the same RLE function, but with min_duration=1 to capture all annotations accurately,
        # or rely on the parsed metadata. The dataset stores frame-wise labels.
        # Let's use RLE on GT frame labels to be consistent.
        gt_frame_labels = data["gt_labels"]
        gt_seq = run_length_encoding(
            gt_frame_labels,
            min_duration=1,  # GT shouldn't be filtered aggressively
            background_class=Config.BACKGROUND_CLASS_ID,
        )

        # Compute Distance
        dist = levenshtein_distance(pred_seq, gt_seq)
        total_dist += dist
        total_gestures += len(gt_seq)

    # Avoid division by zero
    if total_gestures == 0:
        return 0.0

    return total_dist / total_gestures


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    aggregated = aggregate_predictions(loader, model, device)

    submission_lines = []

    # Sort by sample ID to ensure order (optional but good)
    sorted_indices = sorted(aggregated.keys(), key=lambda k: aggregated[k]["id"])

    for s_idx in sorted_indices:
        data = aggregated[s_idx]
        sample_id = data["id"]

        # Get frame-wise predictions
        frame_preds = np.argmax(data["probs"], axis=1)

        # Decode
        pred_seq = run_length_encoding(
            frame_preds,
            min_duration=Config.MIN_GESTURE_DURATION,
            background_class=Config.BACKGROUND_CLASS_ID,
        )

        # Format: SessionID,label1,label2,...
        labels_str = ",".join(map(str, pred_seq))
        line = f"{sample_id},{labels_str}"
        submission_lines.append(line)

    # Write to file
    with open(output_path, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Loading
    # load_cached_data=True will try to load pre-processed .npz files
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RHCKN().to(device)

    # 4. Optimization
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    ce_criterion, smooth_criterion = get_loss_criterion(device)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, smooth_criterion, device
        )

        # Validate
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Levenshtein: {val_score:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print("  -> New best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score:.6f}")

    # 6. Inference on Test Set
    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No model file found. Using current model state.")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
