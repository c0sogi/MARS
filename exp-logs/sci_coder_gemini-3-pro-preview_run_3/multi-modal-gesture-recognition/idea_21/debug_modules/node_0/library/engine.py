import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import library modules
from library import config, utils, model, loss, dataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    metrics_sum = {}

    for features, labels, mask in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        # mask is not directly used in CascadedLoss as per provided loss.py,
        # but could be used if we customized the loss further.
        # Currently relying on class weights.

        optimizer.zero_grad()

        # Forward pass: returns list of outputs [p1, p2, p3]
        outputs = model(features)

        # Compute loss
        loss_val, batch_metrics = criterion(outputs, labels)

        # Backward pass
        loss_val.backward()

        # Gradient clipping (optional but good for RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss_val.item()

        # Accumulate metrics
        for k, v in batch_metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v

    avg_loss = running_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def evaluate(model, dataloader, criterion, device):
    """
    Evaluation loop for validation.
    """
    model.eval()
    running_loss = 0.0
    metrics_sum = {}

    # For accuracy calculation (Stage 3)
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels, mask in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss_val, batch_metrics = criterion(outputs, labels)

            running_loss += loss_val.item()

            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

            # Calculate accuracy on the final stage (Stage 3)
            # outputs[-1] shape: (B, C, T)
            final_logits = outputs[-1]
            preds = torch.argmax(final_logits, dim=1)  # (B, T)

            # Mask out padded regions for accuracy calculation if possible
            # Mask shape (B, T)
            mask = mask.to(device)
            valid_mask = mask > 0.5

            correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    avg_loss = running_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    accuracy = correct / total if total > 0 else 0.0

    avg_metrics["accuracy"] = accuracy

    return avg_loss, avg_metrics


def infer_sequence(model, raw_skeleton, raw_audio, device):
    """
    Performs sliding window inference on a single full sequence.
    Args:
        model: Trained model.
        raw_skeleton: (T, J, 3) numpy array.
        raw_audio: (T, MFCC) numpy array.
        device: torch device.
    Returns:
        final_preds: List of class IDs.
    """
    model.eval()

    # 1. Feature Extraction (Full Sequence)
    # Compute kinematics on the full sequence to ensure continuity
    # (T, InputDim)
    kinematics = utils.compute_kinematics(raw_skeleton)

    # Concatenate with audio
    # Ensure lengths match (audio might be slightly different due to extraction)
    min_len = min(kinematics.shape[0], raw_audio.shape[0])
    kinematics = kinematics[:min_len]
    audio = raw_audio[:min_len]

    features = np.concatenate([kinematics, audio], axis=1)  # (T, D)
    seq_len = features.shape[0]

    # 2. Sliding Window Setup
    window_size = config.WINDOW_SIZE
    stride = window_size // 2  # 50% overlap

    # Buffer to store probabilities: (T, NumClasses)
    # We accumulate probabilities from overlapping windows
    prob_map = np.zeros((seq_len, config.NUM_CLASSES), dtype=np.float32)
    count_map = np.zeros((seq_len, 1), dtype=np.float32)

    # Create windows
    windows = []
    indices = []

    start = 0
    while start < seq_len:
        end = start + window_size

        # Extract window
        win_feats = features[start:end]

        # Pad if needed
        if win_feats.shape[0] < window_size:
            pad_len = window_size - win_feats.shape[0]
            pad = np.zeros((pad_len, win_feats.shape[1]), dtype=np.float32)
            win_feats = np.concatenate([win_feats, pad], axis=0)

        windows.append(win_feats)
        indices.append((start, min(end, seq_len)))

        if end >= seq_len:
            break

        start += stride

    if not windows:
        return []

    # Batch processing
    batch_size = config.BATCH_SIZE

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_wins = windows[i : i + batch_size]
            batch_idxs = indices[i : i + batch_size]

            # Stack: (B, T, D)
            input_tensor = torch.from_numpy(np.array(batch_wins)).float().to(device)

            # Forward
            outputs = model(input_tensor)

            # Get Stage 3 output: (B, C, T)
            final_logits = outputs[-1]
            probs = torch.softmax(final_logits, dim=1).cpu().numpy()  # (B, C, T)

            # Transpose to (B, T, C) for easier mapping
            probs = probs.transpose(0, 2, 1)

            # Accumulate
            for b in range(len(batch_wins)):
                w_start, w_end = batch_idxs[b]
                valid_len = w_end - w_start

                # Add probabilities to map
                prob_map[w_start:w_end] += probs[b, :valid_len, :]
                count_map[w_start:w_end] += 1.0

    # Average probabilities
    # Avoid division by zero
    count_map[count_map == 0] = 1.0
    avg_probs = prob_map / count_map

    # Decode
    frame_preds = np.argmax(avg_probs, axis=1)

    # RLE
    final_gestures = utils.rle_encode(frame_preds)

    return final_gestures


def generate_submission(model, test_dataset, device, output_path):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")
    results = []

    # Access raw data directly from dataset object
    # test_dataset.skeletons, test_dataset.audios, test_dataset.sample_ids

    num_samples = len(test_dataset.sample_ids)

    for i in range(num_samples):
        sample_id = test_dataset.sample_ids[i]
        skel = test_dataset.skeletons[i]
        audio = test_dataset.audios[i]

        # Infer
        pred_labels = infer_sequence(model, skel, audio, device)

        # Format: SessionID,Label1,Label2...
        label_str = ",".join(map(str, pred_labels))
        results.append(f"{sample_id},{label_str}")

    # Write to file
    with open(output_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")


def run_training(
    epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, debug=config.DEBUG
):
    """
    Main execution function.
    """
    # 1. Setup
    config.set_seed()
    device = config.get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = dataset.GestureDataset(
        split="train", load_cached_data=True, augment=True, debug=debug
    )
    val_dataset = dataset.GestureDataset(
        split="val", load_cached_data=True, augment=False, debug=debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 3. Model & Optimization
    print("Initializing model...")
    net = model.RHKRN().to(device)
    criterion = loss.CascadedLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss, train_metrics = train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_metrics = evaluate(net, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc: {val_metrics['accuracy']:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), config.BEST_MODEL_PATH)
            # print(f"  Saved best model (Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Inference & Submission
    print("Loading best model for inference...")
    net.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))

    print("Loading test dataset...")
    test_dataset = dataset.GestureDataset(
        split="test", load_cached_data=True, augment=False, debug=debug
    )

    generate_submission(net, test_dataset, device, config.SUBMISSION_FILE)


# Expose the run function
def run():
    run_training()
