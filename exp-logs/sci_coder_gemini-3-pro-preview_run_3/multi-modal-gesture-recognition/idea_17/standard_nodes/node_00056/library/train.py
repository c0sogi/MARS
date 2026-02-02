import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, compute_levenshtein, LogSpaceSmoothingLoss
from library.data_loader import get_data_loaders, load_data, KinematicAugmentor
from library.model import RS_KRN


def decode_predictions(dense_labels):
    """
    Decodes dense frame-wise labels into a list of gesture IDs.
    1. Collapses consecutive duplicates (Run-Length Encoding).
    2. Removes background class (0).
    """
    if len(dense_labels) == 0:
        return []

    # Collapse duplicates
    collapsed = [dense_labels[0]]
    for i in range(1, len(dense_labels)):
        if dense_labels[i] != dense_labels[i - 1]:
            collapsed.append(dense_labels[i])

    # Remove background (0)
    final_sequence = [x for x in collapsed if x != 0]
    return final_sequence


def train_epoch(model, loader, optimizer, criterion_ce, criterion_smooth, device):
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: returns list [P0, P1, P2, ...]
        outputs = model(features)

        loss = 0.0

        # Recursive Cascaded Loss
        for i, logits in enumerate(outputs):
            # 1. Cross Entropy Loss (Weighted)
            # logits: (Batch, Time, Classes) -> Permute for CE: (Batch, Classes, Time)
            ce_loss = criterion_ce(logits.permute(0, 2, 1), labels)
            loss += ce_loss

            # 2. Log-Space Smoothing Loss (Only for refinement stages P1, P2...)
            # We skip the encoder output (index 0)
            if i > 0:
                log_probs = F.log_softmax(logits, dim=2)
                smooth_loss = criterion_smooth(log_probs)
                loss += smooth_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, val_data_list, device):
    """
    Performs validation on full sequences using sliding window inference.
    """
    model.eval()

    all_preds = []
    all_targets = []

    window_size = Config.WINDOW_SIZE
    stride = Config.TEST_STRIDE

    with torch.no_grad():
        for sample in val_data_list:
            # Prepare Features
            # 1. Kinematics
            raw_skel = sample["skeleton"]  # (T, 20, 3)
            kinematics = KinematicAugmentor.compute_kinematics(raw_skel)  # (T, 180)

            # 2. Audio
            audio = sample["audio"]  # (T, 13)

            # 3. Concat
            full_features = np.concatenate([kinematics, audio], axis=1)  # (T, 193)
            T = full_features.shape[0]

            # Buffer for probability accumulation
            # We accumulate probabilities for the final stage output
            prob_buffer = np.zeros((T, Config.NUM_CLASSES), dtype=np.float32)
            count_buffer = np.zeros((T, 1), dtype=np.float32)

            # Sliding Window Inference
            # Handle short sequences by padding if necessary (though val seqs are usually long)
            if T < window_size:
                # Pad
                pad_len = window_size - T
                feat_padded = np.pad(full_features, ((0, pad_len), (0, 0)), mode="edge")
                feat_tensor = (
                    torch.from_numpy(feat_padded).float().unsqueeze(0).to(device)
                )

                outputs = model(feat_tensor)
                final_logits = outputs[-1]  # Take last refinement stage
                probs = (
                    torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)
                )  # (Window, Classes)

                # Add to buffer (only valid part)
                prob_buffer += probs[:T]
                count_buffer += 1.0
            else:
                # Slide
                for start in range(0, T - window_size + 1, stride):
                    end = start + window_size
                    window_feat = full_features[start:end]

                    feat_tensor = (
                        torch.from_numpy(window_feat).float().unsqueeze(0).to(device)
                    )
                    outputs = model(feat_tensor)
                    final_logits = outputs[-1]
                    probs = torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)

                    prob_buffer[start:end] += probs
                    count_buffer[start:end] += 1.0

                # Handle last window if not covered exactly
                last_start = T - window_size
                if last_start > 0 and (last_start % stride != 0):
                    window_feat = full_features[last_start:T]
                    feat_tensor = (
                        torch.from_numpy(window_feat).float().unsqueeze(0).to(device)
                    )
                    outputs = model(feat_tensor)
                    final_logits = outputs[-1]
                    probs = torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)

                    prob_buffer[last_start:T] += probs
                    count_buffer[last_start:T] += 1.0

            # Average probabilities
            # Avoid division by zero
            count_buffer[count_buffer == 0] = 1.0
            avg_probs = prob_buffer / count_buffer

            # Decode
            pred_dense = np.argmax(avg_probs, axis=1)
            pred_seq = decode_predictions(pred_dense)

            # Ground Truth
            gt_dense = sample["labels"]
            gt_seq = decode_predictions(gt_dense)

            all_preds.append(pred_seq)
            all_targets.append(gt_seq)

    # Compute Metric
    score = compute_levenshtein(all_preds, all_targets)
    return score


def train_model():
    # 1. Configuration & Setup
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing Data Loaders...")
    train_loader, _ = get_data_loaders(batch_size=Config.BATCH_SIZE)
    # We load raw validation data for full sequence evaluation
    val_data_list = load_data("val", load_cached_data=True)

    # 3. Model Initialization
    print("Initializing RS-KRN Model...")
    model = RS_KRN().to(device)

    # 4. Loss & Optimizer
    # Weighted Cross Entropy
    class_weights = Config.CLASS_WEIGHTS.to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)

    # Smoothing Loss
    criterion_smooth = LogSpaceSmoothingLoss(weight=Config.MSE_SMOOTHING_WEIGHT)

    # Optimizer (Adam, no weight decay as per prompt recommendation against AdamW for this)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion_ce, criterion_smooth, device
        )

        # Validate
        val_score = validate(model, val_data_list, device)

        # Logging
        # Print full precision as requested
        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Levenshtein: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    print(f"Training Complete. Best Validation Score: {best_score}")
