import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import nltk
import time
import sys

from library.config import Config
from library.dataset import GestureDataset
from library.model import SA_AKN
from library.loss import CascadedLoss
from library.data_utils import save_submission

# Set seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


def calculate_levenshtein(predicted_seq, target_seq):
    """
    Calculates Levenshtein distance between two lists of integers.
    """
    return nltk.edit_distance(predicted_seq, target_seq)


def decode_predictions(frame_probs):
    """
    Decodes frame-wise probabilities into a list of gesture IDs.
    1. Argmax to get frame labels.
    2. Remove Background class (0).
    3. Run-Length Encoding (collapse consecutive duplicates).
    """
    # frame_probs: (T, NumClasses)
    frame_labels = np.argmax(frame_probs, axis=1)

    predicted_gestures = []
    last_label = -1

    for label in frame_labels:
        # Skip background class
        if label == Config.BACKGROUND_CLASS_ID:
            last_label = (
                -1
            )  # Reset so if gesture appears again after background, it's counted
            continue

        if label != last_label:
            predicted_gestures.append(int(label))
            last_label = label

    return predicted_gestures


def run_inference_on_sequence(model, features, device, window_size, stride):
    """
    Performs sliding window inference with temporal ensembling (averaging).

    Args:
        model: The trained SA_AKN model.
        features: Tensor (SeqLen, InputDim) - Normalized features.
        device: torch device.
        window_size: int
        stride: int

    Returns:
        avg_probs: Numpy array (SeqLen, NumClasses)
    """
    model.eval()
    seq_len = features.shape[0]
    num_classes = Config.NUM_CLASSES

    # Buffers for probability accumulation
    prob_buffer = torch.zeros((seq_len, num_classes), device=device)
    count_buffer = torch.zeros((seq_len, 1), device=device)

    # Prepare windows
    windows = []
    indices = []

    # Handle short sequences
    if seq_len <= window_size:
        # Pad features to window size
        pad_len = window_size - seq_len
        feat_pad = torch.zeros((pad_len, features.shape[1]), device=device)
        window = torch.cat([features, feat_pad], dim=0)
        windows.append(window)
        indices.append((0, seq_len))  # Valid range
    else:
        # Sliding window
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            windows.append(features[start:end])
            indices.append((start, end))

        # Handle remainder
        last_start = indices[-1][0]
        if last_start + window_size < seq_len:
            start = seq_len - window_size
            end = seq_len
            windows.append(features[start:end])
            indices.append((start, end))

    # Batch processing
    batch_size = Config.BATCH_SIZE

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = torch.stack(windows[i : i + batch_size])  # (B, Window, Dim)

            # Forward pass
            _, _, logits3 = model(batch_windows)
            probs3 = torch.softmax(logits3, dim=2)  # (B, Window, Classes)

            # Accumulate
            current_indices = indices[i : i + batch_size]
            for b_idx, (start, end) in enumerate(current_indices):
                # If sequence was short and padded, we only take valid part
                valid_len = end - start

                # Add to buffer
                # Note: For short sequences, indices are (0, seq_len), but window is padded.
                # We need to slice the output prob corresponding to valid data.
                if seq_len <= window_size:
                    prob_slice = probs3[b_idx, :seq_len, :]
                    prob_buffer[0:seq_len] += prob_slice
                    count_buffer[0:seq_len] += 1
                else:
                    prob_buffer[start:end] += probs3[b_idx]
                    count_buffer[start:end] += 1

    # Average
    # Avoid division by zero (though count should be >= 1)
    count_buffer[count_buffer == 0] = 1
    avg_probs = prob_buffer / count_buffer

    return avg_probs.cpu().numpy()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets, _, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass (Cascaded)
        logits1, logits2, logits3 = model(inputs)

        # Loss calculation
        loss, _ = criterion(logits1, logits2, logits3, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataset, device):
    """
    Evaluates the model on the full validation set using Levenshtein distance.
    Iterates over samples manually to perform sliding window inference on full sequences.
    """
    model.eval()

    total_distance = 0
    total_gestures = 0

    # Stride for inference (50% overlap)
    stride = Config.WINDOW_STRIDE_TEST
    window_size = Config.WINDOW_SIZE

    # Iterate over all samples in the dataset
    # We access processed features directly to reconstruct sequences
    num_samples = len(dataset.sample_ids)

    for i in range(num_samples):
        # Get full sequence features (Tensor)
        # dataset.processed_features is a list of numpy arrays
        features_np = dataset.processed_features[i]
        features = torch.from_numpy(features_np).float().to(device)

        # Run inference
        avg_probs = run_inference_on_sequence(
            model, features, device, window_size, stride
        )

        # Decode
        pred_seq = decode_predictions(avg_probs)

        # Get Ground Truth
        # dataset.raw_labels_meta[i] is a list of dicts: [{'id': 1, ...}, ...]
        gt_seq = [int(l["id"]) for l in dataset.raw_labels_meta[i]]

        # Calculate Metric
        dist = calculate_levenshtein(pred_seq, gt_seq)

        total_distance += dist
        total_gestures += len(gt_seq)

    # Error Rate
    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def train_model():
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Datasets
    # Load cached data if available, else process
    train_dataset = GestureDataset("train", load_cached_data=True)
    val_dataset = GestureDataset("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Model & Optimization
    model = SA_AKN().to(device)
    criterion = CascadedLoss().to(device)

    # "Use the Adam optimizer. We avoid AdamW"
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 3. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = evaluate(model, val_dataset, device)

        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Error Rate: {val_score:.6f} | "
            f"Time: {duration:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Error Rate: {best_score:.6f}")


def generate_submission():
    print("Generating submission...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Dataset
    test_dataset = GestureDataset("test", load_cached_data=True)

    # Load Best Model
    model = SA_AKN().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print("Error: Model file not found. Cannot generate submission.")
        return

    model.eval()

    predictions = []
    sample_ids = test_dataset.sample_ids

    stride = Config.WINDOW_STRIDE_TEST
    window_size = Config.WINDOW_SIZE

    for i in range(len(sample_ids)):
        features_np = test_dataset.processed_features[i]
        features = torch.from_numpy(features_np).float().to(device)

        # Inference
        avg_probs = run_inference_on_sequence(
            model, features, device, window_size, stride
        )

        # Decode
        pred_seq = decode_predictions(avg_probs)
        predictions.append(pred_seq)

    # Save
    save_submission(predictions, sample_ids, Config.SUBMISSION_PATH)


def main():
    # 1. Train
    train_model()

    # 2. Generate Submission
    generate_submission()


if __name__ == "__main__":
    main()
