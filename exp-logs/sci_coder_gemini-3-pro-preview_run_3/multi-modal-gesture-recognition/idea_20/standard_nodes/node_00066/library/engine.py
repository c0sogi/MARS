import torch
import torch.nn.functional as F
import numpy as np
from library.config import NUM_CLASSES
from library.utils import decode_predictions, compute_levenshtein


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training using the CascadedSmoothLoss.

    Args:
        model (nn.Module): The LG-KRN model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): CascadedSmoothLoss function.
        optimizer (Optimizer): Adam optimizer.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: returns (logits_1, logits_2, logits_3)
        logits_list = model(inputs)

        # Compute cascaded loss
        loss = criterion(logits_list, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, dataset, criterion, device):
    """
    Evaluates the model on the validation set.
    Performs sliding window inference, aggregates probabilities, decodes sequences,
    and computes the Levenshtein distance error rate.

    Args:
        model (nn.Module): The LG-KRN model.
        dataloader (DataLoader): Validation data loader (must be shuffle=False).
        dataset (GestureDataset): The validation dataset object (for window mapping).
        criterion (nn.Module): Loss function.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (avg_loss, frame_accuracy, levenshtein_score)
    """
    model.eval()
    running_loss = 0.0
    correct_frames = 0
    total_frames = 0

    # Structures for sliding window aggregation
    # Map sample_id (str) -> accumulated probabilities (SeqLen, NumClasses)
    sample_probs = {}
    sample_counts = {}

    # Initialize buffers based on dataset metadata
    for i, sid in enumerate(dataset.sample_ids):
        seq_len = dataset.lengths[i]
        sample_probs[sid] = np.zeros((seq_len, NUM_CLASSES), dtype=np.float32)
        sample_counts[sid] = np.zeros((seq_len,), dtype=np.float32)

    current_window_idx = 0

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits_1, logits_2, logits_3 = model(inputs)

            # Compute Loss (Deep Supervision)
            loss = criterion([logits_1, logits_2, logits_3], targets)
            running_loss += loss.item()

            # Compute Frame-wise Accuracy on the final stage (Stage 3)
            # targets is (Batch, Time)
            preds_stage3 = torch.argmax(logits_3, dim=2)
            correct_frames += (preds_stage3 == targets).sum().item()
            total_frames += targets.numel()

            # Aggregate probabilities for Sequence Metrics
            # Use Softmax on Stage 3 output
            probs = F.softmax(logits_3, dim=2).cpu().numpy()
            batch_size = inputs.size(0)

            for b in range(batch_size):
                if current_window_idx >= len(dataset.windows):
                    break

                # Retrieve window mapping info
                # Note: This assumes dataloader is not shuffled and matches dataset.windows order
                s_idx, start, end = dataset.windows[current_window_idx]
                sid = dataset.sample_ids[s_idx]

                # Determine valid length in this window
                valid_len = end - start

                # Extract valid predictions
                window_preds = probs[b, :valid_len, :]

                # Accumulate
                sample_probs[sid][start:end] += window_preds
                sample_counts[sid][start:end] += 1.0

                current_window_idx += 1

    avg_loss = running_loss / len(dataloader)
    frame_accuracy = correct_frames / total_frames if total_frames > 0 else 0.0

    # --- Sequence Decoding and Metric Calculation ---
    predicted_sequences = []
    target_sequences = []

    for i, sid in enumerate(dataset.sample_ids):
        # 1. Average Probabilities
        counts = sample_counts[sid][:, None]
        counts[counts == 0] = 1.0  # Prevent division by zero
        avg_probs = sample_probs[sid] / counts

        # 2. Decode Prediction (RLE + Background filtering)
        pred_seq = decode_predictions(avg_probs)
        predicted_sequences.append(pred_seq)

        # 3. Get Ground Truth Sequence
        # dataset.labels is (N, MaxLen), we slice to valid length
        valid_len = dataset.lengths[i]
        gt_frames = dataset.labels[i, :valid_len]

        # Decode GT frames to sequence (RLE)
        gt_seq = decode_predictions(gt_frames)
        target_sequences.append(gt_seq)

    # Compute Levenshtein Distance (Error Rate)
    levenshtein_score = compute_levenshtein(predicted_sequences, target_sequences)

    return avg_loss, frame_accuracy, levenshtein_score
