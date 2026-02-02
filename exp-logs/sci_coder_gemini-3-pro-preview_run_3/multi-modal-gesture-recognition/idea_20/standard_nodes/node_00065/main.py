import os
import sys
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import nltk
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import (
    SEED,
    DEVICE,
    NUM_WORKERS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    NUM_CLASSES,
)
from library.utils import setup_logger, decode_predictions, generate_submission_file
from library.dataset import GestureDataset
from library.modules import LGKRN
from library.loss import CascadedSmoothLoss
from library.engine import train_one_epoch, evaluate


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_predictions_and_metrics(model, dataloader, dataset, device, is_test=False):
    """
    Runs inference to get per-sample predictions and metrics (for failure analysis).
    """
    model.eval()

    # Structures for sliding window aggregation
    sample_probs = {}
    sample_counts = {}

    # Initialize buffers
    for i, sid in enumerate(dataset.sample_ids):
        seq_len = dataset.lengths[i]
        sample_probs[sid] = np.zeros((seq_len, NUM_CLASSES), dtype=np.float32)
        sample_counts[sid] = np.zeros((seq_len,), dtype=np.float32)

    current_window_idx = 0

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)

            # Forward pass (Stage 3 is the final output)
            _, _, logits_3 = model(inputs)
            probs = F.softmax(logits_3, dim=2).cpu().numpy()

            batch_size = inputs.size(0)

            for b in range(batch_size):
                if current_window_idx >= len(dataset.windows):
                    break

                s_idx, start, end = dataset.windows[current_window_idx]
                sid = dataset.sample_ids[s_idx]

                # Valid length in this window
                valid_len = end - start

                window_preds = probs[b, :valid_len, :]

                sample_probs[sid][start:end] += window_preds
                sample_counts[sid][start:end] += 1.0

                current_window_idx += 1

    # Decode and compute metrics
    final_predictions = []
    final_ids = []
    lev_distances = []
    seq_lengths = []

    for i, sid in enumerate(dataset.sample_ids):
        counts = sample_counts[sid][:, None]
        counts[counts == 0] = 1.0
        avg_probs = sample_probs[sid] / counts

        pred_seq = decode_predictions(avg_probs)
        final_predictions.append(pred_seq)
        final_ids.append(sid)

        # Compute metrics only if ground truth is available and not testing
        if not is_test:
            valid_len = dataset.lengths[i]
            gt_frames = dataset.labels[i, :valid_len]
            gt_seq = decode_predictions(gt_frames)

            dist = nltk.edit_distance(pred_seq, gt_seq)
            lev_distances.append(dist)
            seq_lengths.append(valid_len)

    return final_predictions, final_ids, lev_distances, seq_lengths


def main():
    # 1. Initialization
    set_seed(SEED)
    logger = setup_logger(os.path.join(os.path.dirname(MODEL_SAVE_PATH), "run.log"))
    logger.info("Starting LG-KRN Runfile Execution")

    # 2. Data Loading
    logger.info("Loading Datasets...")
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH, "train", load_cache=True, augment=True
    )
    val_dataset = GestureDataset(
        VAL_METADATA_PATH, "val", load_cache=True, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Setup
    model = LGKRN().to(DEVICE)
    criterion = CascadedSmoothLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Training Loop
    logger.info("Starting Training Loop...")
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_acc, val_metric = evaluate(
            model, val_loader, val_dataset, criterion, DEVICE
        )

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f} | Val Metric: {val_metric:.5f}"
        )

        # Checkpointing
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"  > New best model saved (Metric: {best_metric:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"  > Early stopping triggered at epoch {epoch}")
                break

    # 5. Final Evaluation & Failure Analysis
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # Re-compute metric to ensure exact value is printed
    _, _, final_metric = evaluate(model, val_loader, val_dataset, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    _, _, errors, lengths = get_predictions_and_metrics(
        model, val_loader, val_dataset, DEVICE, is_test=False
    )

    if len(errors) > 0 and len(lengths) > 0:
        # Calculate correlation between Error Magnitude and Sequence Length
        correlation = np.corrcoef(errors, lengths)[0, 1]
        print(
            f"Correlation between Error Magnitude and Sequence Length: {correlation:.4f}"
        )

    # 6. Submission Generation
    if final_metric < 0.2251:
        logger.info(f"Metric {final_metric:.5f} < 0.2251. Generating submission...")

        test_dataset = GestureDataset(
            TEST_METADATA_PATH, "test", load_cache=True, augment=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        predictions, sample_ids, _, _ = get_predictions_and_metrics(
            model, test_loader, test_dataset, DEVICE, is_test=True
        )

        generate_submission_file(predictions, sample_ids, SUBMISSION_PATH)
        logger.info(f"Submission saved to {SUBMISSION_PATH}")
    else:
        logger.info(
            f"Metric {final_metric:.5f} is not below threshold 0.2251. Skipping submission."
        )


if __name__ == "__main__":
    main()
