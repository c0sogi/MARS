import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_logger, calculate_weighted_log_loss
from library.trainers import FractureDetectionTrainer
from library.data import SequenceDataset, collate_fn_sequence


def pearson_corr(x, y):
    """Calculates Pearson correlation coefficient using NumPy."""
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return 0.0
    return num / den


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")
    logger.info("Starting Runfile Execution...")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        logger.error(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")
        return

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    logger.info(
        f"Loaded Metadata: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    # 3. Initialize Trainer
    trainer = FractureDetectionTrainer()

    # 4. Training Pipeline
    # Stage 1: Segmentation
    logger.info("=== Starting Stage 1: Segmentation ===")
    trainer.train_segmentor(train_df, val_df)

    # Stage 2: Encoder
    logger.info("=== Starting Stage 2: Encoder ===")
    trainer.train_encoder(train_df, val_df)

    # Stage 3: Aggregator (includes feature extraction)
    logger.info("=== Starting Stage 3: Aggregator ===")
    trainer.train_aggregator(train_df, val_df)

    # 5. Final Validation
    logger.info("=== Performing Final Validation ===")

    # Ensure aggregator is in eval mode
    trainer.aggregator.eval()

    # Create validation loader
    val_dataset = SequenceDataset(val_df, phase="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.TRAIN_RNN_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn_sequence,
    )

    all_preds = []
    all_targets = []
    sample_losses = []
    seq_lengths = []

    # BCE Loss for failure analysis (no reduction to get per-sample)
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_weights = Config.LOSS_WEIGHTS.to(Config.DEVICE)

    with torch.no_grad():
        for features, probs, targets, uids in val_loader:
            features = features.to(Config.DEVICE)
            probs = probs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward
            logits = trainer.aggregator(features, probs)
            preds = torch.sigmoid(logits)

            # Store for metric
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # --- Failure Analysis Data Collection ---
            # Calculate weighted loss for this batch
            # (B, 8)
            raw_loss = bce_criterion(logits, targets.float())
            weighted_loss = raw_loss * loss_weights

            # Average across classes (rows in the submission sense) for per-patient error
            patient_loss = weighted_loss.mean(dim=1)
            sample_losses.extend(patient_loss.cpu().numpy())

            # Sequence Length (approximate from padded probs)
            # probs is (B, T, 8). Sum over classes -> (B, T). >0 -> mask. Sum over T.
            lengths = (probs.sum(dim=2) > 0).long().sum(dim=1)
            seq_lengths.extend(lengths.cpu().numpy())

    # 6. Calculate & Print Metric
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    final_metric = calculate_weighted_log_loss(y_pred, y_true)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    logger.info("=== Failure Analysis ===")
    if len(sample_losses) > 1:
        # Correlation between Error Magnitude and Sequence Length (Input Feature)
        corr = pearson_corr(sample_losses, seq_lengths)
        print(f"Correlation between Error Magnitude and Sequence Length: {corr:.4f}")
    else:
        print("Insufficient validation samples for failure analysis.")

    # 8. Conditional Submission
    THRESHOLD = 0.9254394427010018
    if final_metric < THRESHOLD:
        logger.info(
            f"Metric ({final_metric:.6f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict_test_set(test_df)
    else:
        logger.info(
            f"Metric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
