import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.signal import medfilt

# Import from library
from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_CLASSES,
    SEED,
    MEDIAN_FILTER_KERNEL,
)
from library.utils import set_seed, compute_error_rate, levenshtein_distance
from library.model import SSG_CRCN
from library.loss import CombinedLoss
from library.data_loader import prepare_dataset, GestureDataset, collate_fn
from library.train import train_epoch, decode_sequence

# Hyperparameters for Fast Baseline
# Overriding config defaults to ensure execution within time limits
FAST_EPOCHS = 25


def run_pipeline():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("Loading datasets...")
    # Train
    train_pos, train_aud, train_lbl, train_bnd, _ = prepare_dataset(
        TRAIN_METADATA_PATH, "train_data", load_cached_data=True
    )
    # Validation
    val_pos, val_aud, val_lbl, val_bnd, val_ids = prepare_dataset(
        VAL_METADATA_PATH, "val_data", load_cached_data=True
    )

    # Create DataLoaders
    # Enable augmentation for training
    train_dataset = GestureDataset(
        train_pos, train_aud, train_lbl, train_bnd, augment=True
    )
    val_dataset = GestureDataset(val_pos, val_aud, val_lbl, val_bnd, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = SSG_CRCN().to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Training Loop
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    for epoch in range(FAST_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate (Loss only for Early Stopping)
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for feats, lbls, bnds, mask in val_loader:
                feats, lbls, bnds, mask = (
                    feats.to(device),
                    lbls.to(device),
                    bnds.to(device),
                    mask.to(device),
                )
                outputs = model(feats, mask)
                loss, _ = criterion(outputs, lbls, bnds, mask)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training complete.")

    # 5. Detailed Validation & Metric Calculation
    print("Performing final validation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    # Store per-sample data for failure analysis
    sample_errors = []
    sample_lengths = []
    sample_num_gestures = []

    with torch.no_grad():
        # Iterate over validation set again to get predictions
        # Use batch_size=1 for simpler per-sample analysis logic, or process batch
        # We'll use the existing loader and process batch-wise
        for i_batch, (feats, lbls, bnds, mask) in enumerate(val_loader):
            feats, mask = feats.to(device), mask.to(device)

            outputs = model(feats, mask)
            final_stage_out = outputs[-1]
            cls_logits = final_stage_out[:, :NUM_CLASSES, :]
            probs = F.softmax(cls_logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # (B, T)

            targets_np = lbls.numpy()  # (B, T)
            mask_np = mask.cpu().numpy()  # (B, T)

            for i in range(preds.shape[0]):
                valid_len = int(mask_np[i].sum())

                # Raw sequences
                p_seq = preds[i, :valid_len]
                t_seq = targets_np[i, :valid_len]

                # Post-processing
                if MEDIAN_FILTER_KERNEL > 1:
                    p_seq = medfilt(p_seq, kernel_size=MEDIAN_FILTER_KERNEL)

                # Decode
                pred_decoded = decode_sequence(p_seq)
                target_decoded = decode_sequence(t_seq)

                all_preds.append(pred_decoded)
                all_targets.append(target_decoded)

                # Failure Analysis Data
                dist = levenshtein_distance(pred_decoded, target_decoded)
                sample_errors.append(dist)
                sample_lengths.append(valid_len)
                sample_num_gestures.append(len(target_decoded))

    # Compute Final Metric
    final_metric = compute_error_rate(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(sample_errors) > 1:
        # Correlation: Error vs Sequence Length
        corr_len, _ = pearsonr(sample_lengths, sample_errors)
        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")

        # Correlation: Error vs Number of Gestures
        corr_num, _ = pearsonr(sample_num_gestures, sample_errors)
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Average Error
        print(f"Average Levenshtein Distance per sample: {np.mean(sample_errors):.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # 7. Submission
    THRESHOLD = 0.06789606035205364
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_pos, test_aud, test_lbl, test_bnd, test_ids = prepare_dataset(
            TEST_METADATA_PATH, "test_data", load_cached_data=True
        )

        test_dataset = GestureDataset(
            test_pos, test_aud, test_lbl, test_bnd, augment=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
        )

        predictions = []
        with torch.no_grad():
            for feats, _, _, mask in test_loader:
                feats, mask = feats.to(device), mask.to(device)

                outputs = model(feats, mask)
                final_stage_out = outputs[-1]
                cls_logits = final_stage_out[:, :NUM_CLASSES, :]
                probs = F.softmax(cls_logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()[0]

                valid_len = int(mask[0].sum().item())
                preds = preds[:valid_len]

                if MEDIAN_FILTER_KERNEL > 1:
                    preds = medfilt(preds, kernel_size=MEDIAN_FILTER_KERNEL)

                decoded_seq = decode_sequence(preds)
                predictions.append(",".join(map(str, decoded_seq)))

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for sid, pred in zip(test_ids, predictions):
                f.write(f"{sid},{pred}\n")
        print(f"Submission saved to {submission_path}")
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
