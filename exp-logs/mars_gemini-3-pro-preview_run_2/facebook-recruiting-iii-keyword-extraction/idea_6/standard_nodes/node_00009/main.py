import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Import provided library modules
from library.config import (
    DEVICE,
    OUTPUT_DIR,
    SUBMISSION_PATH,
    LR,
    EPOCHS,
    BATCH_SIZE,
    SEED,
    NUM_TAGS,
    NUM_WORKERS,
)
from library.data_processor import prepare_data
from library.dataset import StackExchangeDataset, collate_fn
from library.model import WideAndDeepModel, FocalLoss
from library.trainer import train_one_epoch, evaluate
from library.inference import find_best_threshold, generate_submission


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    print("Loading data...")
    # Load cached data (or process if not exists)
    data = prepare_data(load_cached_data=True)
    (
        X_wide_train,
        X_deep_train,
        y_train,
        X_wide_val,
        X_deep_val,
        y_val,
        X_wide_test,
        X_deep_test,
        test_ids,
        preprocessor,
    ) = data

    # 3. Subsample Training Data (Fast Baseline Strategy)
    # We limit training to 200,000 samples to ensure execution within 2 hours.
    # Validation is performed on the full set for accurate metrics.
    N_TRAIN_SAMPLES = 200000
    if X_deep_train.shape[0] > N_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {X_deep_train.shape[0]} to {N_TRAIN_SAMPLES} for fast baseline..."
        )
        # Use fixed seed for shuffling indices
        rng = np.random.RandomState(SEED)
        indices = rng.choice(X_deep_train.shape[0], N_TRAIN_SAMPLES, replace=False)

        X_wide_train = X_wide_train[indices]
        X_deep_train = X_deep_train[indices]
        y_train = y_train[indices]

    # Create Datasets
    train_dataset = StackExchangeDataset(X_wide_train, X_deep_train, y_train)
    val_dataset = StackExchangeDataset(X_wide_val, X_deep_val, y_val)

    # Create DataLoaders
    # Pin memory helps with transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("Initializing WideAndDeepModel...")
    model = WideAndDeepModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # 5. Training Loop
    best_val_f1 = -1.0
    best_model_state = None

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )

        # Validate
        val_loss, val_probs, val_targets = evaluate(
            model, val_loader, criterion, DEVICE
        )

        # Quick F1 check with default threshold 0.5
        val_preds_default = (val_probs > 0.5).astype(int)
        val_f1_default = f1_score(
            val_targets, val_preds_default, average="samples", zero_division=0
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F1 (0.5): {val_f1_default:.6f}"
        )

        # Save best model state (using F1 as criteria)
        if val_f1_default > best_val_f1:
            best_val_f1 = val_f1_default
            best_model_state = model.state_dict()

    # Load best model for final evaluation
    if best_model_state is not None:
        print("Loading best model state...")
        model.load_state_dict(best_model_state)

    # 6. Threshold Optimization & Final Metric
    print("Optimizing threshold on full validation set...")
    _, val_probs, val_targets = evaluate(model, val_loader, criterion, DEVICE)

    # Use the robust percentile-based threshold finder from inference.py
    best_thresh, best_score = find_best_threshold(val_probs, val_targets)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {best_score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Compute per-sample F1 score
    val_preds = (val_probs > best_thresh).astype(int)

    # Vectorized calculation of Sample F1
    tp = (val_preds * val_targets).sum(axis=1)
    denom = val_preds.sum(axis=1) + val_targets.sum(axis=1)

    # F1 = 2*TP / (Preds + Targets). Handle division by zero (0/0 -> 1.0)
    f1_samples = np.divide(
        2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0
    )
    f1_samples[denom == 0] = 1.0

    error_magnitude = 1.0 - f1_samples

    # Feature 1: Input Text Length (Number of non-pad tokens in Deep sequence)
    # X_deep_val is (N, max_len)
    text_lengths = (X_deep_val != 0).sum(axis=1)

    # Feature 2: Number of Ground Truth Tags
    tag_counts = val_targets.sum(axis=1)

    # Correlations
    corr_len = np.corrcoef(error_magnitude, text_lengths)[0, 1]
    corr_tags = np.corrcoef(error_magnitude, tag_counts)[0, 1]

    print(f"Correlation (Error vs Text Length): {corr_len:.6f}")
    print(f"Correlation (Error vs Tag Count): {corr_tags:.6f}")

    # 8. Submission
    SUBMISSION_THRESHOLD = 0.0542101508997596

    if best_score > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric {best_score} exceeds threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        # Prepare Test Loader
        test_dataset = StackExchangeDataset(X_wide_test, X_deep_test, y=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Generate and Save
        generate_submission(
            model, test_loader, test_ids, preprocessor, best_thresh, DEVICE
        )
    else:
        print(
            f"\nValidation metric {best_score} does not exceed threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
