import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

# Import from provided library files
from library.config import Config
from library.utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    optimize_f1_threshold,
    calculate_f1_samples,
)
from library.model import DilatedWideAndDeep
from library.data_processing import process_data, get_dataloaders
from library.train_eval import Trainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline
    Config.DEBUG = True
    Config.DEBUG_SIZE = 150000  # Limit samples for speed
    Config.EPOCHS = 3  # Limit epochs
    Config.BATCH_SIZE = 512  # Increase batch size for A100 efficiency

    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Processing
    # ==========================================
    print("\n--- Data Processing ---")
    (
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        test_ids,
        tokenizer_handler,
        target_encoder,
    ) = process_data(load_cached_data=True)

    print(f"Train shape: {train_tokens.shape}")
    print(f"Val shape: {val_tokens.shape}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        batch_size=Config.BATCH_SIZE,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- Model Initialization ---")
    num_classes = train_labels.shape[1]
    model = DilatedWideAndDeep(
        vocab_size=Config.VOCAB_SIZE,
        num_classes=num_classes,
        embed_dim=Config.EMBED_DIM,
        num_filters=Config.NUM_FILTERS,
        kernel_size=Config.KERNEL_SIZE,
        dilation_rates=Config.DILATION_RATES,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Trainer instance
    trainer = Trainer(model, optimizer, scheduler, criterion, device)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"\n--- Starting Training ({Config.EPOCHS} epochs) ---")
    best_f1 = 0.0
    best_thr = 0.5

    for epoch in range(1, Config.EPOCHS + 1):
        t0 = time.time()

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_loss, val_f1, thr = trainer.validate(val_loader)

        dt = time.time() - t0
        print(
            f"Epoch {epoch} | Time: {dt:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.5f}"
        )

        # Save Best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_thr = thr
            save_checkpoint(model, optimizer, epoch, val_f1, Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! F1: {best_f1:.5f}")

    # ==========================================
    # 5. Final Evaluation & Failure Analysis
    # ==========================================
    print("\n--- Final Evaluation & Failure Analysis ---")

    # Load best model
    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    model.eval()

    # Get predictions on Validation Set for Analysis
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for tokens, labels in val_loader:
            tokens = tokens.to(device)
            with autocast():
                logits = model(tokens)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy())
            val_targets.append(labels.numpy())

    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)

    # Recalculate metrics on full validation set
    # Using the best threshold found during training
    val_preds_binary = (val_probs >= best_thr).astype(int)
    final_f1 = calculate_f1_samples(val_targets, val_preds_binary)

    # REQUIRED PRINT
    print(f"Final Validation Metric: {final_f1}")

    # --- Failure Analysis ---
    print("Performing Failure Analysis...")

    # Calculate per-sample F1
    # F1 = 2TP / (2TP + FP + FN) = 2 * (y_true * y_pred) / (y_true + y_pred)
    tp = (val_targets * val_preds_binary).sum(axis=1)
    denominator = val_targets.sum(axis=1) + val_preds_binary.sum(axis=1)
    epsilon = 1e-9
    sample_f1s = (2 * tp) / (denominator + epsilon)

    # Error magnitude (1 - F1)
    errors = 1.0 - sample_f1s

    # Feature 1: Input Length (number of non-pad tokens)
    # We use the original val_tokens numpy array
    input_lengths = np.sum(val_tokens != 0, axis=1)

    # Feature 2: Number of Tags
    num_tags = np.sum(val_targets, axis=1)

    # Correlations
    corr_len = np.corrcoef(errors, input_lengths)[0, 1]
    corr_tags = np.corrcoef(errors, num_tags)[0, 1]

    print(f"Correlation (Error vs Input Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Num Tags): {corr_tags:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    SUBMISSION_THRESHOLD_SCORE = 0.33488

    if final_f1 > SUBMISSION_THRESHOLD_SCORE:
        print(
            f"\nMetric {final_f1:.5f} > {SUBMISSION_THRESHOLD_SCORE}. Generating submission..."
        )

        # Predict on Test
        test_probs = trainer.predict(test_loader)

        # Apply Threshold
        test_preds_binary = (test_probs >= best_thr).astype(int)

        # Inverse Transform
        pred_tags_list = target_encoder.inverse_transform(test_preds_binary)
        pred_tags_str = [" ".join(tags) for tags in pred_tags_list]

        # Save
        submission = pd.DataFrame({"Id": test_ids, "Tags": pred_tags_str})
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_f1:.5f} <= {SUBMISSION_THRESHOLD_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
