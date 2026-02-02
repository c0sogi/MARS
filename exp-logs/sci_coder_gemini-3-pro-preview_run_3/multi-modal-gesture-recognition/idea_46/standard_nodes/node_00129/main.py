import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import nltk
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Paths, TrainConfig, DataConfig, ModelConfig
from library.utils import (
    set_seed,
    compute_levenshtein,
    LogSpaceSmoothingLoss,
    rle_encode,
    predictions_to_string,
)
from library.data_loader import get_dataloaders
from library.model import CKARFNet


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fast baseline overrides to ensure execution within time limits
    # The dataset is small (232 samples), but augmentation requires more epochs
    TrainConfig.EPOCHS = 40

    # ==========================================
    # 2. Data Loading
    # ==========================================
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=2,
        debug_size=None,  # Use full dataset
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = CKARFNet().to(device)

    # ==========================================
    # 4. Optimizer & Loss
    # ==========================================
    optimizer = optim.Adam(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY,
    )

    # Class weights: Background (0) gets 0.2, others get 1.0
    class_weights = torch.ones(DataConfig.NUM_CLASSES).to(device)
    class_weights[0] = TrainConfig.BACKGROUND_WEIGHT

    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_smooth = LogSpaceSmoothingLoss(
        lambda_weight=TrainConfig.SMOOTHING_LAMBDA,
        threshold=TrainConfig.SMOOTHING_THRESHOLD,
    )

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_levenshtein = float("inf")
    best_model_path = os.path.join(Paths.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, TrainConfig.EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass returns tuple: (logits1, logits2, logits3)
            outputs = model(features)

            batch_loss = 0.0

            # Calculate Cascaded Loss
            for stage_idx, logits in enumerate(outputs):
                # Cross Entropy expects (B, C, T)
                logits_permuted = logits.permute(0, 2, 1)
                loss_ce = criterion_ce(logits_permuted, labels)

                # Smoothing Loss expects LogProbs (B, T, C)
                log_probs = F.log_softmax(logits, dim=2)
                loss_sm = criterion_smooth(log_probs)

                # Weighted Stage Loss
                stage_loss = loss_ce + loss_sm
                batch_loss += stage_loss * TrainConfig.LOSS_STAGES_WEIGHTS[stage_idx]

            batch_loss.backward()
            optimizer.step()

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(features)
                final_logits = outputs[-1]  # Use final stage

                preds = torch.argmax(final_logits, dim=2)
                pred_seq = preds[0].cpu().numpy()
                target_seq = labels[0].cpu().numpy()

                pred_gestures = rle_encode(pred_seq, min_duration=5, background_class=0)
                target_gestures = rle_encode(
                    target_seq, min_duration=1, background_class=0
                )

                val_preds.append(pred_gestures)
                val_targets.append(target_gestures)

        val_score = compute_levenshtein(val_preds, val_targets)

        if val_score < best_levenshtein:
            best_levenshtein = val_score
            torch.save(model.state_dict(), best_model_path)

    # ==========================================
    # 6. Final Validation & Failure Analysis
    # ==========================================
    print(f"Final Validation Metric: {best_levenshtein}")

    # Load best model for analysis and submission
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    model.eval()

    # Failure Analysis: Correlate Error with Sequence Length
    val_errors = []
    val_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(features)
            final_logits = outputs[-1]
            preds = torch.argmax(final_logits, dim=2)

            pred_seq = preds[0].cpu().numpy()
            target_seq = labels[0].cpu().numpy()

            pred_gestures = rle_encode(pred_seq, min_duration=5, background_class=0)
            target_gestures = rle_encode(target_seq, min_duration=1, background_class=0)

            # Calculate individual Levenshtein distance
            dist = nltk.edit_distance(pred_gestures, target_gestures)

            val_errors.append(dist)
            val_lengths.append(features.shape[1])  # Time dimension

    if len(val_errors) > 1:
        corr, _ = pearsonr(val_errors, val_lengths)
        print(f"Correlation (Error vs Sequence Length): {corr}")

    # ==========================================
    # 7. Conditional Submission
    # ==========================================
    THRESHOLD = 0.1860643185298622

    if best_levenshtein < THRESHOLD:
        submission_lines = []
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                sample_id = batch["sample_id"][0]

                outputs = model(features)
                final_logits = outputs[-1]

                preds = torch.argmax(final_logits, dim=2)
                pred_seq = preds[0].cpu().numpy()

                pred_gestures = rle_encode(pred_seq, min_duration=5, background_class=0)

                line = predictions_to_string(sample_id, pred_gestures)
                submission_lines.append(line)

        with open(Paths.SUBMISSION_FILE, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")
        print(f"Submission saved to {Paths.SUBMISSION_FILE}")
    else:
        print(
            f"Validation metric {best_levenshtein} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
