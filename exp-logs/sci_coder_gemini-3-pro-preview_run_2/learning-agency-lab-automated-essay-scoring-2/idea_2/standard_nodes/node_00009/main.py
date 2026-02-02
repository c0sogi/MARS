import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

# Ensure library modules are importable
sys.path.append(".")

from library.config import Config, seed_everything
from library.data_loader import get_dataloaders
from library.modeling import EssayRegressor
from library.engine import train_fn, eval_fn
from library.utils import compute_qwk, optimize_thresholds, apply_thresholds


def run_orchestration():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Initializing orchestration...")

    # 2. Data Loading
    # Using cached data as per instructions to speed up loading
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = EssayRegressor(pretrained=True).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")

    # 4. Training Loop
    best_qwk = -np.inf
    best_model_state = None

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, device, scheduler, criterion, scaler
        )

        # Validate (Continuous predictions)
        val_preds, val_loss = eval_fn(model, val_loader, device, criterion)

        # Compute Metric for Monitoring (using standard rounding)
        # Note: val_loader is not shuffled, so labels align with predictions
        val_labels = np.array(val_loader.dataset.labels)
        val_preds_rounded = np.clip(np.round(val_preds), 1, 6).astype(int)
        val_labels_int = val_labels.astype(int)

        qwk = compute_qwk(val_labels_int, val_preds_rounded)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | QWK (Rounded): {qwk:.5f}"
        )

        # Save Best Model State
        if qwk > best_qwk:
            best_qwk = qwk
            best_model_state = model.state_dict()

    print(f"Training finished. Best QWK (Rounded) during training: {best_qwk:.5f}")

    # 5. Best Model Evaluation & Threshold Optimization
    print("Loading best model for final validation...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Get raw continuous predictions on validation set
    val_preds, _ = eval_fn(model, val_loader, device)
    val_labels = np.array(val_loader.dataset.labels).astype(int)

    # Optimize Thresholds
    print("Optimizing thresholds...")
    if Config.OPTIMIZE_THRESHOLDS:
        best_thresholds = optimize_thresholds(val_labels, val_preds)
    else:
        best_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    print(f"Optimal Thresholds: {best_thresholds}")

    # Apply Thresholds
    val_preds_opt = apply_thresholds(val_preds, best_thresholds)

    # Final Metric
    final_metric = compute_qwk(val_labels, val_preds_opt)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude (absolute difference between continuous prediction and true label)
    # This captures the regression error before discretization
    error_magnitude = np.abs(val_labels - val_preds)

    # Input Feature: Text Length (derived from attention mask)
    # val_loader.dataset.attention_mask is a list of lists
    lengths = [sum(mask) for mask in val_loader.dataset.attention_mask]
    lengths = np.array(lengths)

    # Correlation
    if len(error_magnitude) == len(lengths):
        corr = np.corrcoef(error_magnitude, lengths)[0, 1]
        print(f"Correlation between Error Magnitude and Input Length: {corr:.6f}")
    else:
        print("Error: Mismatch in lengths for failure analysis.")

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.7436591491466628

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        # Note: eval_fn handles torch.no_grad() and eval mode
        test_preds, _ = eval_fn(model, test_loader, device)

        # Apply Optimized Thresholds to Test Predictions
        test_scores = apply_thresholds(test_preds, best_thresholds)

        # Create Submission DataFrame
        # Read test metadata to ensure IDs match exactly
        df_test = pd.read_csv(Config.TEST_PATH)

        if len(df_test) != len(test_scores):
            print(
                f"Warning: Test set size mismatch. DF: {len(df_test)}, Preds: {len(test_scores)}"
            )

        submission = pd.DataFrame(
            {"essay_id": df_test["essay_id"], "score": test_scores}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_orchestration()
