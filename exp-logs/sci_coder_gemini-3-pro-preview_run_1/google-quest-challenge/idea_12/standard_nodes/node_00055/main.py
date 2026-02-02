import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import spearmanr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DistilRobertaDualEncoder
from library.engine import (
    train_one_epoch,
    validate,
    predict,
    EarlyStopping,
    get_optimizer_params,
)


def main():
    # 1. Setup
    Config.create_dirs()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Tokenizer and Dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DistilRobertaDualEncoder()
    model.to(device)

    # 4. Training Loop
    # Define total epochs
    total_epochs = Config.NUM_EPOCHS
    warmup_epochs = Config.WARMUP_EPOCHS

    # Early Stopping setup
    early_stopping = EarlyStopping(
        patience=3, mode="max", save_path=Config.MODEL_SAVE_PATH
    )

    # --- Stage 1: Head Warmup (Backbone Frozen) ---
    if warmup_epochs > 0:
        print(f"\n--- Stage 1: Head Warmup ({warmup_epochs} Epochs) ---")

        # Freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Optimizer for head only
        optimizer_params = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(optimizer_params)

        for epoch in range(1, warmup_epochs + 1):
            print(f"Epoch {epoch}/{total_epochs} (Warmup)")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, val_score = validate(model, val_loader, device)

            # We don't necessarily need early stopping during warmup, but we track best score
            early_stopping(val_score, model)

    # --- Stage 2: Fine-tuning (Backbone Unfrozen) ---
    fine_tune_epochs = total_epochs - warmup_epochs
    if fine_tune_epochs > 0:
        print(f"\n--- Stage 2: Fine-tuning ({fine_tune_epochs} Epochs) ---")

        # Unfreeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = True

        # Re-initialize optimizer with differential learning rates
        optimizer_params = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(optimizer_params)

        # Scheduler
        num_training_steps = len(train_loader) * fine_tune_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )

        for epoch in range(warmup_epochs + 1, total_epochs + 1):
            print(f"Epoch {epoch}/{total_epochs} (Fine-tuning)")
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch, scheduler
            )
            val_loss, val_score = validate(model, val_loader, device)

            early_stopping(val_score, model)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    # 5. Final Evaluation
    print("\n--- Final Evaluation ---")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)

    # Compute final metric
    _, final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Get predictions on validation set
    val_preds = predict(model, val_loader, device)
    val_targets = val_loader.dataset.targets

    # Compute MAE per sample (averaged across 30 targets)
    # Shape: (N_val, 30) -> (N_val,)
    abs_errors = np.abs(val_preds - val_targets)
    mean_abs_error = np.mean(abs_errors, axis=1)

    # Get metadata features (lengths)
    # Access underlying dataframe from dataset
    val_df = val_loader.dataset.data

    # Calculate lengths
    q_lengths = val_df["question_body"].fillna("").str.len().values
    a_lengths = val_df["answer"].fillna("").str.len().values

    # Compute correlations
    corr_q, _ = spearmanr(mean_abs_error, q_lengths)
    corr_a, _ = spearmanr(mean_abs_error, a_lengths)

    print(f"Correlation between Error and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error and Answer Length: {corr_a:.4f}")

    # 7. Submission
    threshold = 0.40802662717842303
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Generate predictions
        test_preds = predict(model, test_loader, device)

        # Create submission DataFrame
        sub_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)

        # Add qa_id
        test_ids = test_loader.dataset.data["qa_id"].values
        sub_df.insert(0, "qa_id", test_ids)

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
