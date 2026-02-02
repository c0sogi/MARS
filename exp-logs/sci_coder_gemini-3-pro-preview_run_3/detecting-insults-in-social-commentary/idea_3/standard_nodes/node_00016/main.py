import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_dataloaders
from library.model import InsultModel
from library.trainer import get_optimizer_params, train_fn, valid_fn, inference_fn


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    # We use the full dataset as it is small (~3k samples), fitting the 'fast baseline' requirement naturally.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.debug
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = InsultModel()
    model.to(device)

    # 4. Optimizer & Scheduler
    # Configure LLRD (Layer-wise Learning Rate Decay)
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=Config.learning_rate, eps=1e-6
    )

    num_train_steps = int(len(train_loader) * Config.epochs)
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.output_dir, "best_model.bin")

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss, val_auc = valid_fn(val_loader, model, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            print(f"  AUC Improved ({best_auc:.6f} -> {val_auc:.6f}). Saving model...")
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
        else:
            print("  AUC did not improve.")

    # 6. Final Evaluation & Failure Analysis
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    # Get Validation Predictions
    print("Running inference on validation set...")
    val_preds = inference_fn(val_loader, model, device)

    # Load Validation Targets (Order is preserved in DataLoader with shuffle=False)
    # We load from the source file to ensure alignment
    val_df = pd.read_csv(Config.val_path)
    if Config.debug:
        val_df = val_df.head(Config.debug_sample_size).reset_index(drop=True)

    val_targets = val_df[Config.target_col].values

    # Calculate Final Metric
    final_auc = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Feature extraction for analysis
    # Ensure text is string
    val_df["str_comment"] = val_df["Comment"].astype(str)
    val_df["char_len"] = val_df["str_comment"].apply(len)
    val_df["word_count"] = val_df["str_comment"].apply(lambda x: len(x.split()))

    # Correlations
    corr_char = np.corrcoef(errors, val_df["char_len"])[0, 1]
    corr_word = np.corrcoef(errors, val_df["word_count"])[0, 1]

    print(f"Correlation between Error and Character Length: {corr_char:.6f}")
    print(f"Correlation between Error and Word Count: {corr_word:.6f}")

    # 7. Submission
    threshold = 0.9639408866995074
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) meets threshold ({threshold}). Generating submission..."
        )

        # Test Inference
        test_preds = inference_fn(test_loader, model, device)

        # Prepare Submission DataFrame
        test_df = pd.read_csv(Config.test_path)
        if Config.debug:
            test_df = test_df.head(Config.debug_sample_size).reset_index(drop=True)

        submission = pd.DataFrame()
        submission["Insult"] = test_preds
        submission["Date"] = test_df["Date"]
        submission["Comment"] = test_df["Comment"]

        # Ensure column order matches sample
        submission = submission[["Insult", "Date", "Comment"]]

        print(f"Saving submission to {Config.submission_path}...")
        submission.to_csv(Config.submission_path, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"\nValidation metric ({final_auc}) did NOT meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
