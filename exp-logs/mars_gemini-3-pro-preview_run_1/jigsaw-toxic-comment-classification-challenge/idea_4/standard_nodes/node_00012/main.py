import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import load_dataset, make_loader
from library.model import ToxicityModel, inference_fn
from library.engine import train_one_epoch, valid_one_epoch


def run_training():
    # 1. Setup and Overrides for Fast Baseline
    # We override epochs to 2 to ensure high performance within the time limit on A100.
    # DeBERTa-v3-base converges quickly.
    Config.epochs = 2
    Config.train_batch_size = 16
    Config.valid_batch_size = 32

    # Ensure reproducibility
    seed_everything(Config.seed)

    print(f"Device: {Config.device}")

    # 2. Load Data
    print("Loading datasets...")
    # load_cached_data=True will use parquet caches in ./working if available
    train_df = load_dataset("train", load_cached_data=True)
    val_df = load_dataset("val", load_cached_data=True)
    test_df = load_dataset("test", load_cached_data=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 3. Tokenizer and DataLoaders
    print("Initializing Tokenizer and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    train_loader = make_loader(
        train_df, tokenizer, is_train=True, batch_size=Config.train_batch_size
    )
    val_loader = make_loader(
        val_df, tokenizer, is_train=False, batch_size=Config.valid_batch_size
    )
    test_loader = make_loader(
        test_df, tokenizer, is_train=False, batch_size=Config.valid_batch_size
    )

    # 4. Model Initialization
    print("Initializing Model...")
    model = ToxicityModel()
    model.to(Config.device)

    # 5. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=Config.epochs,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    # 6. Training Loop
    best_score = -np.inf

    print("Starting training...")
    for epoch in range(Config.epochs):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            Config.device,
            scaler=scaler,
            max_grad_norm=Config.max_grad_norm,
        )

        # Validate
        val_loss, val_preds, val_labels = valid_one_epoch(
            model, val_loader, criterion, Config.device
        )

        # Compute Metric
        epoch_score = get_score(val_labels, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {epoch_score:.6f}"
        )

        if epoch_score > best_score:
            best_score = epoch_score
            print(f"New best score! Saving model to {Config.model_save_path}")
            torch.save(model.state_dict(), Config.model_save_path)

    # 7. Final Evaluation on Best Model
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.model_save_path))
    model.to(Config.device)

    # We run validation again to get the exact predictions of the best model state
    val_loss, val_preds, val_labels = valid_one_epoch(
        model, val_loader, criterion, Config.device
    )

    final_metric = get_score(val_labels, val_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    print("\nRunning Failure Analysis...")
    # Calculate Mean Absolute Error per sample across all 6 classes
    # val_labels and val_preds are shape (N, 6)
    errors = np.abs(val_labels - val_preds).mean(axis=1)

    # Get text lengths from validation dataframe
    # Ensure the order matches the loader (make_loader does not shuffle val)
    val_df["char_len"] = val_df["comment_text"].str.len()
    val_df["error"] = errors

    # Calculate correlation
    correlation = val_df["error"].corr(val_df["char_len"])
    print(f"Correlation between Error Magnitude and Comment Length: {correlation:.6f}")

    # 9. Submission
    THRESHOLD = 0.9920650979347099

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = inference_fn(model, test_loader, Config.device)

        # Create Submission DataFrame
        submission = pd.DataFrame(test_preds, columns=Config.target_cols)
        submission["id"] = test_df["id"]

        # Reorder columns
        cols = ["id"] + Config.target_cols
        submission = submission[cols]

        # Save
        os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_training()
