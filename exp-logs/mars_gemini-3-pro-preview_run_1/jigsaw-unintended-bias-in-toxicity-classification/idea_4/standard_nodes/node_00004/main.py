import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.config import Config
from library.data_processing import load_and_process_data, ToxicityDataset
from library.model import MultiTaskTransformer
from library.engine import set_seed, train_one_epoch, validate
from library.metrics import calculate_final_score


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Configuration for Fast Baseline
    # We limit training data to ensure the script completes quickly (within 2 hours)
    # but we will evaluate on the full validation set as required.
    TRAIN_LIMIT = 50000
    EPOCHS = 1
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VAL = 128  # Larger batch size for faster inference since no gradients

    print("Loading and processing data...")
    # Load full dataframes from cache or process from scratch
    train_df, val_df, test_df = load_and_process_data(load_cached_data=True)

    # Limit training data for speed
    print(f"Truncating training data to {TRAIN_LIMIT} samples for fast baseline.")
    train_df = train_df.iloc[:TRAIN_LIMIT]

    # Initialize Tokenizer
    print(f"Initializing tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    # We create datasets manually to control the data split sizes explicitly
    train_dataset = ToxicityDataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_dataset = ToxicityDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    test_dataset = ToxicityDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    print("Initializing model...")
    model = MultiTaskTransformer().to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {loss:.6f}")

    # Validation
    print("Performing validation on the entire hold-out set...")
    val_preds, val_targets, val_identities = validate(model, val_loader, device)

    # Reconstruct Validation DataFrame for Metric Calculation
    val_eval_df = pd.DataFrame({"target": val_targets, "prediction": val_preds})

    # Add identity columns back (needed for Bias AUCs)
    # val_identities is shape [N, num_identities], mapped to Config.IDENTITY_COLUMNS
    for idx, col in enumerate(Config.IDENTITY_COLUMNS):
        val_eval_df[col] = val_identities[:, idx]

    # Calculate Metrics
    metrics = calculate_final_score(val_eval_df)
    final_score = metrics["score"]

    # Print required metric
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute error
    val_eval_df["error"] = (val_eval_df["target"] - val_eval_df["prediction"]).abs()

    print("Correlation between Error and Input Features:")
    # Correlate error with identity columns to check for systematic bias
    for col in Config.IDENTITY_COLUMNS:
        if col in val_eval_df.columns:
            corr = val_eval_df["error"].corr(val_eval_df[col])
            print(f"  {col}: {corr:.6f}")

    # Correlate error with target to see if model struggles with toxic or non-toxic more
    target_corr = val_eval_df["error"].corr(val_eval_df["target"])
    print(f"  target: {target_corr:.6f}")

    # Submission Generation
    THRESHOLD = 0.9022848229047395

    if final_score > THRESHOLD:
        print(
            f"\nValidation score ({final_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_preds, _, _ = validate(model, test_loader, device)

        # Load sample submission
        submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Ensure alignment (though test_loader should match sample_submission)
        if len(test_preds) != len(submission):
            print(
                f"Warning: Prediction count {len(test_preds)} differs from submission file {len(submission)}. Truncating."
            )
            submission = submission.iloc[: len(test_preds)]

        submission["prediction"] = test_preds
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score ({final_score}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
