import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_loaders, prepare_test_loader
from library.model import ToxicityModel
from library.engine import train_loop, valid_one_epoch, predict


def main():
    # Set seed for reproducibility
    seed_everything(Config.seed)

    print("Configuring Fast Baseline...")
    # 1. Adjust Configuration for Fast Execution
    # Reduce epochs to 1 for speed
    Config.epochs = 1

    # Create a subset of the training data to ensure completion within time limits
    # We aim for 60,000 samples (approx 50%) to balance speed and performance
    # This should take roughly 30-40 minutes to train on A100
    full_train_meta = pd.read_csv(Config.train_meta_path)
    subset_size = 60000
    if len(full_train_meta) > subset_size:
        print(
            f"Subsampling training data from {len(full_train_meta)} to {subset_size} samples."
        )
        subset_train_meta = full_train_meta.sample(
            n=subset_size, random_state=Config.seed
        ).reset_index(drop=True)
        subset_meta_path = os.path.join(Config.working_dir, "train_subset.csv")
        subset_train_meta.to_csv(subset_meta_path, index=False)
        Config.train_meta_path = subset_meta_path
    else:
        print("Training data smaller than subset limit. Using full data.")

    # 2. Prepare Data Loaders
    # We set load_cached_data=False to ensure the new subset is processed and not the full cached data
    print("Loading and preprocessing data...")
    train_loader, val_loader = prepare_loaders(load_cached_data=False)

    # 3. Initialize Model, Optimizer, and Scheduler
    print("Initializing model...")
    device = Config.device
    model = ToxicityModel()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
        eps=Config.eps,
    )

    # Calculate total training steps for OneCycleLR
    # The scheduler steps once per accumulation cycle, not per batch
    steps_per_epoch = len(train_loader) // Config.accumulate_grad_batches
    total_steps = steps_per_epoch * Config.epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        total_steps=total_steps,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # 4. Training Loop
    # train_loop handles training, validation monitoring, and saving the best model
    print("Starting training loop...")
    model = train_loop(
        model, train_loader, val_loader, optimizer, scheduler, device, patience=1
    )

    # 5. Final Validation on Hold-out Set
    print("Performing final validation...")
    val_loss, val_score, val_preds = valid_one_epoch(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    print("Running Failure Analysis...")
    # Calculate error per sample: Mean Absolute Error across the 6 classes
    # Retrieve targets from the dataset (faster than iterating loader)
    val_targets = val_loader.dataset.labels

    # Compute MAE per sample
    # val_preds shape: (N, 6), val_targets shape: (N, 6)
    sample_errors = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Get input lengths (sum of attention mask)
    val_lengths = np.sum(val_loader.dataset.attention_mask, axis=1)

    # Calculate correlation
    correlation = np.corrcoef(sample_errors, val_lengths)[0, 1]
    print(f"Correlation between Error Magnitude and Input Length: {correlation}")

    # 7. Submission Generation
    threshold = 0.9920650979347099
    if val_score > threshold:
        print(
            f"Validation score ({val_score}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Prepare Test Loader (can use cache if available, or process new)
        test_loader = prepare_test_loader(load_cached_data=True)

        # Predict
        test_preds = predict(model, test_loader, device)

        # Create Submission File
        submission = pd.read_csv(Config.sample_submission_path)
        submission[Config.target_cols] = test_preds

        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"Validation score ({val_score}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
