import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import create_dataloaders
from library.modeling import ToxicityModel
from library.engine import run_training, valid_fn, generate_submission


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Configuration
    # Initialize Config with 2 epochs to allow AWP (starts at epoch 1) to function.
    # Batch size 32 is chosen to maximize A100 GPU utilization and speed.
    cfg = Config(debug=False, epochs=2, batch_size=32)

    # Override submission path to meet the specific requirement "./submission/submission.csv"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    cfg.submission_path = os.path.join(submission_dir, "submission.csv")

    # Set fixed seeds for reproducibility
    seed_everything(cfg.seed)

    # 2. Data Loading
    # Load cached data if available to minimize preprocessing time
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = ToxicityModel(cfg, pretrained=True)
    model.to(cfg.device)

    # 4. Optimizer and Scheduler
    # We use AdamW and OneCycleLR as specified in the idea description
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    num_training_steps = len(train_loader) * cfg.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=num_training_steps,
        pct_start=cfg.pct_start,
        anneal_strategy="cos",
    )

    # 5. Training
    # run_training handles the training loop, AWP execution, and saving the best model
    print("Starting Training...")
    run_training(cfg, model, train_loader, val_loader, optimizer, scheduler, cfg.device)

    # 6. Evaluation & Metrics
    print("Performing Final Evaluation...")
    # Load the best model weights saved during training
    if os.path.exists(cfg.model_save_path):
        model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
    else:
        print("Warning: Best model weights not found. Using current weights.")

    # Calculate final validation metric on the hold-out set
    val_loss, val_score, val_preds = valid_fn(val_loader, model, cfg.device, cfg)

    # Print the required metric string
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    val_labels = []
    val_lengths = []

    # Extract labels and input lengths from validation loader
    # val_loader is not shuffled, so the order matches val_preds
    for data in val_loader:
        # Length = number of non-padding tokens (sum of attention mask)
        lengths = data["attention_mask"].sum(dim=1).numpy()
        labels = data["labels"].numpy()

        val_lengths.append(lengths)
        val_labels.append(labels)

    val_lengths = np.concatenate(val_lengths)
    val_labels = np.concatenate(val_labels)

    # Calculate error magnitude per sample (Mean Absolute Error averaged across 6 classes)
    # val_preds are probabilities [0, 1], val_labels are binary [0, 1]
    error_magnitude = np.abs(val_labels - val_preds).mean(axis=1)

    # Calculate correlation between input length and error magnitude
    correlation = np.corrcoef(val_lengths, error_magnitude)[0, 1]
    print(f"Correlation between Input Length and Model Error: {correlation}")

    # 8. Submission Generation
    # Only generate submission if metric exceeds the specified threshold
    threshold = 0.9920650979347099

    if val_score > threshold:
        print(
            f"Validation score {val_score} exceeds threshold {threshold}. Generating submission..."
        )
        # generate_submission uses the current model state (best weights loaded)
        # and saves to cfg.submission_path
        generate_submission(cfg, model, test_loader, cfg.device)
    else:
        print(
            f"Validation score {val_score} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
