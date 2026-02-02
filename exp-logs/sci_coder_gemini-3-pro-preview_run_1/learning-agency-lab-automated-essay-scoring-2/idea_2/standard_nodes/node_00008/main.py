import os
import torch
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import MSELoss
import importlib

# Import from provided library files
import library.config

importlib.reload(library.config)
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import EssayRegressor
from library.engine import train_model, validate, predict, generate_submission


def main():
    # 1. Setup and Configuration
    # Set fixed seed for reproducibility
    seed_everything(Config.seed)

    # Suppress verbose transformer logging
    transformers.logging.set_verbosity_error()

    print("=== Starting Essay Scoring Pipeline ===")

    # 2. Data Preparation
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    print("Loading Data...")
    # get_dataloaders handles loading from metadata and caching
    train_loader, val_loader, test_loader = get_dataloaders(tokenizer)

    # 3. Model Initialization
    print("Initializing Model...")
    model = EssayRegressor(pretrained=True)
    model.to(Config.device)

    # 4. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(Config.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    print(f"Starting training for {Config.epochs} epochs...")
    # train_model handles the loop, validation, and saving the best model state
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        Config.device,
        epochs=Config.epochs,
        patience=2,
    )

    # 6. Final Validation Assessment
    print("Performing Final Validation...")
    criterion = MSELoss()
    # validate returns the loss and the QWK metric
    val_loss, val_qwk = validate(model, val_loader, Config.device, criterion)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {val_qwk}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    # Get raw predictions (continuous scores) for the validation set
    val_preds = predict(model, val_loader, Config.device)
    val_preds = np.array(val_preds)

    # Get ground truth scores and original dataframe for metadata
    val_df = val_loader.dataset.df.copy()
    val_targets = val_df["score"].values

    # Calculate residuals (absolute difference between prediction and truth)
    # We use the raw predictions to capture the magnitude of error
    residuals = np.abs(val_preds - val_targets)

    # Calculate essay length (character count)
    val_df["char_count"] = val_df["full_text"].astype(str).apply(len)
    val_df["residual"] = residuals

    # Calculate correlation between error magnitude and essay length
    corr_len = val_df["char_count"].corr(val_df["residual"])
    print(
        f"Correlation between Error Magnitude and Essay Length (Char Count): {corr_len}"
    )

    # 8. Submission Generation
    THRESHOLD = 0.8148349183359068

    if val_qwk > THRESHOLD:
        print(
            f"Validation metric ({val_qwk}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, Config.device)
    else:
        print(
            f"Validation metric ({val_qwk}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )

    print("=== Pipeline Completed ===")


if __name__ == "__main__":
    main()
