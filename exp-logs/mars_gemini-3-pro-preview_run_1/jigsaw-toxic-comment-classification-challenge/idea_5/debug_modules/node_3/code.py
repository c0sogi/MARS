import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_loaders, prepare_test_loader
from library.model import ToxicityModel
from library.engine import train_loop, predict


def main():
    print("=== Toxicity Classification Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config attributes for a fast demonstration.
    # Enabling debug mode samples only 1000 rows from the metadata.
    print("Configuring for fast demonstration (Debug Mode)...")
    Config.debug = True
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.accumulate_grad_batches = 2

    # Ensure reproducibility
    seed_everything(Config.seed)

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("\n[Step 1] Preparing DataLoaders...")
    # load_cached_data=False forces the pipeline to process raw text -> tokens
    # This verifies the preprocessing logic in dataset.py
    train_loader, val_loader = prepare_loaders(load_cached_data=False)

    # Validation assertions
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Val loader should not be empty."

    # Inspect a batch to verify structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "labels" in sample_batch

    print(f"  Batch Input Shape: {sample_batch['input_ids'].shape}")
    print(f"  Batch Label Shape: {sample_batch['labels'].shape}")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[Step 2] Initializing Model...")
    # Instantiates ToxicityModel with DeBERTa-v3-large backbone (as per Config)
    model = ToxicityModel()
    model.to(Config.device)
    print(f"  Model initialized and moved to {Config.device}")

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("\n[Step 3] Setting up Optimizer and Scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
        eps=Config.eps,
    )

    num_train_steps = len(train_loader) * Config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        total_steps=num_train_steps,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    print("\n[Step 4] Starting Training Loop...")
    # Train for 1 epoch. The engine handles validation and saving the best model.
    model = train_loop(
        model, train_loader, val_loader, optimizer, scheduler, Config.device, patience=1
    )

    # Verify model checkpoint was created
    if not os.path.exists(Config.model_save_path):
        raise FileNotFoundError(f"Model file was not saved at {Config.model_save_path}")
    print(f"  Model successfully saved to {Config.model_save_path}")

    # ---------------------------------------------------------
    # 5. Inference
    # ---------------------------------------------------------
    print("\n[Step 5] Running Inference on Test Set...")
    # Prepare test loader (also sampled in debug mode)
    test_loader = prepare_test_loader(load_cached_data=False)

    # Generate predictions
    predictions = predict(model, test_loader, Config.device)
    print(f"  Predictions generated. Shape: {predictions.shape}")

    # Verify prediction shape matches expected output (N_samples, 6 classes)
    assert predictions.shape[1] == Config.num_classes

    # ---------------------------------------------------------
    # 6. Submission Construction
    # ---------------------------------------------------------
    print("\n[Step 6] Creating Submission File...")

    # Create DataFrame from predictions
    submission_df = pd.DataFrame(predictions, columns=Config.target_cols)

    # Add dummy IDs for the demo (since we are using a random debug subset)
    # In a real run, IDs would align with the full sample_submission.csv
    submission_df["id"] = [f"demo_id_{i}" for i in range(len(predictions))]

    # Reorder columns to match submission format: id, toxic, severe_toxic, ...
    cols = ["id"] + Config.target_cols
    submission_df = submission_df[cols]

    # Save submission
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"  Submission saved to {Config.submission_path}")

    # Verify file content
    print("  First 3 rows of submission:")
    print(submission_df.head(3))

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
