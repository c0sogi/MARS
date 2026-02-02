import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import DistilRobertaForTagging
from library.trainer import Trainer
from library.inference import run_inference


def main():
    print("=== Starting Demonstration of Stack Exchange Tag Predictor ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and demo purposes
    Config.debug = True
    Config.debug_sample_size = 200  # Use only 200 samples for quick execution
    Config.epochs = 1
    Config.train_batch_size = 8
    Config.valid_batch_size = 16
    Config.num_workers = 0  # Disable multiprocessing for simple script execution

    # Create a specific working directory for this demo
    Config.working_dir = "./working/demo_run"
    os.makedirs(Config.working_dir, exist_ok=True)

    # Manually update dependent paths since they were initialized at import time
    Config.tags_path = os.path.join(Config.working_dir, "tags.json")
    Config.model_save_path = os.path.join(
        Config.working_dir, "distilroberta_finetuned.pth"
    )
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    os.makedirs(Config.cache_dir, exist_ok=True)

    # Update submission path
    Config.submission_dir = os.path.join(Config.working_dir)
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Set seed for reproducibility
    set_seed(Config.seed)
    print(f"Debug Mode: {Config.debug}")
    print(f"Device: {Config.device}")
    print(f"Working Directory: {Config.working_dir}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data and creating DataLoaders...")

    # Force reload of data (ignore cache) to demonstrate processing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Update Config.num_labels to match the actual vocabulary size found in the data
    # This handles cases (like debug mode) where available tags < requested tags
    actual_num_labels = train_loader.dataset.labels.shape[1]
    if Config.num_labels != actual_num_labels:
        print(
            f"Updating Config.num_labels from {Config.num_labels} to {actual_num_labels}"
        )
        Config.num_labels = actual_num_labels

    # Validation: Check DataLoader properties
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Validation: Check Batch Structure
    print("Verifying batch structure...")
    sample_batch = next(iter(train_loader))

    # Check keys
    expected_keys = {"input_ids", "attention_mask", "labels", "id"}
    assert expected_keys.issubset(
        sample_batch.keys()
    ), f"Batch missing keys. Found: {sample_batch.keys()}"

    # Check shapes
    batch_size = sample_batch["input_ids"].shape[0]
    seq_len = sample_batch["input_ids"].shape[1]
    num_labels = sample_batch["labels"].shape[1]

    assert (
        batch_size == Config.train_batch_size
    ), f"Expected batch size {Config.train_batch_size}, got {batch_size}"
    assert (
        seq_len == Config.max_length
    ), f"Expected sequence length {Config.max_length}, got {seq_len}"
    assert (
        num_labels == Config.num_labels
    ), f"Expected num_labels {Config.num_labels}, got {num_labels}"

    print("Data loading verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model...")

    model = DistilRobertaForTagging(config=Config)
    model.to(Config.device)

    # Validation: Check Forward Pass
    print("Verifying model forward pass...")
    model.eval()
    with torch.no_grad():
        input_ids = sample_batch["input_ids"].to(Config.device)
        attention_mask = sample_batch["attention_mask"].to(Config.device)

        logits = model(input_ids, attention_mask)

        assert logits.shape == (
            batch_size,
            Config.num_labels,
        ), f"Output shape mismatch. Expected {(batch_size, Config.num_labels)}, got {logits.shape}"

    print("Model initialization verification successful.")

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    print("\n[4] Starting Training Loop...")

    trainer = Trainer(model, train_loader, val_loader, config=Config)

    # Execute training (1 epoch on small subset)
    trainer.fit()

    # Validation: Check if model artifact was saved
    if not os.path.exists(Config.model_save_path):
        raise FileNotFoundError(f"Model file was not saved to {Config.model_save_path}")

    print(f"Training complete. Model saved to {Config.model_save_path}")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference Pipeline...")

    # Reload model state to ensure we are using the saved best model
    # (Though trainer.model is already updated, this simulates a fresh inference run)
    model.load_state_dict(
        torch.load(Config.model_save_path, map_location=Config.device)
    )

    # Run full inference: Validation -> Threshold Opt -> Test -> Submission
    run_inference(model, val_loader, test_loader, load_cached_data=False)

    # Validation: Check Submission File
    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    print(f"Submission generated at {Config.submission_path}")

    # Validate Submission Content
    df_sub = pd.read_csv(Config.submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission missing required columns."
    assert (
        len(df_sub) == Config.debug_sample_size
    ), f"Submission row count mismatch. Expected {Config.debug_sample_size}, got {len(df_sub)}"

    # Check if Tags are strings (even empty ones)
    assert (
        df_sub["Tags"].apply(lambda x: isinstance(x, str)).all()
    ), "Tags column contains non-string values."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
