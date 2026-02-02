import os
import sys
import torch
import pandas as pd
import shutil
import logging
from transformers import logging as hf_logging

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_optimizer_grouped_parameters
from library.data import prepare_data, QADataset
from library.model import CustomXLMRoberta
from library.engine import train_fn
from library.inference import run_inference


def main():
    # 0. Setup & Configuration
    print("Setting up demonstration...")

    # Suppress verbose logs from transformers
    hf_logging.set_verbosity_error()
    logging.basicConfig(level=logging.ERROR)

    # Modify Config for a fast demonstration run
    Config.DEBUG = True  # Reduces dataset size (Train: 50, Val/Test: 20)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.INFERENCE_BATCH_SIZE = 2
    Config.SEEDS = [42]  # Run only one seed
    Config.USE_FGM = False  # Disable adversarial training for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup working directories for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "output")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean and recreate directories to ensure a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set fixed seeds for reproducibility
    seed_everything(42)

    # 1. Data Preparation
    print("\n=== Step 1: Data Preparation ===")
    # We disable loading cached data to demonstrate the raw processing logic
    train_dataset, test_dataset = prepare_data(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Validation: Ensure datasets are not empty
    if len(train_dataset) == 0:
        raise AssertionError("Train dataset is empty!")
    if len(test_dataset) == 0:
        raise AssertionError("Test dataset is empty!")

    # Create DataLoader for training
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Initialization & Verification
    print("\n=== Step 2: Model Initialization ===")
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = CustomXLMRoberta(Config.MODEL_NAME)
    model.to(device)

    # Verification: Run a single forward pass to check shapes
    print("Verifying forward pass...")
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        start_logits, end_logits, relevance_logits = model(input_ids, attention_mask)

    # Check output shapes
    # start/end logits: (Batch, Seq_Len)
    # relevance logits: (Batch, 1)
    seq_len = input_ids.shape[1]

    assert start_logits.shape == (
        Config.BATCH_SIZE,
        seq_len,
    ), f"Expected start_logits shape {(Config.BATCH_SIZE, seq_len)}, got {start_logits.shape}"
    assert end_logits.shape == (
        Config.BATCH_SIZE,
        seq_len,
    ), f"Expected end_logits shape {(Config.BATCH_SIZE, seq_len)}, got {end_logits.shape}"
    assert relevance_logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected relevance_logits shape {(Config.BATCH_SIZE, 1)}, got {relevance_logits.shape}"

    print("Forward pass verification successful.")

    # 3. Training Loop Demonstration
    print("\n=== Step 3: Training Loop ===")

    # Setup Optimizer with Layer-wise Learning Rate Decay
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, Config)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    # Run one epoch of training
    # Note: DEBUG mode ensures this is very short
    avg_loss = train_fn(train_loader, model, optimizer, device)
    print(f"Epoch 1 Average Loss: {avg_loss:.4f}")

    if not (isinstance(avg_loss, float) and avg_loss > 0):
        raise AssertionError("Training loss is invalid.")

    # 4. Saving Model
    print("\n=== Step 4: Saving Model Checkpoint ===")
    # Save the model corresponding to seed 42 (as configured in SEEDS)
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "model_seed_42.pth")
    torch.save(model.state_dict(), checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise AssertionError(f"Checkpoint not found at {checkpoint_path}")
    print(f"Model saved to {checkpoint_path}")

    # 5. Inference Pipeline
    print("\n=== Step 5: Inference Pipeline ===")

    # Run inference using the saved model
    # We disable loading cached data to ensure the inference pipeline processes the raw test data
    run_inference(load_cached_data=False)

    # Verify Submission File
    submission_path = Config.SUBMISSION_FILE
    if not os.path.exists(submission_path):
        raise AssertionError(f"Submission file not generated at {submission_path}")

    sub_df = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(sub_df)}")

    # Check columns
    expected_cols = ["id", "PredictionString"]
    if not all(col in sub_df.columns for col in expected_cols):
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {sub_df.columns.tolist()}"
        )

    # Display sample predictions
    print("Sample Predictions:")
    print(sub_df.head())

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
