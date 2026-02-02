import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, get_logger
from library.data import ChatbotDataset, augment_data
from library.model import SiameseDebertaMultiLayer
from library.engine import train_fn, eval_fn, inference_fn


def run_demo():
    # 1. Setup and Configuration
    print("--- 1. Setup and Configuration ---")

    # Modify Config for a fast demo run
    Config.working_dir = "./working/demo_run/"
    Config.output_dir = os.path.join(Config.working_dir, "output")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.submission_path = os.path.join(
        Config.working_dir, "submission/submission.csv"
    )

    # Ensure directories exist
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Reduce compute load for demo
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.epochs = 1
    Config.accumulation_steps = 1
    Config.debug = True

    # Set seed for reproducibility
    set_seed(Config.seed)

    # Initialize Logger
    logger = get_logger("demo", log_file=os.path.join(Config.working_dir, "demo.log"))
    logger.info("Configuration updated for demo run.")

    # 2. Data Processing Demonstration
    print("\n--- 2. Data Processing Demonstration ---")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load a tiny subset of data from metadata
    # We use the provided metadata paths
    full_train_df = pd.read_csv(Config.train_path)

    # Take top 20 rows for speed
    subset_df = full_train_df.head(20).copy()
    print(f"Loaded subset of {len(subset_df)} rows for demonstration.")

    # Demonstrate Augmentation
    aug_df = augment_data(subset_df)
    print(f"Augmented dataframe shape: {aug_df.shape}")

    # Verify Augmentation Logic
    # Original row 0 response_a should match Augmented row 20 (index 0 of second half) response_b
    orig_resp_a = subset_df.iloc[0]["response_a"]
    aug_resp_b = aug_df.iloc[20]["response_b"]

    if orig_resp_a != aug_resp_b:
        raise AssertionError(
            "Augmentation logic failed: Swapped responses do not match."
        )
    print("Augmentation logic verified.")

    # Create Dataset and Loader
    train_dataset = ChatbotDataset(aug_df, tokenizer, max_length=128, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in simple demo
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    expected_keys = [
        "input_ids_a",
        "attention_mask_a",
        "response_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "response_mask_b",
        "scalars",
        "target",
    ]

    for key in expected_keys:
        if key not in batch:
            raise AssertionError(f"Missing key in batch: {key}")

    # Verify Shapes
    # Input IDs: (Batch, SeqLen)
    if batch["input_ids_a"].shape != (Config.train_batch_size, 128):
        raise AssertionError(f"Incorrect input_ids shape: {batch['input_ids_a'].shape}")
    # Scalars: (Batch, 3) -> [prompt_len, resp_a_len, resp_b_len]
    if batch["scalars"].shape != (Config.train_batch_size, 3):
        raise AssertionError(f"Incorrect scalars shape: {batch['scalars'].shape}")
    # Targets: (Batch, 3) -> [win_a, win_b, tie]
    if batch["target"].shape != (Config.train_batch_size, 3):
        raise AssertionError(f"Incorrect target shape: {batch['target'].shape}")

    print("Dataset and DataLoader verified.")

    # 3. Model Initialization and Forward Pass
    print("\n--- 3. Model Initialization ---")

    device = Config.device
    model = SiameseDebertaMultiLayer()
    model.to(device)

    # Move batch to device
    inputs = {
        "input_ids_a": batch["input_ids_a"].to(device),
        "attention_mask_a": batch["attention_mask_a"].to(device),
        "response_mask_a": batch["response_mask_a"].to(device),
        "input_ids_b": batch["input_ids_b"].to(device),
        "attention_mask_b": batch["attention_mask_b"].to(device),
        "response_mask_b": batch["response_mask_b"].to(device),
        "scalars": batch["scalars"].to(device),
    }

    # Forward Pass (using mixed precision context as in engine.py)
    model.eval()
    with torch.cuda.amp.autocast(enabled=Config.use_fp16):
        with torch.no_grad():
            outputs = model(**inputs)

    print(f"Model output shape: {outputs.shape}")

    if outputs.shape != (Config.train_batch_size, 3):
        raise AssertionError(
            f"Model output shape mismatch. Expected ({Config.train_batch_size}, 3), got {outputs.shape}"
        )
    print("Model forward pass verified.")

    # 4. Training Loop Demonstration
    print("\n--- 4. Training Loop Execution ---")

    # Setup Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Run Train Function
    print("Running training step...")
    avg_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch=1)

    if not isinstance(avg_loss, float) or np.isnan(avg_loss):
        raise AssertionError("Training function returned invalid loss.")
    print(f"Training step complete. Loss: {avg_loss:.4f}")

    # Run Eval Function (using same loader for simplicity)
    print("Running evaluation step...")
    metrics = eval_fn(model, train_loader, device)
    print(f"Eval Metrics: {metrics}")

    if "log_loss" not in metrics or "accuracy" not in metrics:
        raise AssertionError("Eval metrics missing required keys.")

    # 5. Inference Demonstration
    print("\n--- 5. Inference Execution ---")

    # Load Test Metadata (Subset)
    full_test_df = pd.read_csv(Config.test_path)
    test_subset_df = full_test_df.head(10).copy()

    # Create Test Dataset/Loader
    test_dataset = ChatbotDataset(
        test_subset_df, tokenizer, max_length=128, mode="test"
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.valid_batch_size, shuffle=False, num_workers=0
    )

    # Run Inference
    submission_df = inference_fn(model, test_loader, device)

    # Verify Submission
    print("Verifying submission...")
    if len(submission_df) != 10:
        raise AssertionError(
            f"Submission length mismatch. Expected 10, got {len(submission_df)}"
        )

    required_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    if not all(col in submission_df.columns for col in required_cols):
        raise AssertionError(f"Submission missing columns. Expected {required_cols}")

    # Check if probabilities sum to roughly 1 (tolerance for float precision)
    probs_sum = submission_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(
        axis=1
    )
    if not np.allclose(probs_sum, 1.0, atol=1e-5):
        raise AssertionError("Submission probabilities do not sum to 1.")

    print(f"Submission generated successfully at {Config.submission_path}")
    print(submission_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
