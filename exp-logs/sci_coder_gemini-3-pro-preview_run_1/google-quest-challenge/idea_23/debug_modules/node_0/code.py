import os
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import load_and_cache_data, get_tokenizer, QuestDataset
from library.model import SharedBottomSplitTopRoBERTa
from library.engine import run_training, predict_and_submit


def main():
    print("Initializing Demonstration...")

    # 1. Configuration Override for Speed and Demo Purposes
    # We modify the Config class attributes directly to run a fast, lightweight version.
    Config.epochs = 1
    Config.phantom_epochs = 1
    Config.max_len = 64  # Reduce sequence length for speed
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.accum_steps = 1

    # Set up specific output directories for this demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.OUTPUT_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.seed)

    device = Config.device
    print(f"Device: {device}")

    # 2. Data Preparation
    print("Loading and subsetting data...")

    # Load dataframes using the library function
    train_df = load_and_cache_data(Config.TRAIN_PATH, "train_demo")
    val_df = load_and_cache_data(Config.VAL_PATH, "val_demo")
    test_df = load_and_cache_data(Config.TEST_PATH, "test_demo")

    # Subset data for rapid execution (50 train, 20 val, 20 test)
    train_subset = train_df.iloc[:50].reset_index(drop=True)
    val_subset = val_df.iloc[:20].reset_index(drop=True)
    test_subset = test_df.iloc[:20].reset_index(drop=True)

    print(f"Train subset shape: {train_subset.shape}")
    print(f"Val subset shape: {val_subset.shape}")

    # Initialize Tokenizer
    tokenizer = get_tokenizer()

    # Create Datasets
    train_dataset = QuestDataset(train_subset, tokenizer, mode="train")
    val_dataset = QuestDataset(val_subset, tokenizer, mode="val")
    test_dataset = QuestDataset(test_subset, tokenizer, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple debugging/demo
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 3. Model Initialization and Logic Verification
    print("Initializing model...")
    model = SharedBottomSplitTopRoBERTa()
    model.to(device)

    print("Verifying model forward pass logic...")
    # Fetch a single batch to verify shapes
    dummy_batch = next(iter(train_loader))
    q_ids = dummy_batch["q_input_ids"].to(device)
    q_mask = dummy_batch["q_attention_mask"].to(device)
    a_ids = dummy_batch["a_input_ids"].to(device)
    a_mask = dummy_batch["a_attention_mask"].to(device)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        output = model(q_ids, q_mask, a_ids, a_mask)

    # Check output shape: [Batch_Size, Num_Targets]
    expected_shape = (q_ids.size(0), Config.num_labels)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    print(f"Model verification passed. Output shape: {output.shape}")

    # 4. Training Execution
    print("Starting training loop...")
    # run_training handles the loop, validation, and saving the best model
    best_score = run_training(model, train_loader, val_loader, device)

    print(f"Training finished. Best Spearman Score on subset: {best_score:.4f}")

    # 5. Inference and Submission
    print("Generating predictions...")
    # predict_and_submit loads the best model saved during training and creates the CSV
    predict_and_submit(model, test_loader, test_subset, device)

    # 6. Final Validation of Submission File
    print("Validating submission file...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    submission = pd.read_csv(Config.SUBMISSION_PATH)

    # Check 1: Row count matches test subset
    if len(submission) != len(test_subset):
        raise AssertionError(
            f"Submission row count {len(submission)} does not match test set size {len(test_subset)}"
        )

    # Check 2: Column check (qa_id + 30 targets)
    expected_cols = ["qa_id"] + Config.target_cols
    if list(submission.columns) != expected_cols:
        raise AssertionError("Submission columns do not match the expected format.")

    # Check 3: Value ranges [0, 1]
    # Select only target columns
    pred_values = submission[Config.target_cols].values
    if np.any(pred_values < 0) or np.any(pred_values > 1):
        raise AssertionError("Found predictions outside the [0, 1] range.")

    print("Submission validation passed successfully.")
    print("Demonstration complete.")


if __name__ == "__main__":
    main()
