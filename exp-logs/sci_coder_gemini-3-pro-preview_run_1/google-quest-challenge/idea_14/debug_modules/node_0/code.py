import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, logging

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_optimizer_params
from library.dataset import load_data, StackExchangeDataset, get_target_columns
from library.model import ContextualizedDualEncoder
from library.engine import run_training

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

if __name__ == "__main__":
    print("Initializing Demonstration...")

    # 1. Configuration Overrides for Speed and Isolation
    # We modify the Config class attributes directly to affect the library modules
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_subset.parquet")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_subset.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_subset.parquet")

    # Hyperparameters for fast demo
    Config.EPOCHS = 2
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.MAX_LEN = 64  # Short sequence length for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure demo directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading & Preparation
    print("Loading and preparing data subsets...")

    # Load raw data
    # We take a small head of the data to ensure the script finishes quickly
    df_train = load_data("train", load_cached_data=False).head(20)
    df_val = load_data("val", load_cached_data=False).head(20)
    df_test = load_data("test", load_cached_data=False).head(10)

    target_cols = get_target_columns()
    assert (
        len(target_cols) == 30
    ), f"Expected 30 target columns, found {len(target_cols)}"

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = StackExchangeDataset(
        df_train, tokenizer, target_cols=target_cols, max_len=Config.MAX_LEN
    )
    val_dataset = StackExchangeDataset(
        df_val, tokenizer, target_cols=target_cols, max_len=Config.MAX_LEN
    )
    test_dataset = StackExchangeDataset(
        df_test, tokenizer, max_len=Config.MAX_LEN, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify DataLoader output
    sample_batch = next(iter(train_loader))
    assert "input_ids_q" in sample_batch
    assert "labels" in sample_batch
    assert sample_batch["labels"].shape == (Config.TRAIN_BATCH_SIZE, 30)
    print("Data loading verified.")

    # 3. Model Initialization & Verification
    print("Initializing model...")
    model = ContextualizedDualEncoder()
    model.to(device)

    # Dummy forward pass verification
    with torch.no_grad():
        # Move sample batch to device
        ids_q = sample_batch["input_ids_q"].to(device)
        mask_q = sample_batch["attention_mask_q"].to(device)
        ids_a = sample_batch["input_ids_a"].to(device)
        mask_a = sample_batch["attention_mask_a"].to(device)
        pool_mask_a = sample_batch["pooling_mask_a"].to(device)

        output = model(ids_q, mask_q, ids_a, mask_a, pool_mask_a)
        assert output.shape == (
            Config.TRAIN_BATCH_SIZE,
            30,
        ), f"Model output shape mismatch. Expected {(Config.TRAIN_BATCH_SIZE, 30)}, got {output.shape}"
    print("Model architecture verified.")

    # 4. Optimizer & Scheduler Setup
    print("Setting up optimizer and scheduler...")
    optimizer_grouped_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.LEARNING_RATE,
        head_lr=Config.HEAD_LR,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    print("Starting training loop...")
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,  # Strict patience for demo
    )

    # Verify model file creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("Training completed and model saved.")

    # 6. Inference on Test Set
    print("Running inference on test set...")

    # Load best model state
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    test_preds = []
    test_ids = []

    with torch.no_grad():
        for data in test_loader:
            ids_q = data["input_ids_q"].to(device, dtype=torch.long)
            mask_q = data["attention_mask_q"].to(device, dtype=torch.long)
            ids_a = data["input_ids_a"].to(device, dtype=torch.long)
            mask_a = data["attention_mask_a"].to(device, dtype=torch.long)
            pool_mask_a = data["pooling_mask_a"].to(device, dtype=torch.float)
            qa_ids = data["qa_id"].numpy()

            outputs = model(ids_q, mask_q, ids_a, mask_a, pool_mask_a)
            probs = torch.sigmoid(outputs).cpu().numpy()

            test_preds.append(probs)
            test_ids.append(qa_ids)

    test_preds = np.concatenate(test_preds, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)

    assert test_preds.shape == (len(df_test), 30), "Prediction shape mismatch"
    assert len(test_ids) == len(df_test), "ID count mismatch"

    # 7. Submission Generation
    print("Generating submission file...")
    submission_df = pd.DataFrame(test_preds, columns=target_cols)
    submission_df.insert(0, "qa_id", test_ids)

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file not created."

    # Validate submission format
    saved_sub = pd.read_csv(submission_path)
    assert saved_sub.shape == (10, 31), f"Submission shape incorrect: {saved_sub.shape}"
    assert (
        list(saved_sub.columns) == ["qa_id"] + target_cols
    ), "Submission columns mismatch"

    print(
        f"Demonstration completed successfully. Submission saved to {submission_path}"
    )
