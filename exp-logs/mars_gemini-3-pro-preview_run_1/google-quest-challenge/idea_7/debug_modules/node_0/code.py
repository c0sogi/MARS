import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as hf_logging

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import QADataset, Collate, preprocess_df
from library.model import MultiTaskDualEncoder, get_optimizer_params
from library.engine import train_one_epoch, validate, predict
from library.utils import compute_spearman_metric


def run_demo():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("--- Starting Demonstration ---")

    # Set seed for reproducibility
    seed_everything(42)

    # Runtime overrides for Config to ensure speed
    print("Configuring runtime parameters for speed...")
    Config.MAX_LEN = 64  # Reduce sequence length for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.WORKING_DIR = "./working/demo_run/"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    print("\n[Data] Loading and preprocessing subset...")

    # Load a tiny subset of training data
    df_train = pd.read_csv(Config.TRAIN_PATH).head(20)

    # Apply preprocessing (concatenating title and body)
    df_train = preprocess_df(df_train)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Instantiate Dataset
    train_ds = QADataset(df_train, tokenizer, Config.MAX_LEN, is_test=False)

    # Instantiate Collate function
    collate_fn = Collate(tokenizer)

    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Verification: Check Batch Structure
    print("[Data] Verifying batch structure...")
    batch = next(iter(train_loader))

    expected_keys = [
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "labels",
        "aux_labels",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key {key} in batch"

    # Check shapes
    # Labels should be [Batch, 30]
    assert batch["labels"].shape == (
        Config.BATCH_SIZE,
        30,
    ), f"Incorrect label shape: {batch['labels'].shape}"
    # Aux labels should be [Batch, 21]
    assert batch["aux_labels"].shape == (
        Config.BATCH_SIZE,
        21,
    ), f"Incorrect aux label shape: {batch['aux_labels'].shape}"

    print("Data pipeline verified successfully.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[Model] Initializing MultiTaskDualEncoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultiTaskDualEncoder()
    model.to(device)

    print("[Model] Running forward pass check...")
    # Move batch to device
    q_ids = batch["q_input_ids"].to(device)
    q_mask = batch["q_attention_mask"].to(device)
    a_ids = batch["a_input_ids"].to(device)
    a_mask = batch["a_attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        main_logits, aux_logits = model(q_ids, q_mask, a_ids, a_mask)

    # Verify Output Shapes
    assert main_logits.shape == (
        Config.BATCH_SIZE,
        30,
    ), f"Main logits shape mismatch. Expected {(Config.BATCH_SIZE, 30)}, got {main_logits.shape}"
    assert aux_logits.shape == (
        Config.BATCH_SIZE,
        21,
    ), f"Aux logits shape mismatch. Expected {(Config.BATCH_SIZE, 21)}, got {aux_logits.shape}"

    print("Model architecture verified successfully.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n[Engine] Testing training loop (1 epoch, subset)...")

    # Setup Optimizer
    optimizer_params = get_optimizer_params(
        model, Config.LR_BACKBONE, Config.LR_HEAD, Config.WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    # Setup Scheduler (Dummy for demo)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=10
    )

    # Setup Scaler
    scaler = torch.cuda.amp.GradScaler()

    # Run one epoch
    loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, epoch=0, scaler=scaler
    )

    assert isinstance(loss, float), "Train loss should be a float"
    assert not np.isnan(loss), "Train loss is NaN"
    print(f"Training step complete. Loss: {loss:.4f}")

    # ==========================================
    # 5. Validation & Prediction Demonstration
    # ==========================================
    print("\n[Engine] Testing validation and prediction...")

    # Validate
    # Using the same loader for validation demo to save time
    val_score = validate(model, train_loader, device)
    print(f"Validation Score (Spearman): {val_score:.4f}")
    assert -1.0 <= val_score <= 1.0, "Validation score out of bounds [-1, 1]"

    # Predict
    preds = predict(model, train_loader, device)
    print(f"Predictions shape: {preds.shape}")
    assert preds.shape == (
        len(df_train) // Config.BATCH_SIZE * Config.BATCH_SIZE,
        30,
    ), "Prediction shape mismatch"

    # ==========================================
    # 6. Metric Utility Verification
    # ==========================================
    print("\n[Utils] Verifying Spearman metric calculation...")

    # Create synthetic data
    # Case 1: Perfect correlation
    y_true = np.random.rand(10, 30)
    y_pred = y_true.copy()
    score_perfect = compute_spearman_metric(y_true, y_pred)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected 1.0 for perfect correlation, got {score_perfect}"

    # Case 2: Random data
    y_true = np.random.rand(10, 30)
    y_pred = np.random.rand(10, 30)
    score_random = compute_spearman_metric(y_true, y_pred)
    assert -1.0 <= score_random <= 1.0, "Metric returned value out of range"

    print("Metric utility verified successfully.")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
