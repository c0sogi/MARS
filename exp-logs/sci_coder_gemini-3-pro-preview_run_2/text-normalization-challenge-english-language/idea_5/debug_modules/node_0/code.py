import os
import shutil
import torch
import pandas as pd
import numpy as np
from transformers import logging as hf_logging

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Import Config and modify it BEFORE importing other modules that might rely on it
from library.config import Config

# Set up a demo-specific working directory
DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Override Config for a fast demonstration
Config.WORKING_DIR = DEMO_DIR
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 sentences
Config.EPOCHS = 1
Config.TRAIN_BATCH_SIZE = 4
Config.VALID_BATCH_SIZE = 4
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
Config.MODEL_NAME = "roberta-base"  # Ensure we use a standard model

# Update paths to point to the demo directory
Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_processed.parquet")
Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_processed.parquet")
Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_processed.parquet")
Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_DIR, "model_checkpoint.bin")
Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

# Now import the rest of the library
from library.utils import seed_everything, get_logger
from library.normalization_rules import normalize_token
from library.dataset import load_data, get_tokenizer, TextNormalizationDataset
from library.model import TransformerCRF
from library.engine import fit, predict

# Suppress HF warnings for cleaner output
hf_logging.set_verbosity_error()

if __name__ == "__main__":
    # Initialize logger and seed
    logger = get_logger()
    logger.info("Starting Text Normalization Demo Script...")
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Verify Normalization Rules
    # ==========================================
    logger.info("Step 1: Verifying Normalization Rules...")

    # Test Case 1: Money
    raw_money = "$3.50"
    norm_money = normalize_token(raw_money, "MONEY")
    expected_money = "three dollars, fifty cents"
    assert (
        norm_money == expected_money
    ), f"Money Norm Failed: {norm_money} != {expected_money}"

    # Test Case 2: Cardinal
    raw_card = "123"
    norm_card = normalize_token(raw_card, "CARDINAL")
    expected_card = "one hundred twenty-three"
    assert (
        norm_card == expected_card
    ), f"Cardinal Norm Failed: {norm_card} != {expected_card}"

    # Test Case 3: PLAIN (No change)
    raw_plain = "Hello"
    norm_plain = normalize_token(raw_plain, "PLAIN")
    assert norm_plain == raw_plain, "PLAIN Norm Failed"

    logger.info("Normalization rules verified successfully.")

    # ==========================================
    # 3. Data Loading & Preparation
    # ==========================================
    logger.info("Step 2: Loading and Preparing Data...")

    # We use the validation set source for both train/val in this demo to avoid
    # processing the massive training file (7M rows) which takes time.
    # load_data with debug=True returns Config.DEBUG_SAMPLE_SIZE rows (grouped by sentence).

    logger.info("Loading subset of validation data to simulate training data...")
    df_demo = load_data(split="val", load_cached_data=False, debug=True)

    # Split the demo dataframe into train and val
    split_idx = int(0.8 * len(df_demo))
    train_df = df_demo.iloc[:split_idx].reset_index(drop=True)
    val_df = df_demo.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"Demo Train Size: {len(train_df)} sentences")
    logger.info(f"Demo Val Size: {len(val_df)} sentences")

    # Load test data
    logger.info("Loading subset of test data...")
    test_df = load_data(split="test", load_cached_data=False, debug=True)
    logger.info(f"Demo Test Size: {len(test_df)} sentences")

    # ==========================================
    # 4. Dataset & Tokenizer
    # ==========================================
    logger.info("Step 3: Initializing Dataset and Tokenizer...")

    tokenizer = get_tokenizer()

    # Create Dataset objects
    train_dataset = TextNormalizationDataset(
        train_df, tokenizer, Config.LABEL2ID, is_test=False
    )
    val_dataset = TextNormalizationDataset(
        val_df, tokenizer, Config.LABEL2ID, is_test=False
    )
    test_dataset = TextNormalizationDataset(
        test_df, tokenizer, Config.LABEL2ID, is_test=True
    )

    # Verify Dataset Output
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "labels" in sample_item
    assert sample_item["input_ids"].shape == (Config.MAX_LEN,)
    assert sample_item["labels"].shape == (Config.MAX_LEN,)

    logger.info("Dataset verified successfully.")

    # ==========================================
    # 5. Model Initialization & Check
    # ==========================================
    logger.info("Step 4: Initializing Model...")

    device = Config.DEVICE
    model = TransformerCRF()
    model.to(device)

    # Dummy Forward Pass check
    dummy_input = sample_item["input_ids"].unsqueeze(0).to(device)
    dummy_mask = sample_item["attention_mask"].unsqueeze(0).to(device)
    dummy_labels = sample_item["labels"].unsqueeze(0).to(device)

    # Check Loss Calculation
    model.train()
    loss = model(dummy_input, dummy_mask, labels=dummy_labels)
    assert not torch.isnan(loss), "Model returned NaN loss"
    logger.info(f"Initial dummy loss: {loss.item():.4f}")

    # Check Inference Decoding
    model.eval()
    tags = model(dummy_input, dummy_mask)
    assert isinstance(tags, list), "Inference should return a list"
    assert len(tags) == 1, "Should return 1 sequence for batch size 1"
    assert isinstance(tags[0], list), "Sequence should be a list of tags"

    logger.info("Model initialized and verified.")

    # ==========================================
    # 6. Training Loop (Engine)
    # ==========================================
    logger.info("Step 5: Running Training Loop (Fit)...")

    # Run fit (1 epoch, tiny dataset)
    fit(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=Config.EPOCHS,
        batch_size=Config.TRAIN_BATCH_SIZE,
        device=device,
    )

    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved."
    logger.info("Training loop completed.")

    # ==========================================
    # 7. Inference & Submission
    # ==========================================
    logger.info("Step 6: Running Inference (Predict)...")

    predict(model, test_dataset, device=device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df_sub.columns
    assert "after" in df_sub.columns
    assert len(df_sub) > 0

    logger.info(f"Submission generated with {len(df_sub)} rows.")
    logger.info("Head of submission:")
    print(df_sub.head())

    logger.info("Demo Script Completed Successfully.")
