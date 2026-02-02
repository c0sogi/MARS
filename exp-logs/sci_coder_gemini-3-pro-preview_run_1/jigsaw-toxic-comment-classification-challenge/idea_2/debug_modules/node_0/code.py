import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
# Disable tokenizer parallelism to avoid potential deadlock warnings in some environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import ToxicityRoBERTa
from library.engine import run_training, predict, set_seed


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- Setting up Configuration for Demo ---")

    # Modify Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for ultra-fast execution
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16
    Config.MAX_LEN = 64  # Reduce sequence length to speed up processing

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n--- Loading Data ---")

    # We set load_cached_data=False to ensure we process the 'debug' subset
    # rather than loading potentially full-sized cached arrays from a previous run.
    train_loader, val_loader, tokenizer = get_dataloaders(load_cached_data=False)

    # Validate DataLoaders
    try:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Assert shapes
        assert input_ids.shape == (
            Config.TRAIN_BATCH_SIZE,
            Config.MAX_LEN,
        ), f"Expected input shape {(Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)}, got {input_ids.shape}"
        assert labels.shape == (
            Config.TRAIN_BATCH_SIZE,
            Config.NUM_LABELS,
        ), f"Expected label shape {(Config.TRAIN_BATCH_SIZE, Config.NUM_LABELS)}, got {labels.shape}"

        print("DataLoader validation passed: Batch shapes are correct.")
    except StopIteration:
        raise ValueError("Train DataLoader is empty!")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- Initializing Model ---")
    model = ToxicityRoBERTa()
    model.to(Config.DEVICE)

    # Validate Model Forward Pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        # Move sample batch to device
        ids = batch["input_ids"].to(Config.DEVICE)
        mask = batch["attention_mask"].to(Config.DEVICE)

        logits = model(ids, mask)

        assert logits.shape == (
            Config.TRAIN_BATCH_SIZE,
            Config.NUM_LABELS,
        ), f"Expected output shape {(Config.TRAIN_BATCH_SIZE, Config.NUM_LABELS)}, got {logits.shape}"

    print("Model validation passed: Output shape is correct.")

    # ==========================================
    # 4. Training
    # ==========================================
    print("\n--- Starting Training Loop ---")
    # This runs training, validation, and saves the best model to Config.MODEL_SAVE_PATH
    run_training(model, train_loader, val_loader)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file was not created at {Config.MODEL_SAVE_PATH}"
        )

    print("Training finished successfully.")

    # ==========================================
    # 5. Inference
    # ==========================================
    print("\n--- Running Inference on Test Set ---")

    # Load Test Data
    test_loader = get_test_dataloader(tokenizer, load_cached_data=False)

    # Load Best Saved Model
    print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    # Generate Predictions
    preds = predict(model, test_loader, Config.DEVICE)

    # In Debug mode, the test set is also truncated to DEBUG_SAMPLE_SIZE
    expected_test_size = min(Config.DEBUG_SAMPLE_SIZE, 153164)

    assert preds.shape == (
        expected_test_size,
        Config.NUM_LABELS,
    ), f"Expected prediction shape {(expected_test_size, Config.NUM_LABELS)}, got {preds.shape}"

    print(f"Inference complete. Predictions shape: {preds.shape}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- Generating Submission File ---")

    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # Apply the same slicing logic as the dataset loader to match IDs with Predictions
    if Config.DEBUG:
        test_meta = test_meta.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Construct Submission DataFrame
    submission = pd.DataFrame(preds, columns=Config.LABEL_COLS)
    submission.insert(0, "id", test_meta["id"].values)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df_check = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_check.shape}")
        print("First 3 rows:")
        print(df_check.head(3))
    else:
        raise FileNotFoundError("Submission file was not saved.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
