import os
import sys
import torch
import pandas as pd
import logging
import shutil
from transformers import AutoTokenizer

# Import library components
from library.config import Config, set_seed
from library.model import MultiTaskXLMR
from library.data import create_loaders
from library.trainer import Trainer
from library.inference import run_inference


def setup_environment():
    """
    Configures the environment, overrides Config for speed, and suppresses warnings.
    """
    # Suppress transformers logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Override Config for a fast demonstration
    print("Configuring environment for fast demonstration...")

    # Use a smaller model for speed in this demo
    Config.MODEL_CHECKPOINT = "xlm-roberta-base"

    # Enable Debug mode to use a tiny subset of data (e.g., 20 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20

    # Reduce training duration
    Config.EPOCHS = 1

    # Use a single seed for the demo
    Config.SEEDS = [42]

    # Adjust batch size if necessary (4 is usually fine for base models)
    Config.BATCH_SIZE = 4

    # Disable multiprocessing for simple demo stability
    Config.NUM_WORKERS = 0

    # Set specific working directory for this demo run
    Config.WORKING_DIR = os.path.join(Config.ROOT_DIR, "working", "demo_run")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    set_seed(42)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")


def verify_data_loading(tokenizer):
    """
    Demonstrates data loading and verifies the structure of the batches.
    """
    print("\n=== Verifying Data Loading ===")

    # Force reload/reprocess by setting load_cached_data=False for the demo
    # This ensures we actually test the processing logic
    train_loader, test_loader, test_features_df = create_loaders(
        tokenizer, load_cached_data=False
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    # Assertions
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."
    assert not test_features_df.empty, "Test features DataFrame is empty."

    # Inspect a single batch
    batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "relevance_labels",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    print("Data loading verification passed.")
    return train_loader


def verify_model_forward_pass(model, loader):
    """
    Demonstrates model instantiation and verifies the forward pass shapes.
    """
    print("\n=== Verifying Model Forward Pass ===")

    model.to(Config.DEVICE)
    model.train()

    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(Config.DEVICE)
    attention_mask = batch["attention_mask"].to(Config.DEVICE)

    # Run forward pass
    start_logits, end_logits, rel_logits = model(input_ids, attention_mask)

    # Check shapes
    batch_size, seq_len = input_ids.shape

    print(f"Input Shape: {input_ids.shape}")
    print(f"Start Logits Shape: {start_logits.shape}")
    print(f"End Logits Shape: {end_logits.shape}")
    print(f"Relevance Logits Shape: {rel_logits.shape}")

    assert start_logits.shape == (batch_size, seq_len), "Incorrect start_logits shape"
    assert end_logits.shape == (batch_size, seq_len), "Incorrect end_logits shape"
    assert rel_logits.shape == (batch_size, 1), "Incorrect relevance_logits shape"

    print("Model forward pass verification passed.")


def demonstrate_training(model, train_loader):
    """
    Demonstrates the training loop using the Trainer class.
    """
    print("\n=== Demonstrating Training Loop ===")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=Config.DEVICE,
        seed=Config.SEEDS[0],
    )

    # Train for the configured epochs (1 epoch in this demo)
    save_path = trainer.train()

    # Verify model was saved
    assert os.path.exists(save_path), f"Model file not found at {save_path}"
    print(f"Training complete. Model saved to: {save_path}")


def demonstrate_inference():
    """
    Demonstrates the full inference pipeline: loading models, predicting, and saving submission.
    """
    print("\n=== Demonstrating Inference Pipeline ===")

    # run_inference handles loading test data, loading the saved model(s),
    # computing predictions, and saving the CSV.
    run_inference()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df)} rows.")

    # Check columns
    assert "id" in df.columns, "Submission missing 'id' column"
    assert (
        "PredictionString" in df.columns
    ), "Submission missing 'PredictionString' column"

    # Check content (should not be empty given we have a test set)
    assert len(df) > 0, "Submission file is empty"

    print("Inference verification passed.")
    print("First 3 rows of submission:")
    print(df.head(3))


if __name__ == "__main__":
    # 1. Setup
    setup_environment()

    # 2. Initialize Tokenizer
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # 3. Verify Data Loading
    train_loader = verify_data_loading(tokenizer)

    # 4. Initialize Model
    print("Initializing Model...")
    model = MultiTaskXLMR(pretrained_model_name=Config.MODEL_CHECKPOINT)

    # 5. Verify Model Logic
    verify_model_forward_pass(model, train_loader)

    # 6. Run Training
    demonstrate_training(model, train_loader)

    # 7. Run Inference
    demonstrate_inference()

    print("\nAll demonstrations completed successfully.")
