import sys
import os
import shutil
import torch
import pandas as pd
import numpy as np

# Ensure the library module is accessible
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_processing import create_dataloaders, TextVectorizer
from library.model import ChatbotMLP
from library.trainer import ModelTrainer


def run_demo():
    # --- 1. Configuration & Setup ---
    print("Initializing Demo Configuration...")

    # Override Config for a fast, lightweight demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 rows
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.EARLY_STOPPING_PATIENCE = 1

    # Use a specific directory for this demo to avoid clutter/conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Initialize Logger
    logger = get_logger("demo_runner")
    logger.info("Configuration set for Debug mode.")

    # --- 2. Data Processing Demonstration ---
    logger.info("--- Demonstrating Data Processing ---")

    # 2a. Test Vectorizer independently
    logger.info("Testing TextVectorizer...")
    vectorizer = TextVectorizer(
        model_name=Config.TRANSFORMER_MODEL, device=Config.DEVICE
    )
    test_sentences = ["This is a test prompt.", "This is a test response."]
    embeddings = vectorizer.encode(test_sentences)

    # Verify embeddings shape: (2, EMBEDDING_DIM)
    assert embeddings.shape == (
        2,
        Config.EMBEDDING_DIM,
    ), f"Vectorizer output shape mismatch. Expected (2, {Config.EMBEDDING_DIM}), got {embeddings.shape}"
    logger.info("TextVectorizer output shape verified.")

    # 2b. Create DataLoaders
    logger.info(
        "Creating DataLoaders (this involves computing embeddings for the debug subset)..."
    )
    # We set load_cached_data=False to force the pipeline to run through the encoding step
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Verify Train Loader Batch
    features, targets = next(iter(train_loader))

    # Expected input dimension: Prompt + ResA + ResB + Diff + Prod = 5 vectors
    expected_input_dim = Config.EMBEDDING_DIM * 5

    assert features.shape == (
        Config.BATCH_SIZE,
        expected_input_dim,
    ), f"Batch feature shape mismatch. Expected ({Config.BATCH_SIZE}, {expected_input_dim}), got {features.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Batch target shape mismatch. Expected ({Config.BATCH_SIZE},), got {targets.shape}"

    logger.info(
        f"DataLoader batch shapes verified: Features {features.shape}, Targets {targets.shape}"
    )

    # --- 3. Model Initialization Demonstration ---
    logger.info("--- Demonstrating Model Initialization ---")

    model = ChatbotMLP(
        input_dim=expected_input_dim,
        hidden_layers=[32, 16],  # Small layers for demo speed
        dropout_rate=0.1,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = features.to(Config.DEVICE)
        logits = model(dummy_input)

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, {Config.NUM_CLASSES}), got {logits.shape}"

    logger.info("Model forward pass verified.")

    # --- 4. Training Demonstration ---
    logger.info("--- Demonstrating Training Loop ---")

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Run training
    trainer.train()

    # Verify model checkpoint creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    logger.info("Training complete and model checkpoint verified.")

    # --- 5. Inference Demonstration ---
    logger.info("--- Demonstrating Inference/Submission ---")

    trainer.generate_submission()

    # Verify submission file creation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    logger.info(f"Submission loaded. Shape: {df_sub.shape}")

    # Check row count (should match debug sample size)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check probability validity
    probs = df_sub[["winner_model_a", "winner_model_b", "winner_tie"]].values
    # Sum should be approx 1.0
    assert np.allclose(
        probs.sum(axis=1), 1.0, atol=1e-5
    ), "Probabilities do not sum to 1.0"
    # Values should be between 0 and 1
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    logger.info("Submission content verified successfully.")
    print("\nSUCCESS: All demonstrations and verifications passed.")


if __name__ == "__main__":
    run_demo()
