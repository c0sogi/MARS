import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler

# Import library modules
from library.config import Config
from library.utils import (
    set_seed,
    get_logger,
    AverageMeter,
    calculate_metrics,
    EarlyStopping,
)
from library.data import clean_text, Vocabulary, get_dataloaders
from library.model import DualStreamTextCNN
from library.train import Trainer
from library.inference import predict_probs, optimize_threshold, generate_submission

# Initialize Logger
logger = get_logger("demo_script")


def run_demo():
    logger.info("Starting Stack Exchange Tag Prediction Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    logger.info("1. Configuring environment for fast demo execution...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 500  # Small subset for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.VOCAB_SIZE = 1000
    Config.EMBED_DIM = 64  # Smaller embedding for demo
    Config.NUM_FILTERS = 16  # Fewer filters for demo

    # Use a separate working directory for this demo
    Config.WORKING_DIR = "./working/demo_pipeline"

    # Update dependent paths in Config manually since they were initialized at import time
    Config.TOKENIZER_PATH = os.path.join(Config.WORKING_DIR, "tokenizer.json")
    Config.MLB_PATH = os.path.join(
        Config.WORKING_DIR, "mlb.json"
    )  # Changed extension to match data.py logic if needed, though data.py uses mlb_classes.json
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_demo.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up demo directory if it exists to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Run setup (creates directories, sets seeds)
    Config.setup()
    set_seed(Config.SEED)
    device = Config.get_device()
    logger.info(f"Running on device: {device}")

    # ==========================================
    # 2. Data Processing Verification
    # ==========================================
    logger.info("2. Verifying Data Processing...")

    # Test clean_text
    raw_text = "<p>Hello <b>World</b>!</p> C++ code: std::cout"
    cleaned = clean_text(raw_text)
    expected = "hello world c++ code std cout"
    assert (
        cleaned == expected
    ), f"clean_text failed. Got: '{cleaned}', Expected: '{expected}'"
    logger.info("   [Pass] clean_text logic verified.")

    # Load DataLoaders
    # We set load_cached_data=False to force processing of the DEBUG subset
    logger.info("   Generating DataLoaders (this may take a moment)...")
    train_loader, val_loader, test_loader, vocab, mlb = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Verify Vocabulary
    assert len(vocab) > 2, "Vocabulary should contain more than just PAD and UNK."
    encoded = vocab.encode("unknown_token_xyz", 5)
    assert len(encoded) == 5, "Vocabulary encoding length mismatch."
    assert (
        encoded[0] == vocab.token2idx[vocab.unk_token]
    ), "Unknown token not handled correctly."
    logger.info(f"   [Pass] Vocabulary built with {len(vocab)} tokens.")

    # Verify DataLoader shapes
    batch = next(iter(train_loader))
    assert "title" in batch and "body" in batch and "target" in batch
    assert batch["title"].shape == (Config.BATCH_SIZE, Config.MAX_TITLE_LEN)
    assert batch["body"].shape == (Config.BATCH_SIZE, Config.MAX_BODY_LEN)
    num_classes = len(mlb.classes_)
    assert batch["target"].shape == (Config.BATCH_SIZE, num_classes)
    logger.info(f"   [Pass] DataLoader yields correct shapes. Classes: {num_classes}")

    # ==========================================
    # 3. Model Initialization & Verification
    # ==========================================
    logger.info("3. Initializing Model...")

    model = DualStreamTextCNN(num_classes=num_classes)
    model.to(device)

    # Verify Forward Pass
    dummy_title = torch.zeros((2, Config.MAX_TITLE_LEN), dtype=torch.long).to(device)
    dummy_body = torch.zeros((2, Config.MAX_BODY_LEN), dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(dummy_title, dummy_body)

    assert logits.shape == (
        2,
        num_classes,
    ), f"Model output shape mismatch. Got {logits.shape}"
    logger.info("   [Pass] Model forward pass successful.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    logger.info("4. Running Training Loop (1 Epoch)...")

    # Setup Training Components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.NUM_EPOCHS,
        pct_start=0.1,
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=False, path=Config.MODEL_SAVE_PATH
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        early_stopping=early_stopping,
    )

    # Run Training
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Verify Checkpoint Creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    logger.info("   [Pass] Training completed and checkpoint saved.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    logger.info("5. Running Inference Pipeline...")

    # Load best model state
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict on Validation to find threshold
    logger.info("   Predicting on Validation set...")
    val_probs = predict_probs(model, val_loader, device)

    # Get sparse targets from dataset
    val_targets = val_loader.dataset.labels

    # Optimize Threshold
    best_threshold = optimize_threshold(val_probs, val_targets)
    assert 0.0 < best_threshold < 1.0, "Threshold optimization produced invalid value."

    # Predict on Test set
    logger.info("   Predicting on Test set...")
    test_probs = predict_probs(model, test_loader, device)
    test_ids = test_loader.dataset.ids

    # Generate Submission
    generate_submission(
        test_probs, test_ids, best_threshold, mlb, Config.SUBMISSION_FILE
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created."
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Check dimensions
    # Note: In DEBUG mode, test_loader only loads the sampled subset.
    # The get_dataloaders function samples the metadata DF if DEBUG is True.
    # So the submission file will have DEBUG_SAMPLE_SIZE rows (or less if test set is smaller).
    assert len(df_sub) > 0, "Submission file is empty."
    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission columns missing."

    # Check format of Tags (should be string, even if empty)
    if len(df_sub) > 0:
        first_tag = df_sub.iloc[0]["Tags"]
        assert isinstance(first_tag, str) or pd.isna(
            first_tag
        ), "Tags column format incorrect."

    logger.info(f"   [Pass] Submission generated with {len(df_sub)} rows.")
    logger.info("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
