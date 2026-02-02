import os
import sys
import shutil
import warnings
import logging
import numpy as np
import pandas as pd
import torch

# Suppress warnings and verbose logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_dataloaders
from library.model import SiameseRoBERTa
from library.trainer import Trainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for a fast demonstration run
    # We use a separate working directory to avoid messing with existing caches
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_SAVE_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Reduce compute requirements for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VAL_BATCH_SIZE = 8
    Config.PATIENCE = 1

    # Create necessary directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.create_dirs()

    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading data (Debug Mode)...")

    # get_dataloaders with debug=True loads a small subset (100 samples)
    # We set load_cached_data=False to ensure we demonstrate processing logic
    # (though on a tiny subset it's fast)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=None, load_cached_data=False, debug=True
    )

    # Verification: Check if loaders have data
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Inspect a single batch
    sample_batch = next(iter(train_loader))
    print(f"Batch keys: {list(sample_batch.keys())}")
    print(f"Input IDs shape: {sample_batch['q_input_ids'].shape}")
    print(f"Labels shape: {sample_batch['labels'].shape}")

    assert (
        sample_batch["q_input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.TRAIN_BATCH_SIZE}, got {sample_batch['q_input_ids'].shape[0]}"

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing Model...")
    model = SiameseRoBERTa()
    model.to(Config.DEVICE)

    # Verification: Dummy Forward Pass
    print("Running dummy forward pass...")
    model.eval()
    with torch.no_grad():
        q_ids = sample_batch["q_input_ids"].to(Config.DEVICE)
        q_mask = sample_batch["q_attention_mask"].to(Config.DEVICE)
        a_ids = sample_batch["a_input_ids"].to(Config.DEVICE)
        a_mask = sample_batch["a_attention_mask"].to(Config.DEVICE)

        outputs = model(q_ids, q_mask, a_ids, a_mask)

    print(f"Output shape: {outputs.shape}")
    assert outputs.shape == (
        Config.TRAIN_BATCH_SIZE,
        30,
    ), f"Output shape mismatch. Expected ({Config.TRAIN_BATCH_SIZE}, 30), got {outputs.shape}"
    assert (outputs >= 0).all() and (
        outputs <= 1
    ).all(), "Model outputs are not in range [0, 1] (Sigmoid check failed)."

    # ==========================================
    # 4. Training
    # ==========================================
    print("\nStarting Training Loop...")
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # Run training (1 epoch due to config override)
    trainer.fit()

    # Verification: Check if model was saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("Training complete. Best model saved.")

    # ==========================================
    # 5. Prediction & Metric Verification
    # ==========================================
    print("\nGenerating Predictions...")

    # Predict on Validation set to verify metric calculation
    val_preds = trainer.predict(val_loader)

    # Get actual targets from the dataset (since debug=True, we need the subset targets)
    # The dataset object stores the full array of the subset
    val_targets = val_loader.dataset.targets

    # Compute metric
    score = compute_spearman_metric(val_preds, val_targets)
    print(f"Validation Spearman Correlation: {score:.4f}")

    # Predict on Test set for submission
    test_preds = trainer.predict(test_loader)
    print(f"Test Predictions Shape: {test_preds.shape}")

    assert test_preds.shape[1] == 30, "Test predictions must have 30 columns."

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\nCreating Submission File...")

    # Load sample submission to get IDs
    # Note: Since we used debug=True, the test_loader only has 100 samples.
    # We need to map these to the correct QA IDs.
    # The get_dataloaders function caches 'test_qa_ids.npy'.

    test_qa_ids_path = os.path.join(Config.WORKING_DIR, "test_qa_ids.npy")
    if os.path.exists(test_qa_ids_path):
        all_test_ids = np.load(test_qa_ids_path)
        # Because of debug=True, the loader only iterated over the first 100
        current_test_ids = all_test_ids[: len(test_preds)]
    else:
        # Fallback if cache logic differs (shouldn't happen with provided code)
        # We just create dummy IDs for the sake of the demo if file missing
        current_test_ids = np.arange(len(test_preds))

    submission_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "qa_id", current_test_ids)

    submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")
    print(f"Submission Head:\n{submission_df.head(3)}")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
