import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, compute_metrics
from library.data_processing import get_dataloaders
from library.model import SiameseDeberta
from library.engine import train_fn, eval_fn, inference_fn


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("--- Setting up configuration for demo run ---")
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small sample size for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRAD_ACCUM_STEPS = 1  # Simplify for demo
    Config.NUM_WORKERS = 2

    # Ensure working directories exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    logger = get_logger("demo_script")
    logger.info("Configuration updated for demo execution.")

    # 2. Data Loading
    print("\n--- Initializing DataLoaders ---")
    # We force load_cached_data=False to demonstrate processing logic on the subset
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verification: Check Batch Structure
    sample_batch = next(iter(train_loader))
    required_keys = [
        "input_ids_a",
        "attention_mask_a",
        "token_type_ids_a",
        "input_ids_b",
        "attention_mask_b",
        "token_type_ids_b",
        "scalars",
        "target",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Missing key {key} in batch"

    # Verify shapes
    batch_size = sample_batch["input_ids_a"].size(0)
    seq_len = sample_batch["input_ids_a"].size(1)
    assert (
        batch_size == Config.TRAIN_BATCH_SIZE
    ), f"Expected batch size {Config.TRAIN_BATCH_SIZE}, got {batch_size}"
    assert (
        seq_len == Config.MAX_LENGTH
    ), f"Expected sequence length {Config.MAX_LENGTH}, got {seq_len}"
    assert (
        sample_batch["target"].size(1) == 3
    ), "Target should have 3 columns (Model A, Model B, Tie)"

    logger.info(f"DataLoaders initialized. Train batches: {len(train_loader)}")

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    model = SiameseDeberta()
    model.to(Config.DEVICE)

    # Verification: Check Model Output Shape
    # Run a dummy forward pass with the sample batch (without grad)
    model.eval()
    with torch.no_grad():
        # Move batch to device
        inputs = {
            k: v.to(Config.DEVICE) for k, v in sample_batch.items() if k != "target"
        }
        logits = model(**inputs)
        assert logits.shape == (
            batch_size,
            3,
        ), f"Expected output shape ({batch_size}, 3), got {logits.shape}"

    logger.info("Model initialized and forward pass verified.")

    # 4. Training Setup
    print("\n--- Starting Training Loop ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate training steps
    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    scaler = torch.amp.GradScaler("cuda")

    # Run Training for 1 Epoch
    avg_train_loss = train_fn(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        scaler=scaler,
        epoch=1,
    )

    assert not np.isnan(avg_train_loss), "Training loss returned NaN"
    logger.info(f"Training complete. Average Loss: {avg_train_loss:.4f}")

    # 5. Evaluation
    print("\n--- Starting Evaluation ---")
    val_loss, val_preds = eval_fn(model, val_loader, Config.DEVICE)

    # Verification: Check Predictions
    assert val_preds.shape[1] == 3, "Validation predictions must have 3 columns"
    assert len(val_preds) == len(
        val_loader.dataset
    ), "Prediction count mismatch with validation dataset"

    # Check if probabilities sum to approx 1
    row_sums = np.sum(val_preds, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    logger.info(f"Evaluation complete. Val Loss: {val_loss:.4f}")

    # 6. Inference on Test Set
    print("\n--- Starting Inference ---")
    test_preds = inference_fn(model, test_loader, Config.DEVICE)

    assert test_preds.shape[1] == 3, "Test predictions must have 3 columns"
    assert len(test_preds) == len(
        test_loader.dataset
    ), "Prediction count mismatch with test dataset"

    logger.info(f"Inference complete. Generated {len(test_preds)} predictions.")

    # 7. Submission Generation
    print("\n--- Generating Submission File ---")
    # Load test metadata to get IDs (using the subset logic if debug was applied to loading,
    # but get_dataloaders debug mode slices the dataframe. We need the IDs corresponding to that slice.)
    # Since get_dataloaders slices the head, we can read the test csv and slice the head too.
    test_df = pd.read_csv(Config.TEST_PATH)
    if Config.DEBUG:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": test_preds[:, 0],
            "winner_model_b": test_preds[:, 1],
            "winner_tie": test_preds[:, 2],
        }
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verification: Check saved file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert saved_df.shape == (
        len(test_df),
        4,
    ), f"Submission shape mismatch. Expected {(len(test_df), 4)}, got {saved_df.shape}"
    assert list(saved_df.columns) == [
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ], "Incorrect columns in submission"

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
