import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import shutil
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import DebertaV3MultiTask
from library.loss import HybridLoss
from library.awp import AWP
from library.engine import train_one_epoch, validate, inference
from library.metrics import calculate_final_score


def main():
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print(">>> Step 1: Configuring Environment for Demo")

    # Override Config for speed and demonstration purposes
    # Using a tiny model to ensure this runs in seconds rather than hours
    Config.MODEL_NAME = "prajjwal1/bert-tiny"
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.DEBUG = True  # Subsamples data to ~2000 rows
    Config.ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Enable AWP immediately for demonstration
    Config.USE_AWP = True
    Config.AWP_START_EPOCH = 0

    # Set up output directory
    Config.EXP_NAME = "demo_run"
    Config.OUTPUT_DIR = f"./working/{Config.EXP_NAME}/"
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    logger = get_logger(os.path.join(Config.OUTPUT_DIR, "demo.log"))
    logger.info("Configuration overridden for fast demonstration.")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n>>> Step 2: Verifying Data Pipeline")

    # Get dataloaders (this handles tokenization and dataset creation)
    # We set load_cached_data=False to force processing logic to run
    train_loader, val_loader, test_loader, val_df = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=False
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))

    # Assertions to check batch integrity
    assert "ids" in batch, "Batch missing 'ids'"
    assert "mask" in batch, "Batch missing 'mask'"
    assert "target" in batch, "Batch missing 'target'"
    assert "aux_targets" in batch, "Batch missing 'aux_targets'"
    assert "weight" in batch, "Batch missing 'weight'"

    # Check shapes
    batch_size = batch["ids"].size(0)
    seq_len = batch["ids"].size(1)
    assert (
        batch_size == Config.TRAIN_BATCH_SIZE
    ), f"Expected batch size {Config.TRAIN_BATCH_SIZE}, got {batch_size}"
    assert (
        seq_len == Config.MAX_LEN
    ), f"Expected seq len {Config.MAX_LEN}, got {seq_len}"

    logger.info("Data Pipeline verified successfully.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n>>> Step 3: Verifying Model Architecture")

    device = Config.DEVICE
    model = DebertaV3MultiTask(pretrained_model_name=Config.MODEL_NAME)
    model.to(device)

    # Move batch to device
    ids = batch["ids"].to(device)
    mask = batch["mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    # Forward pass
    outputs = model(ids, mask, token_type_ids)

    # Verify output keys
    assert "toxicity_logits" in outputs
    assert "identity_logits" in outputs
    assert "attack_logits" in outputs

    # Verify output shapes
    # Toxicity: (B, 1)
    assert outputs["toxicity_logits"].shape == (batch_size, 1)
    # Identity: (B, num_identities)
    assert outputs["identity_logits"].shape == (batch_size, len(Config.IDENTITY_COLS))
    # Attack: (B, num_aux_cols)
    assert outputs["attack_logits"].shape == (batch_size, len(Config.AUX_COLS))

    logger.info("Model architecture verified successfully.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n>>> Step 4: Verifying Hybrid Loss")

    criterion = HybridLoss()

    # Prepare batch data for loss
    batch_data = {
        "target": batch["target"].to(device),
        "aux_targets": batch["aux_targets"].to(device),
        "weight": batch["weight"].to(device),
    }

    loss_dict = criterion(outputs, batch_data)

    assert "loss" in loss_dict
    assert "loss_main" in loss_dict
    assert "loss_rank" in loss_dict
    assert "loss_aux" in loss_dict

    total_loss = loss_dict["loss"]
    assert not torch.isnan(total_loss).any(), "Loss contains NaNs"
    assert total_loss.item() > 0, "Loss should be positive"

    # Verify gradients can be computed
    total_loss.backward()
    logger.info("Hybrid Loss calculation and backward pass verified.")

    # ==========================================
    # 5. Training Loop Demonstration (with AWP)
    # ==========================================
    print("\n>>> Step 5: Running Training Loop (1 Epoch)")

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS * len(train_loader))

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
    )

    # Run one epoch
    # This exercises the engine.py logic, including mixed precision and AWP steps
    epoch_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=0,
        awp=awp,
    )

    logger.info(f"Training loop complete. Epoch Loss: {epoch_loss:.4f}")

    # ==========================================
    # 6. Validation and Metrics
    # ==========================================
    print("\n>>> Step 6: Running Validation")

    # Run validation loop
    val_score, val_loss = validate(model, val_loader, device, val_df)

    logger.info(f"Validation complete. Score: {val_score:.4f}, Loss: {val_loss:.4f}")

    # Manual check of metric calculation logic with synthetic data
    print("  Performing manual metric sanity check...")

    # Create a synthetic dataframe representing a tricky case
    # 10 samples: 5 Toxic, 5 Non-Toxic
    # Identity 'male' present in some
    synth_data = {
        "id": range(10),
        "target": [
            0.9,
            0.8,
            0.1,
            0.0,
            0.9,
            0.1,
            0.8,
            0.2,
            0.1,
            0.0,
        ],  # 0.5 threshold -> 1, 1, 0, 0, 1, 0, 1, 0, 0, 0
        "male": [1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],  # Mentions
    }
    # Add other identity cols as copies of 'male' so all subgroups are populated
    # This ensures we don't get 0.5 default scores for empty subgroups
    for col in Config.IDENTITY_COLS:
        if col != "male":
            synth_data[col] = synth_data["male"]

    synth_df = pd.DataFrame(synth_data)

    # Perfect predictions
    perfect_preds = np.array([0.9, 0.8, 0.1, 0.0, 0.9, 0.1, 0.8, 0.2, 0.1, 0.0])
    score, metrics = calculate_final_score(synth_df, perfect_preds)

    assert score == 1.0, f"Perfect predictions should yield score 1.0, got {score}"
    logger.info("Metric calculation logic verified.")

    # ==========================================
    # 7. Inference and Submission
    # ==========================================
    print("\n>>> Step 7: Running Inference")

    submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    inference(model, test_loader, device, submission_path=submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created"
    sub_df = pd.read_csv(submission_path)

    # Check format
    assert "id" in sub_df.columns and "prediction" in sub_df.columns
    assert len(sub_df) == len(pd.read_csv(Config.SAMPLE_SUBMISSION_PATH))

    logger.info(f"Inference complete. Submission saved to {submission_path}")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
