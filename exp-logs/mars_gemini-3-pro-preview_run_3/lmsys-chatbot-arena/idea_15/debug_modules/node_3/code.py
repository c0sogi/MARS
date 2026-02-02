import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data_processing import get_dataloaders
from library.model_components import SiameseDeberta
from library.engine import train_fn, inference_fn


def run_demo():
    print("==== Starting Library Demonstration ====")

    # 1. Setup and Configuration Overrides
    # We override Config values to ensure the demo runs quickly and uses the debug subset.
    print("\n[1] Configuring environment...")
    seed_everything(42)

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 2
    Config.VALID_BATCH_SIZE = 4
    Config.GRAD_ACCUM_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists for demo artifacts
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Processing Demonstration
    print("\n[2] Testing Data Loading...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        debug=Config.DEBUG,
        batch_size=Config.TRAIN_BATCH_SIZE,
        val_batch_size=Config.VALID_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    print("Batch Keys:", batch.keys())

    # Assertions for batch structure
    required_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalars",
        "ids",
        "labels",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Shapes
    # Labels should be (Batch, 3)
    assert batch["labels"].shape == (
        Config.TRAIN_BATCH_SIZE,
        3,
    ), f"Incorrect label shape: {batch['labels'].shape}"
    # Scalars should be (Batch, 3)
    assert batch["scalars"].shape == (
        Config.TRAIN_BATCH_SIZE,
        3,
    ), f"Incorrect scalar shape: {batch['scalars'].shape}"

    print("Data Loading Verification Passed.")

    # 3. Model Initialization & Forward Pass
    print("\n[3] Testing Model Architecture...")
    model = SiameseDeberta()
    model.to(Config.DEVICE)

    # Move batch to device
    input_ids_a = batch["input_ids_a"].to(Config.DEVICE)
    mask_a = batch["attention_mask_a"].to(Config.DEVICE)
    input_ids_b = batch["input_ids_b"].to(Config.DEVICE)
    mask_b = batch["attention_mask_b"].to(Config.DEVICE)
    scalars = batch["scalars"].to(Config.DEVICE)

    # Run Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids_a, mask_a, input_ids_b, mask_b, scalars)

    print(f"Logits Shape: {logits.shape}")

    # Assertions for Model Output
    assert logits.shape == (Config.TRAIN_BATCH_SIZE, 3), "Logits shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN logits"

    print("Model Architecture Verification Passed.")

    # 4. Training Loop Demonstration
    print("\n[4] Testing Training Loop (Engine)...")

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)
    )

    # Run one epoch of training
    # This tests: Mixed Precision, Gradient Accumulation, Loss Calculation
    train_loss = train_fn(
        model, train_loader, optimizer, scheduler, Config.DEVICE, epoch=0
    )

    print(f"Training Loss after 1 epoch: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    print("Training Loop Verification Passed.")

    # 5. Inference Demonstration
    print("\n[5] Testing Inference (TTA)...")

    # Run inference on the small test set
    ids, preds = inference_fn(model, test_loader, Config.DEVICE)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Sample Prediction: {preds[0]}")

    # Assertions
    assert len(ids) == len(preds), "Mismatch between IDs and Predictions"
    assert preds.shape[1] == 3, "Predictions should have 3 columns (Win_A, Win_B, Tie)"

    # Check probability properties (Sum close to 1)
    row_sums = np.sum(preds, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("Inference Verification Passed.")

    # 6. Metric Utility Verification
    print("\n[6] Testing Metric Calculation...")

    # Create dummy ground truth and predictions
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    loss = compute_log_loss(y_true, y_pred)
    print(f"Computed Log Loss: {loss:.4f}")

    assert loss < 0.7, "Log loss should be low for good predictions"
    assert loss > 0, "Log loss must be positive"

    print("Metric Verification Passed.")

    print("\n==== All Demonstrations Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
