import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import prepare_loaders
from library.model import DeepSupervisedModel, AWP
from library.engine import train_fn, valid_fn, inference_fn


def main():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print("Initializing configuration for demo...")

    # Override Config for speed and debugging
    Config.debug = True  # Uses small subset (200 train, 100 val/test)
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.epochs = 1
    Config.awp_start_epoch = 0  # Start AWP immediately to test logic
    Config.working_dir = "./working/demo_execution"
    Config.output_dir = os.path.join(Config.working_dir, "output")

    # Update cache paths to be inside the demo directory to avoid conflicts
    Config.train_cache_path = os.path.join(
        Config.working_dir, "cache", "train_cache.parquet"
    )
    Config.test_cache_path = os.path.join(
        Config.working_dir, "cache", "test_cache.parquet"
    )
    Config.val_cache_path = os.path.join(
        Config.working_dir, "cache", "val_cache.parquet"
    )

    # Create directories
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.train_cache_path), exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)
    logger = get_logger(os.path.join(Config.output_dir, "train.log"))

    print(f"Device: {Config.device}")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n--- Testing Data Loading ---")

    # Test Supervised Loaders
    train_loader, val_loader = prepare_loaders(
        stage="supervised",
        load_cached_data=False,  # Force reload to test raw reading
        debug=True,
    )

    # Verify Train Batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    mask = batch["attention_mask"]
    labels = batch["labels"]

    print(
        f"Batch shapes - Input: {input_ids.shape}, Mask: {mask.shape}, Labels: {labels.shape}"
    )

    # Assertions
    assert input_ids.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), "Incorrect input_ids shape"
    assert labels.shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), "Incorrect labels shape"
    assert labels.dtype == torch.float, "Labels should be float for BCE loss"

    # Test DAPT Loader (Masked Language Modeling)
    print("Testing DAPT (MLM) Loader...")
    dapt_loader = prepare_loaders(stage="dapt", debug=True)
    dapt_batch = next(iter(dapt_loader))

    assert "labels" in dapt_batch, "DAPT batch missing labels for MLM"
    assert dapt_batch["input_ids"].shape == (Config.train_batch_size, Config.max_len)
    print("Data loading verified.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n--- Testing Model Initialization ---")

    model = DeepSupervisedModel(pretrained=True)
    model.to(Config.device)

    # Move batch to device
    input_ids = input_ids.to(Config.device)
    mask = mask.to(Config.device)
    labels = labels.to(Config.device)

    print("Running forward pass...")
    outputs = model(input_ids, mask)

    # Verify outputs
    assert "main_logits" in outputs
    assert "aux_logits" in outputs
    assert outputs["main_logits"].shape == (Config.train_batch_size, Config.num_classes)
    assert outputs["aux_logits"].shape == (Config.train_batch_size, Config.num_classes)
    print("Forward pass successful.")

    # ==========================================
    # 4. Training Loop & AWP Verification
    # ==========================================
    print("\n--- Testing Training Loop with AWP ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler setup
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        total_steps=num_train_steps,
        pct_start=Config.pct_start,
    )

    # Initialize AWP
    awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

    # Run one epoch of training
    # Note: awp_start_epoch is set to 0, so AWP logic will execute
    avg_loss = train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=Config.device,
        awp=awp,
        logger=logger,
    )

    print(f"Training Epoch 0 Loss: {avg_loss:.4f}")
    assert np.isfinite(avg_loss), "Training loss is not finite"

    # ==========================================
    # 5. Validation Verification
    # ==========================================
    print("\n--- Testing Validation ---")

    val_loss, preds, targets = valid_fn(
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        device=Config.device,
        logger=logger,
    )

    print(f"Validation Loss: {val_loss:.4f}")

    # Check predictions shape
    # In debug mode, val set is 100 samples
    expected_val_size = 100
    assert (
        len(preds) == expected_val_size
    ), f"Expected {expected_val_size} predictions, got {len(preds)}"
    assert preds.shape == (expected_val_size, Config.num_classes)
    assert (preds >= 0).all() and (preds <= 1).all(), "Predictions not probabilities"

    # ==========================================
    # 6. Inference Verification
    # ==========================================
    print("\n--- Testing Inference ---")

    test_loader = prepare_loaders(stage="test", debug=True)
    test_preds = inference_fn(test_loader, model, Config.device)

    # In debug mode, test set is 100 samples
    expected_test_size = 100
    assert len(test_preds) == expected_test_size
    assert test_preds.shape == (expected_test_size, Config.num_classes)

    print("Inference successful.")

    # ==========================================
    # 7. Mock Submission Generation
    # ==========================================
    print("\n--- Generating Mock Submission ---")

    # Load sample submission to get IDs (using the debug slice logic implicitly)
    # In a real run, we'd use the full test set IDs.
    # Here we just demonstrate saving the file.

    # Create a dummy dataframe matching the predictions
    submission_df = pd.DataFrame(test_preds, columns=Config.target_cols)
    # Add dummy IDs
    submission_df.insert(0, "id", [f"id_{i}" for i in range(len(test_preds))])

    submission_path = os.path.join(Config.output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path)
    print(f"Submission saved to {submission_path}")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
