import os
import torch
import pandas as pd
import numpy as np
import logging
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, AverageMeter, get_logger
from library.transforms import get_transforms, MixUp, EEGTransform, SpectrogramTransform
from library.data import get_dataloaders, EEGDataset
from library.model import get_model
from library.engine import train_one_epoch, validate, inference


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> 1. Configuring Environment for Demo...")

    # Override Config for speed and offline execution
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.PRETRAINED = (
        False  # Disable downloading weights for demo speed/offline safety
    )
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead in small demo
    Config.OUTPUT_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Setup Logger
    logger = get_logger(os.path.join(Config.OUTPUT_DIR, "demo.log"))
    logger.info("Logger initialized successfully.")

    # Set Seeds
    seed_everything(Config.SEED)
    logger.info(f"Random seed set to {Config.SEED}")

    # Device
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n>>> 2. Verifying Utilities...")
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=2)
    # Average should be (10*2 + 20*2) / 4 = 15
    assert (
        meter.avg == 15.0
    ), f"AverageMeter logic failed. Expected 15.0, got {meter.avg}"
    logger.info("AverageMeter verified.")

    # ==========================================
    # 3. Verify Transforms (Unit Test)
    # ==========================================
    print("\n>>> 3. Verifying Transforms...")

    # Mock EEG Data: 10000 samples, 19 channels (columns)
    # We need column names matching Config.CHAIN_CONFIG
    eeg_cols = []
    for chain in Config.CHAIN_CONFIG.values():
        eeg_cols.extend(chain)
    eeg_cols = list(set(eeg_cols))  # Unique columns

    dummy_eeg_df = pd.DataFrame(
        np.random.randn(Config.EEG_SAMPLES, len(eeg_cols)), columns=eeg_cols
    )

    # Mock Spec Data: (4, 100, 300) -> 4 regions, Freq, Time
    dummy_spec_arr = np.random.rand(4, 100, 300).astype(np.float32)

    # Instantiate Transforms
    transforms = get_transforms(mode="train")

    # Test EEG Transform
    eeg_tensor = transforms["eeg"](dummy_eeg_df)
    # Expected: (4 views, 5 channels, 128 freq, 256 time)
    expected_eeg_shape = (4, 5, 128, 256)
    assert (
        eeg_tensor.shape == expected_eeg_shape
    ), f"EEG Transform shape mismatch. Expected {expected_eeg_shape}, got {eeg_tensor.shape}"

    # Test Spec Transform
    spec_tensor = transforms["spec"](dummy_spec_arr)
    # Expected: (4 regions, 256, 256)
    expected_spec_shape = (4, 256, 256)
    assert (
        spec_tensor.shape == expected_spec_shape
    ), f"Spec Transform shape mismatch. Expected {expected_spec_shape}, got {spec_tensor.shape}"

    logger.info("Transforms verified successfully.")

    # ==========================================
    # 4. Verify Data Loading
    # ==========================================
    print("\n>>> 4. Verifying Data Loading...")

    # Get DataLoaders (Debug mode loads a small subset)
    dataloaders = get_dataloaders(debug=True)
    train_loader = dataloaders["train"]

    logger.info(f"Train batches: {len(train_loader)}")

    # Fetch one batch
    batch_data, batch_targets = next(iter(train_loader))

    # Verify Batch Structure
    assert "eeg" in batch_data
    assert "spec" in batch_data
    assert "eeg_id" in batch_data

    # Verify Shapes in Batch
    # EEG: (Batch, 4, 5, 128, 256)
    assert batch_data["eeg"].shape == (
        Config.BATCH_SIZE,
        4,
        5,
        128,
        256,
    ), f"Batch EEG shape incorrect: {batch_data['eeg'].shape}"

    # Spec: (Batch, 4, 256, 256)
    assert batch_data["spec"].shape == (
        Config.BATCH_SIZE,
        4,
        256,
        256,
    ), f"Batch Spec shape incorrect: {batch_data['spec'].shape}"

    # Targets: (Batch, 6)
    assert batch_targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Batch Targets shape incorrect: {batch_targets.shape}"

    logger.info("Data Loading verified successfully.")

    # ==========================================
    # 5. Verify Model Architecture
    # ==========================================
    print("\n>>> 5. Verifying Model Architecture...")

    model = get_model(pretrained=Config.PRETRAINED)
    model.to(device)

    # Run dummy forward pass with the batch we just loaded
    eeg_input = batch_data["eeg"].to(device)
    spec_input = batch_data["spec"].to(device)

    with torch.no_grad():
        logits = model(eeg_input, spec_input)

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    logger.info("Model forward pass verified successfully.")

    # ==========================================
    # 6. Verify Training Loop (Engine)
    # ==========================================
    print("\n>>> 6. Verifying Training Loop...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    mixup = MixUp(alpha=0.2, prob=0.5)

    # Train for 1 epoch
    avg_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=1,
        mixup_fn=mixup,
    )

    logger.info(f"Training Epoch 1 completed. Avg Loss: {avg_loss:.4f}")

    # Validate
    val_loader = dataloaders["val"]
    val_loss = validate(model, val_loader, device)
    logger.info(f"Validation completed. Avg Loss: {val_loss:.4f}")

    # ==========================================
    # 7. Verify Inference
    # ==========================================
    print("\n>>> 7. Verifying Inference...")

    # Create a dummy test loader if test set exists, otherwise use val loader as proxy
    if "test" in dataloaders:
        test_loader = dataloaders["test"]
    else:
        logger.info(
            "Test set not found (likely missing test.csv), using validation set for inference demo."
        )
        test_loader = val_loader

    inference(
        model=model,
        dataloader=test_loader,
        device=device,
        save_path=Config.SUBMISSION_PATH,
    )

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        list(sub_df.columns) == ["eeg_id"] + Config.CLASS_NAMES
    ), "Submission columns mismatch."
    assert len(sub_df) > 0, "Submission file is empty."

    # Check probabilities sum to 1 (approx)
    row_sums = sub_df[Config.CLASS_NAMES].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1."

    logger.info(f"Inference verified. Submission saved to {Config.SUBMISSION_PATH}")
    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
