import os
import torch
import pandas as pd
import numpy as np
import random
import library.config as config
import library.data_loader as data_loader
import library.model as model_lib
import library.engine as engine
import library.utils as utils


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # 1. Setup and Configuration Override
    set_seed(42)

    print("Configuring for quick demonstration...")
    # Override config for speed
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 200  # Small sample for quick execution
    config.BATCH_SIZE = 16
    config.EPOCHS = 1
    config.NUM_WORKERS = (
        2  # Reduce worker overhead to prevent potential shared memory lag in demo
    )

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = data_loader.get_loaders()

    # Validation: Check Train Loader
    try:
        images, targets = next(iter(train_loader))
        print(f"Train Batch: Images {images.shape}, Targets {targets.shape}")

        # Assertions
        assert images.shape == (
            config.BATCH_SIZE,
            3,
            config.IMG_SIZE,
            config.IMG_SIZE,
        ), f"Expected train images shape {(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)}, got {images.shape}"
        assert targets.shape == (
            config.BATCH_SIZE,
        ), f"Expected train targets shape {(config.BATCH_SIZE,)}, got {targets.shape}"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Validation: Check Val Loader (Batch size 1, variable N images)
    try:
        val_images, val_targets = next(iter(val_loader))
        # val_images shape: (1, N, C, H, W)
        print(f"Val Batch: Images {val_images.shape}")
        assert val_images.dim() == 5, "Val images should be 5D tensor (B, N, C, H, W)"
        assert val_images.size(0) == 1, "Val batch size should be 1"
    except StopIteration:
        raise AssertionError("Val loader is empty!")

    # 3. Model Initialization
    print("Initializing Model...")
    model = model_lib.get_model()

    # Validation: Check Model Output Shape
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {config.NUM_CLASSES}), got {output.shape}"
    print("Model initialized and verified.")

    # 4. Training Loop Execution
    print("Starting Training Demonstration...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, total_steps=len(train_loader) * config.EPOCHS
    )

    trainer = engine.Trainer(model, optimizer, scheduler, config.DEVICE)

    # Run one epoch
    train_acc, train_loss = trainer.train_one_epoch(train_loader, epoch=1)

    # Validation: Check Training Metrics
    print(f"Training Completed. Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0 <= train_acc <= 100, "Training accuracy out of bounds"

    # 5. Validation Loop Execution
    print("Starting Validation Demonstration...")
    val_acc = trainer.validate(val_loader)
    print(f"Validation Completed. Acc: {val_acc:.4f}")
    assert 0 <= val_acc <= 100, "Validation accuracy out of bounds"

    # 6. Inference Execution
    print("Starting Inference Demonstration...")
    engine.inference(model, test_loader, config.DEVICE)

    # Validation: Check Submission File
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file not found"

    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Check columns
    required_cols = ["_id", "category_id"]
    for col in required_cols:
        assert col in df_sub.columns, f"Missing column {col} in submission"

    # Check content
    assert len(df_sub) > 0, "Submission file is empty"

    print("Demonstration completed successfully.")
