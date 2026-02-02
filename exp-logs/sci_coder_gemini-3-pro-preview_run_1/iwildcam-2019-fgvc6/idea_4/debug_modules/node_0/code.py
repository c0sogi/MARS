import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_loaders
from library.model import AnimalModel, ModelEMA
from library.loss import FocalLoss, get_class_weights
from library.engine import train_one_epoch, validate, predict_and_submit

# Initialize logger for the demo script
logger = get_logger("demo_script")


def main():
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    logger.info("Step 1: Setting up configuration and environment...")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small sample size for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Set specific paths for this demo to avoid overwriting main run artifacts
    Config.WORKING_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    logger.info(f"Working directory set to: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    logger.info("\nStep 2: Demonstrating Data Loading...")

    # Get data loaders with debug=True
    train_loader, val_loader, test_loader = get_loaders(
        debug=True, load_cached_data=False
    )

    # Verify Train Loader
    try:
        images, labels = next(iter(train_loader))
        logger.info(
            f"Train Batch - Images Shape: {images.shape}, Labels Shape: {labels.shape}"
        )

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.INPUT_SIZE,
            Config.INPUT_SIZE,
        ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.INPUT_SIZE, Config.INPUT_SIZE)}, got {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
        assert labels.dtype == torch.long, "Labels should be of type torch.long"

        logger.info("Data Loading verification passed.")
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # ==========================================
    # 3. Model Initialization Demonstration
    # ==========================================
    logger.info("\nStep 3: Demonstrating Model Initialization...")

    device = Config.DEVICE
    model = AnimalModel(pretrained=False)  # False for speed in demo, usually True
    model.to(device)

    # Move batch to device
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass check
    outputs = model(images)
    logger.info(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"

    # EMA Initialization check
    ema_model = ModelEMA(model, decay=0.99, device=device)
    ema_model.update(model)
    logger.info("Model and EMA verification passed.")

    # ==========================================
    # 4. Loss Function Demonstration
    # ==========================================
    logger.info("\nStep 4: Demonstrating Loss Function...")

    # Load metadata to calculate weights (using the debug subset logic implicitly or loading full for weights)
    # In a real run, we load the full metadata for weights. Here we load the file directly.
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    class_weights = get_class_weights(df_train_meta)

    criterion = FocalLoss(weight=class_weights, gamma=Config.FOCAL_LOSS_GAMMA)

    loss = criterion(outputs, labels)
    logger.info(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Infinite"
    assert loss.item() > 0, "Loss should be positive"

    logger.info("Loss function verification passed.")

    # ==========================================
    # 5. Training Engine Demonstration
    # ==========================================
    logger.info("\nStep 5: Demonstrating Training Engine...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train one epoch
    logger.info("Running train_one_epoch...")
    epoch_loss, epoch_f1 = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1, ema_model=ema_model
    )

    logger.info(f"Train Result - Loss: {epoch_loss:.4f}, F1: {epoch_f1:.4f}")

    # Validate
    logger.info("Running validate...")
    # Use EMA model for validation if available
    val_model = ema_model.ema
    val_loss, val_f1 = validate(val_model, val_loader, criterion, device)

    logger.info(f"Validation Result - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

    # Assertions
    assert isinstance(epoch_loss, float), "Epoch loss should be float"
    assert 0 <= epoch_f1 <= 1.0, "F1 score should be between 0 and 1"

    logger.info("Training engine verification passed.")

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    logger.info("\nStep 6: Demonstrating Inference and Submission...")

    predict_and_submit(
        model=val_model,
        loader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    logger.info(f"Submission file created with {len(df_sub)} rows.")
    logger.info(f"First few rows:\n{df_sub.head()}")

    # Assertions on submission content
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns missing"
    # In debug mode, we subsampled the test set to DEBUG_SAMPLE_SIZE (100)
    # The loader might drop last if configured, but here drop_last=False for test.
    # We expect exactly DEBUG_SAMPLE_SIZE rows.
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(df_sub)}"

    logger.info("Inference verification passed.")

    logger.info("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
