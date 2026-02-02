import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.transforms import get_transforms
from library.dataset import get_dataset
from library.model import get_model
from library.engine import train_one_epoch, validate, inference


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.OUTPUT_DIR = "./working/demo_submission"
    Config.BATCH_SIZE = 8
    Config.PHASE1_EPOCHS = 1
    Config.NUM_WORKERS = 2
    # Ensure directories exist
    Config.setup_directories()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Setup Logger
    logger = get_logger("demo.log")
    logger.info("Starting Cassava Leaf Disease Classification Demo...")

    # Device configuration
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading (Debug Mode)
    # -------------------------------------------------------------------------
    logger.info("Initializing Datasets (Debug Mode)...")

    # Training Dataset
    train_dataset = get_dataset(
        phase="train",
        transform=get_transforms("train", Config.PHASE1_IMG_SIZE),
        debug=True,  # Loads only 100 samples
    )

    # Validation Dataset
    val_dataset = get_dataset(
        phase="val", transform=get_transforms("val", Config.PHASE1_IMG_SIZE), debug=True
    )

    # Test Dataset (for Inference)
    test_dataset = get_dataset(
        phase="test",
        transform=get_transforms("inference", Config.PHASE1_IMG_SIZE),
        debug=True,
        return_id=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Verification: Check batch shapes
    dummy_images, dummy_labels = next(iter(train_loader))
    assert dummy_images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.PHASE1_IMG_SIZE,
        Config.PHASE1_IMG_SIZE,
    ), f"Incorrect image batch shape: {dummy_images.shape}"
    assert dummy_labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label batch shape: {dummy_labels.shape}"
    logger.info("DataLoaders initialized and verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    logger.info("Initializing Model...")

    # We use pretrained=False to avoid downloading weights during this demo execution
    # if internet is restricted, but the code structure supports pretrained=True.
    # For the purpose of verifying logic, random initialization is sufficient.
    model, model_ema = get_model(
        model_name=Config.MODEL_BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        use_ema=True,
    )

    # Verification: Forward pass
    model.eval()
    with torch.no_grad():
        dummy_output = model(dummy_images.to(device))

    assert dummy_output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {dummy_output.shape}"
    logger.info("Model initialized and verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    logger.info("Starting Training Loop Demo (1 Epoch)...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch
    train_metrics = train_one_epoch(
        model=model,
        criterion=criterion,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=0,
        model_ema=model_ema,
        mixup_fn=None,  # Skipping mixup for simple demo
    )

    logger.info(f"Training completed. Metrics: {train_metrics}")
    assert "train_loss" in train_metrics, "train_one_epoch did not return train_loss"
    assert train_metrics["train_loss"] > 0, "Training loss should be positive"

    # -------------------------------------------------------------------------
    # 5. Validation Demonstration
    # -------------------------------------------------------------------------
    logger.info("Starting Validation Demo...")

    val_metrics = validate(
        model=model, criterion=criterion, data_loader=val_loader, device=device
    )

    logger.info(f"Validation completed. Metrics: {val_metrics}")
    assert "val_acc1" in val_metrics, "validate did not return val_acc1"

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    logger.info("Starting Inference Demo...")

    # Use the EMA model for inference if available, otherwise the main model
    inference_model = model_ema.module if model_ema else model

    predictions = inference(
        model=inference_model, data_loader=test_loader, device=device
    )

    # Verification: Check predictions format
    assert isinstance(predictions, list), "Inference should return a list"
    if len(predictions) > 0:
        assert (
            "image_id" in predictions[0] and "label" in predictions[0]
        ), "Prediction items must have 'image_id' and 'label' keys"
        assert isinstance(predictions[0]["label"], int), "Label should be an integer"

    logger.info(f"Inference generated {len(predictions)} predictions.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    logger.info("Generating Submission File...")

    df_sub = pd.DataFrame(predictions)

    # Ensure we cover the sample submission requirements (though we used debug subset)
    # In a real scenario, we would predict on the full test set.
    # Here we just save what we have.
    submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to {submission_path}")

    # Verify file existence
    assert os.path.exists(submission_path), "Submission file was not created"

    logger.info("Demo execution completed successfully.")


if __name__ == "__main__":
    # Filter warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
