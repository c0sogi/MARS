import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import get_model
from library.trainer import Trainer


def main():
    # 1. Configuration
    # Initialize config in debug mode for fast execution (subset of data, fewer epochs)
    print("Initializing Configuration...")
    cfg = Config(debug=True, epochs=2, batch_size=32)

    # Ensure reproducibility
    cfg.seed_everything()

    print(f"Device: {cfg.device}")
    print(f"Debug Mode: {cfg.debug}")
    print(f"Batch Size: {cfg.batch_size}")

    # 2. Data Loading
    print("\nSetting up DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Validation: Check if dataloaders are not empty and yield correct shapes
    try:
        images, labels = next(iter(train_loader))
        print(f"Train Batch Shape: Images {images.shape}, Labels {labels.shape}")

        # Assert image shape is (Batch, 3, H, W)
        assert images.shape == (
            cfg.batch_size,
            3,
            cfg.img_size,
            cfg.img_size,
        ), f"Expected image shape {(cfg.batch_size, 3, cfg.img_size, cfg.img_size)}, got {images.shape}"

        # Assert label shape is (Batch)
        assert labels.shape == (
            cfg.batch_size,
        ), f"Expected label shape {(cfg.batch_size,)}, got {labels.shape}"

    except StopIteration:
        raise RuntimeError("Train DataLoader is empty!")

    # 3. Model Initialization
    print("\nInitializing Model...")
    model = get_model(cfg)

    # Validation: Check model device and output head
    # Timm Swin Transformer usually has a classifier head named 'head'
    if hasattr(model, "head"):
        # For Swin Transformer
        out_features = (
            model.head.fc.out_features
            if hasattr(model.head, "fc")
            else model.head.out_features
        )
        assert (
            out_features == cfg.num_classes
        ), f"Model output features {out_features} does not match num_classes {cfg.num_classes}"

    # Check if model is on the correct device
    param_device = next(model.parameters()).device
    assert (
        param_device == cfg.device
    ), f"Model is on {param_device}, expected {cfg.device}"

    print(f"Model {cfg.model_name} initialized successfully on {cfg.device}.")

    # 4. Optimizer and Scheduler Setup
    print("\nSetting up Optimizer and Scheduler...")
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    # 5. Trainer Initialization and Training
    print("\nStarting Training Loop...")
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Execute training
    trainer.fit()

    # Validation: Check if best model checkpoint was saved
    if not os.path.exists(cfg.best_model_path):
        raise FileNotFoundError(
            f"Best model not found at {cfg.best_model_path} after training."
        )
    print(f"Training completed. Best model saved at {cfg.best_model_path}")

    # 6. Prediction / Inference
    print("\nGenerating Predictions...")
    trainer.predict()

    # Validation: Check submission file
    if not os.path.exists(cfg.submission_path):
        raise FileNotFoundError(f"Submission file not found at {cfg.submission_path}")

    # Validate submission content
    submission_df = pd.read_csv(cfg.submission_path)
    print(f"Submission generated with {len(submission_df)} rows.")

    assert (
        "Id" in submission_df.columns and "Predicted" in submission_df.columns
    ), "Submission file missing required columns 'Id' or 'Predicted'"

    # In debug mode, the test set is subsetted to debug_sample_size (2000)
    # However, DataLoader drop_last=False for test, so we should have exactly debug_sample_size rows
    expected_rows = cfg.debug_sample_size
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(submission_df)}"

    print("\nWorkflow completed successfully!")


if __name__ == "__main__":
    main()
