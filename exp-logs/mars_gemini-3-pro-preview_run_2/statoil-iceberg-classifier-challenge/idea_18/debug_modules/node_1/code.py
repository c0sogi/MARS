import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.utils import seed_everything, get_logger
from library.data_loader import (
    process_data,
    get_folds,
    get_dataloaders,
    get_test_loader,
)
from library.model import WB_DSN
from library.trainer import ModelTrainer


def run_demo():
    # 1. Setup
    # ----------------------------------------------------------------
    seed_everything(42)
    logger = get_logger("DemoRunner", log_file="./working/demo.log")
    logger.info("Starting demonstration script...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Data Loading & Processing
    # ----------------------------------------------------------------
    logger.info("Step 1: Processing Data...")
    # process_data handles loading JSONs, creating 3-channel images, normalization, and caching.
    # We set load_cached_data=False to demonstrate the processing logic,
    # or True if we want to rely on existing cache (if any).
    # Given the environment, we'll try to load or process.
    train_data, test_data = process_data(load_cached_data=True)

    # Assertions to verify data integrity
    assert "images" in train_data
    assert "angles" in train_data
    assert "labels" in train_data
    assert train_data["images"].shape[1] == 3, "Images should have 3 channels"
    assert train_data["images"].shape[2] == 75, "Images should be 75x75"
    logger.info(f"Train data shape: {train_data['images'].shape}")
    logger.info(f"Test data shape: {test_data['images'].shape}")

    # 3. Fold Generation
    # ----------------------------------------------------------------
    logger.info("Step 2: Generating Folds...")
    folds = get_folds(train_data, n_splits=5, seed=42)
    logger.info(f"Generated {len(folds)} folds.")

    # Get DataLoaders for Fold 0
    train_loader, val_loader = get_dataloaders(
        fold_idx=0, folds=folds, train_data=train_data, batch_size=32, num_workers=2
    )

    # Verify DataLoader yields correct shapes
    sample_inputs, sample_targets = next(iter(train_loader))
    sample_imgs, sample_angles = sample_inputs
    logger.info(f"Batch Image Shape: {sample_imgs.shape}")
    logger.info(f"Batch Angle Shape: {sample_angles.shape}")
    assert sample_imgs.shape == (32, 3, 75, 75)
    assert sample_angles.shape == (32, 1)

    # 4. Model Initialization
    # ----------------------------------------------------------------
    logger.info("Step 3: Initializing Model...")
    model = WB_DSN().to(device)

    # Verify Forward Pass with dummy data
    with torch.no_grad():
        dummy_img = torch.randn(2, 3, 75, 75).to(device)
        dummy_ang = torch.randn(2, 1).to(device)
        dummy_out = model(dummy_img, dummy_ang)
        assert dummy_out.shape == (
            2,
            1,
        ), f"Expected output (2, 1), got {dummy_out.shape}"
    logger.info("Model forward pass verification successful.")

    # 5. Training Loop Demonstration
    # ----------------------------------------------------------------
    logger.info("Step 4: Training (Short Demo)...")
    trainer = ModelTrainer(model, device, logger=logger, learning_rate=1e-4)

    # Run for just 2 epochs to demonstrate functionality quickly
    best_val_loss = trainer.fit(train_loader, val_loader, epochs=2, patience=2)
    logger.info(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")

    # 6. Inference
    # ----------------------------------------------------------------
    logger.info("Step 5: Running Inference on Test Set...")
    test_loader = get_test_loader(test_data, batch_size=32, num_workers=2)

    predictions = trainer.predict(test_loader)

    assert len(predictions) == len(
        test_data["ids"]
    ), f"Prediction count {len(predictions)} mismatch with test IDs {len(test_data['ids'])}"

    logger.info(f"Generated {len(predictions)} predictions.")

    # 7. Submission File Generation
    # ----------------------------------------------------------------
    logger.info("Step 6: Generating Submission File...")
    submission_df = pd.DataFrame({"id": test_data["ids"], "is_iceberg": predictions})

    output_path = "./working/demo_submission.csv"
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    # Verify file creation
    assert os.path.exists(output_path)
    logger.info("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
