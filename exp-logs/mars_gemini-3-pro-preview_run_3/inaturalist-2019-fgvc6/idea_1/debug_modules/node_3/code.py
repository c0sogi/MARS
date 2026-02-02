import os
import sys
import torch
import pandas as pd
import numpy as np
import random

# Import provided library modules
from library.config import Config
from library.dataset import INatDataset, get_transforms
from library.model import INatModel
from library.trainer import Trainer
from library.inference import generate_submission


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("=== Starting iNaturalist 2019 Library Demo ===")

    # 1. Configuration Override for Speed
    # We modify the Config class attributes directly to run a fast demonstration.
    print("\n[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 images for demo
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.VAL_BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce worker overhead

    # Ensure directories exist (Config.setup() does this, but good to verify)
    Config.setup()
    set_seed(Config.SEED)

    # 2. Dataset Verification
    print("\n[2] Verifying Dataset and Transforms...")
    # Load metadata manually to test Dataset class
    train_df = pd.read_csv(Config.TRAIN_METADATA).head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Dataset
    dataset = INatDataset(train_df, transforms=get_transforms("train"))

    # Check length
    assert (
        len(dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(dataset)}"

    # Check item retrieval
    img_tensor, target, img_id = dataset[0]

    # Verify shapes
    # Image should be (3, 224, 224) based on Config.IMAGE_SIZE
    expected_shape = (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        img_tensor.shape == expected_shape
    ), f"Image tensor shape mismatch. Expected {expected_shape}, got {img_tensor.shape}"

    # Verify target type
    assert isinstance(target.item(), int), "Target should be an integer"
    assert isinstance(img_id.item(), int), "Image ID should be an integer"

    print("    Dataset verification passed: Shapes and types are correct.")

    # 3. Model Verification
    print("\n[3] Verifying Model Architecture...")
    model = INatModel(pretrained=False)  # No need to download weights for shape check
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    # Verify output shape: (Batch_Size, Num_Classes)
    expected_output_shape = (2, Config.NUM_CLASSES)
    assert (
        outputs.shape == expected_output_shape
    ), f"Model output shape mismatch. Expected {expected_output_shape}, got {outputs.shape}"

    print(
        "    Model verification passed: Forward pass successful with correct output shape."
    )

    # 4. Training Loop Verification
    print("\n[4] Verifying Training Loop (Trainer)...")
    trainer = Trainer()

    # Get DataLoaders (this uses the overridden Config.DEBUG=True)
    train_loader, val_loader, test_loader = trainer.get_dataloaders(debug=True)

    # Verify DataLoader batch size
    batch_images, _, _ = next(iter(train_loader))
    assert (
        batch_images.shape[0] == Config.BATCH_SIZE
    ), f"DataLoader batch size mismatch. Expected {Config.BATCH_SIZE}, got {batch_images.shape[0]}"

    # Run training for 1 epoch
    print("    Running trainer.fit() for 1 epoch...")
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}"

    print("    Training verification passed: Model trained and checkpoint saved.")

    # 5. Inference Verification
    print("\n[5] Verifying Inference and Submission Generation...")

    # Run prediction using the trainer
    trainer.predict(test_loader)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    # Check submission content format
    submission_df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    expected_cols = ["id", "predicted"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Check row count (should match test_loader size, which is DEBUG_SAMPLE_SIZE)
    # Note: DataLoader might drop last if drop_last=True, but test loader usually doesn't.
    # In Config, test loader drop_last is default (False).
    assert (
        len(submission_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(submission_df)}"

    # Check prediction format (string of space-separated integers)
    first_pred = submission_df.iloc[0]["predicted"]
    assert isinstance(first_pred, str), "Prediction should be a string"
    pred_ids = first_pred.split()
    assert len(pred_ids) == 5, f"Expected 5 top predictions, got {len(pred_ids)}"

    print(
        "    Inference verification passed: Submission file generated with correct format."
    )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
