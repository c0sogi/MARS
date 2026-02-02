import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import ResNet18Classifier
from library.train import Trainer
from library.predict import Predictor

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of Artwork Classification Pipeline ===")

    # ------------------------------------------------------------------------
    # 1. Setup and Config Overrides for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 images
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Low worker count for small data
    Config.PRETRAINED = False  # Skip downloading weights for demo
    Config.IMG_SIZE = 128  # Smaller image size for faster processing
    Config.RESIZE_SIZE = 140

    # Ensure directories exist
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, SUBSET=50")

    # ------------------------------------------------------------------------
    # 2. Dataset and DataLoader Demonstration
    # ------------------------------------------------------------------------
    print("\n[2] Verifying DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Targets Shape: {targets.shape}")

    # Assertions
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        images.shape == expected_img_shape
    ), f"Expected images {expected_img_shape}, got {images.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Expected targets {expected_target_shape}, got {targets.shape}"
    assert targets.dtype == torch.float32, "Targets must be float32"
    print("DataLoader verification successful.")

    # ------------------------------------------------------------------------
    # 3. Model Demonstration
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = ResNet18Classifier(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)

    # Forward pass check
    with torch.no_grad():
        # Move images to device
        input_tensor = images.to(Config.DEVICE)
        output = model(input_tensor)

    print(f"Model Output Shape: {output.shape}")

    assert output.shape == expected_target_shape, "Model output shape mismatch"
    print("Model forward pass successful.")

    # ------------------------------------------------------------------------
    # 4. Training Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Trainer.fit)...")

    trainer = Trainer(device=torch.device(Config.DEVICE))

    # Run training
    # Note: Trainer.fit() handles the loop, validation, and saving best model
    trainer.fit(debug=Config.DEBUG)

    # Ensure a model file exists for the next step.
    # If the model didn't improve (F1=0), Trainer might not have saved 'resnet18_best.pth'.
    # For this demo, we force save if it doesn't exist to allow the pipeline to continue.
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            "Note: No best model saved during training (likely F1=0). Saving current state for demo."
        )
        torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint file is missing."
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------------
    # 5. Inference and Submission Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Running Inference and Generating Submission...")

    predictor = Predictor(device=torch.device(Config.DEVICE))

    # Load the model we just trained/saved
    predictor.load_checkpoint(Config.MODEL_SAVE_PATH)

    # Optimize threshold
    print("Optimizing threshold on validation set...")
    best_thresh = predictor.optimize_threshold(val_loader)
    print(f"Selected Threshold: {best_thresh}")

    # Generate submission
    predictor.generate_submission(test_loader, threshold=best_thresh)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV was not created."
    print(f"Submission generated at {Config.SUBMISSION_PATH}")

    # ------------------------------------------------------------------------
    # 6. Validate Submission Format
    # ------------------------------------------------------------------------
    print("\n[6] Validating Submission File Format...")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {sub_df.shape}")
    print(f"First 3 rows:\n{sub_df.head(3)}")

    # Check columns
    required_cols = {"id", "attribute_ids"}
    assert required_cols.issubset(
        sub_df.columns
    ), f"Missing columns. Found: {sub_df.columns}"

    # Check row count (should match debug subset size)
    assert (
        len(sub_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} rows, found {len(sub_df)}"

    # Check content format
    # attribute_ids should be string of space-separated ints or NaN/empty
    for idx, row in sub_df.iterrows():
        attr = row["attribute_ids"]
        if pd.notna(attr) and str(attr).strip() != "":
            try:
                # Try converting split parts to ints
                parts = [int(x) for x in str(attr).split()]
            except ValueError:
                raise AssertionError(
                    f"Row {idx}: attribute_ids contains non-integer format: '{attr}'"
                )

    print("Submission format validation successful.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
