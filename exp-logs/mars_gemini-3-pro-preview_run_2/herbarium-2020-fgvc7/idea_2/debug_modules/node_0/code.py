import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.model import PlantClassifier
from library.loss import FocalLoss
from library.engine import train_model, predict
from library.utils import set_seed, calculate_macro_f1

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("==== Starting Library Demonstration ====")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to ensure the demo runs quickly.
    print("\n[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 images for the demo
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size for the demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, SUBSET_SIZE=100, EPOCHS=1")

    # 2. Data Loading Demonstration
    print("\n[2] Demonstrating Data Loading...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        debug=Config.DEBUG, batch_size=Config.BATCH_SIZE
    )

    print(f"Number of classes detected: {len(classes)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"Train Batch - Image Shape: {images.shape}, Label Shape: {labels.shape}")

        # Assertions
        assert (
            images.shape[0] == Config.BATCH_SIZE
        ), "Batch size mismatch in train_loader"
        assert images.shape[1] == 3, "Image channels should be 3"
        assert (
            images.shape[2] == Config.IMG_SIZE and images.shape[3] == Config.IMG_SIZE
        ), "Image dimensions mismatch"
        assert isinstance(labels, torch.Tensor), "Labels should be a torch.Tensor"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Test Loader (returns images and image_ids)
    try:
        test_images, test_ids = next(iter(test_loader))
        print(
            f"Test Batch - Image Shape: {test_images.shape}, ID Shape: {test_ids.shape}"
        )
        assert (
            test_images.shape[0] == Config.BATCH_SIZE
        ), "Batch size mismatch in test_loader"
    except StopIteration:
        raise AssertionError("Test loader is empty!")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Demonstrating Model Initialization and Forward Pass...")
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Instantiate model
    # Note: We use the actual number of classes found in the subset/dataset
    # In debug mode with a small subset, the number of unique classes might be smaller
    # than Config.NUM_CLASSES, but the code in get_dataloaders maps them correctly.
    # However, the model expects a fixed number of output neurons.
    # For this demo, we use len(classes) to match the loader's mapping.
    model = PlantClassifier(num_classes=len(classes), pretrained=True)
    model.to(device)

    # Forward pass with the batch fetched earlier
    images = images.to(device)
    outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        Config.BATCH_SIZE,
        len(classes),
    ), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"

    # 4. Loss Function Demonstration
    print("\n[4] Demonstrating Focal Loss...")
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    labels = labels.to(device)

    loss = criterion(outputs, labels)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 5. Training Loop Demonstration
    print("\n[5] Demonstrating Training Loop (1 Epoch)...")
    # We use the train_model function from library.engine
    # It handles the optimizer, loop, validation, and saving best model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created"
    print("Training complete. Checkpoint verified.")

    # 6. Inference Demonstration
    print("\n[6] Demonstrating Inference and Submission Generation...")
    # Predict using the trained model
    predict(
        model=trained_model,
        test_loader=test_loader,
        classes_list=classes,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Rows: {len(submission_df)}")
    print(submission_df.head())

    assert (
        "Id" in submission_df.columns and "Predicted" in submission_df.columns
    ), "Submission file missing required columns"
    assert len(submission_df) > 0, "Submission file is empty"

    # 7. Utility Function Demonstration
    print("\n[7] Demonstrating Utility Functions (Macro F1)...")
    # Create dummy ground truth and predictions
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])

    f1 = calculate_macro_f1(y_true, y_pred)
    print(f"Calculated Macro F1: {f1:.4f}")

    # Manual calculation check:
    # Class 0: TP=2, FP=0, FN=0 -> F1=1.0
    # Class 1: TP=1, FP=1, FN=1 -> Precision=0.5, Recall=0.5 -> F1=0.5
    # Class 2: TP=1, FP=1, FN=1 -> Precision=0.5, Recall=0.5 -> F1=0.5
    # Macro F1 = (1.0 + 0.5 + 0.5) / 3 = 0.6667
    assert 0.66 < f1 < 0.67, "Macro F1 calculation logic seems incorrect"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
