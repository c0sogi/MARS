import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam

# Import library components
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders, get_test_dataloader
from library.networks import get_model
from library.engine import train_one_epoch, evaluate, train_fold, predict


def run_demo():
    print("--- 1. Setup and Configuration ---")

    # Override Config for rapid demonstration
    # We modify the class attributes directly to affect all modules using Config
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 64  # Small number for fast execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run
    Config.MODEL_ARCHITECTURES = ["resnet18"]  # Focus on one architecture

    # Redirect working directory to a demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Configuration: DEBUG={Config.DEBUG}, DEVICE={Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print("\n--- 2. Dataset and DataLoader Demonstration ---")

    # Generate dataloaders for Fold 0
    # load_cached_data=False forces the folds to be regenerated in our new WORKING_DIR
    train_loader, val_loader = get_dataloaders(
        fold_id=0,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
        debug=Config.DEBUG,
    )

    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Val Loader: {len(val_loader)} batches")

    # Verify Train Batch Structure
    images, labels = next(iter(train_loader))
    print(f"Sample Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions to ensure data pipeline is correct
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.CROP_SIZE, Config.CROP_SIZE)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"

    print("\n--- 3. Model Instantiation Demonstration ---")

    model_name = "resnet18"
    # Use pretrained=False to ensure speed and avoid internet dependency for demo
    model = get_model(model_name, pretrained=False)
    model = model.to(Config.DEVICE)

    print(f"Model '{model_name}' instantiated successfully.")

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(Config.DEVICE)
        output = model(dummy_input)
        # Output shape should be [Batch, 1] (Binary Classification Logits)
        print(f"Forward pass output shape: {output.shape}")
        assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    print("\n--- 4. Training Engine Demonstration ---")

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Demonstrate train_one_epoch
    print("Executing train_one_epoch...")
    avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
    print(f"Average Train Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss returned NaN"

    # Demonstrate evaluate (Validation)
    print("Executing evaluate (Validation)...")
    val_auc = evaluate(model, val_loader, Config.DEVICE, use_tta=False)
    print(f"Validation AUC: {val_auc:.4f}")
    assert 0.0 <= val_auc <= 1.0, "AUC score must be between 0 and 1"

    print("\n--- 5. Full Fold Training Loop Demonstration ---")

    save_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

    # Re-initialize model and optimizer for a clean training run
    model = get_model(model_name, pretrained=False).to(Config.DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Run the high-level train_fold function
    # This encapsulates the loop, validation, TTA, and model saving
    print(f"Starting training for {Config.EPOCHS} epoch(s)...")
    trained_model = train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=save_path,
    )

    # Verify model checkpoint was created
    assert os.path.exists(save_path), f"Model checkpoint not found at {save_path}"
    print("Training complete. Model saved.")

    print("\n--- 6. Inference and Submission Demonstration ---")

    # Get Test Loader
    test_loader = get_test_dataloader(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )
    print(f"Test Loader: {len(test_loader)} batches")

    # Run Prediction
    print("Generating predictions...")
    preds = predict(trained_model, test_loader, Config.DEVICE, use_tta=True)

    print(f"Predictions shape: {preds.shape}")

    # Verify predictions are valid probabilities
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions must be probabilities [0, 1]"

    # Create Submission DataFrame
    # In debug mode, we must match the subset of IDs loaded by the test loader
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SAMPLES)

    # Ensure lengths match
    assert len(df_test) == len(
        preds
    ), f"Mismatch: {len(df_test)} IDs vs {len(preds)} predictions"

    submission = pd.DataFrame({"id": df_test["id"], "label": preds})

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("First 5 rows:")
    print(submission.head())

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
