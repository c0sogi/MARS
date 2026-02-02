import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data_loader import make_folds, get_dataloaders, get_test_dataloader
from library.modeling import get_model
from library.trainer import train_model


def main():
    print("Starting Library Usage Demonstration...")

    # --- 1. Configuration Overrides for Speed ---
    print("\n[1] Configuring for Fast Execution")
    # Enable debug mode to use a tiny subset of data (32 train, 16 val samples)
    Config.DEBUG = True
    # Run only 1 epoch
    Config.NUM_EPOCHS = 1
    # Small batch size
    Config.BATCH_SIZE = 4
    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.setup()

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # --- 2. Data Loading Demonstration ---
    print("\n[2] Verifying Data Pipeline")

    # Generate folds (force regeneration to test logic)
    # This reads metadata/train.csv and metadata/val.csv, splits them, and saves to parquet
    df_folds = make_folds(load_cached_data=False)
    assert isinstance(df_folds, pd.DataFrame)
    assert "fold" in df_folds.columns
    print("Folds generated successfully.")

    # Get DataLoaders for Fold 0
    train_loader, val_loader = get_dataloaders(fold_idx=0, load_cached_data=True)

    # Verify Train Loader Batch
    tr_images, tr_labels = next(iter(train_loader))
    print(f"Train Batch - Images: {tr_images.shape}, Labels: {tr_labels.shape}")

    # Assertions for shapes: (Batch, 3, 224, 448) and (Batch, 19)
    assert tr_images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        448,
    ), "Train image shape mismatch"
    assert tr_labels.shape == (Config.BATCH_SIZE, 19), "Train label shape mismatch"
    assert tr_images.dtype == torch.float32
    assert tr_labels.dtype == torch.float32

    # Verify Val Loader Batch
    val_images, val_labels = next(iter(val_loader))
    print(f"Val Batch   - Images: {val_images.shape}, Labels: {val_labels.shape}")
    assert val_images.shape == (Config.BATCH_SIZE, 3, 224, 448)

    # --- 3. Model Instantiation Demonstration ---
    print("\n[3] Verifying Model Architecture")

    # Initialize model (using ResNet18, no pretrained weights for speed)
    model = get_model("resnet18", pretrained=False)
    model.to(Config.DEVICE)

    # Verify Forward Pass
    dummy_input = tr_images.to(Config.DEVICE)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 19), "Model output logits shape mismatch"

    # --- 4. Metric Utility Demonstration ---
    print("\n[4] Verifying Metric Calculation (ROC AUC)")

    # Create synthetic ground truth and predictions
    # 3 samples, 3 classes
    y_true_syn = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    # Predictions (probabilities)
    y_pred_syn = np.array([[0.9, 0.1, 0.8], [0.2, 0.8, 0.1], [0.7, 0.9, 0.2]])

    auc_score = calculate_roc_auc(y_true_syn, y_pred_syn)
    print(f"Calculated AUC on synthetic data: {auc_score:.4f}")
    assert 0.0 <= auc_score <= 1.0, "AUC score out of valid range"

    # --- 5. Training Loop Demonstration ---
    print("\n[5] Verifying Training Loop")

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training for the configured number of epochs (1)
    # This tests: Mixup, Loss (BCEWithLogits + PosWeights), Backprop, Validation
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    print("Training finished.")
    print("History keys:", history.keys())

    assert "train_loss" in history
    assert "val_loss" in history
    assert "val_auc" in history
    assert len(history["train_loss"]) == Config.NUM_EPOCHS
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val AUC: {history['val_auc'][-1]:.4f}")

    # --- 6. Test/Inference Loader Demonstration ---
    print("\n[6] Verifying Test Inference Loader")

    test_loader = get_test_dataloader()
    test_images, test_ids = next(iter(test_loader))

    print(f"Test Batch  - Images: {test_images.shape}, IDs: {test_ids.shape}")
    assert test_images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        448,
    ), "Test image shape mismatch"
    # IDs should be integers (rec_id)
    assert test_ids.dtype in [torch.int32, torch.int64], "Test IDs should be integers"

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
