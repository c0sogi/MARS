import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import AppleDataset, get_transforms
from library.loss import WeightedCrossEntropyLoss, get_class_weights
from library.model import AppleResNet34
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def run_demo():
    print("==== Starting Apple Disease Detection Pipeline Demo ====")

    # 1. Setup & Configuration Overrides for Speed
    print("\n[1] Setting up Configuration...")
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small subset
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.IMAGE_SIZE = 128  # Smaller size for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Preparation
    print("\n[2] Preparing Data...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    print(f"Loaded Train Metadata: {df_train.shape}")
    print(f"Loaded Val Metadata: {df_val.shape}")

    # Calculate Class Weights
    print("Calculating class weights...")
    class_weights = get_class_weights(df_train, load_cached_data=False, device=device)
    print(f"Class Weights: {class_weights}")
    assert class_weights.shape[0] == Config.NUM_CLASSES, "Class weights shape mismatch"

    # Instantiate Datasets
    print("Instantiating Datasets...")
    train_dataset = AppleDataset(
        df_train, transforms=get_transforms("train"), debug=Config.DEBUG
    )
    val_dataset = AppleDataset(
        df_val, transforms=get_transforms("valid"), debug=Config.DEBUG
    )

    # Verification: Check Dataset Output
    sample_img, sample_label = train_dataset[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Label: {sample_label}")

    assert sample_img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a tensor"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    print("DataLoaders created successfully.")

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = AppleResNet34(pretrained=True)
    model.to(device)

    # Verification: Dummy Forward Pass
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape ({Config.BATCH_SIZE}, {Config.NUM_CLASSES}), got {dummy_output.shape}"

    # 4. Loss, Optimizer, Scheduler
    print("\n[4] Setting up Training Components...")
    criterion = WeightedCrossEntropyLoss(weights=class_weights, device=device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
    )

    # Verification: Loss Calculation
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (Config.BATCH_SIZE,)).to(
        device
    )
    loss_val = criterion(dummy_output, dummy_targets)
    print(f"Initial Dummy Loss: {loss_val.item():.4f}")
    assert not torch.isnan(loss_val), "Loss is NaN"

    # 5. Training Loop Execution
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # Train
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, scheduler
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)
    print(f"Epoch 1 Val Loss: {val_loss:.4f}")
    print(f"Epoch 1 Val AUC:  {val_auc:.4f}")

    assert train_loss > 0, "Train loss should be positive"
    assert 0 <= val_auc <= 1, "AUC should be between 0 and 1"

    # 6. Inference Demonstration
    print("\n[6] Running Inference with TTA...")

    # Using validation set as a proxy for test set for demonstration
    test_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    predictions = inference_fn(model, test_loader, device, use_tta=True)

    print(f"Predictions Shape: {predictions.shape}")
    assert predictions.shape == (
        len(val_dataset),
        Config.NUM_CLASSES,
    ), "Prediction shape mismatch"

    # Verify probabilities sum to ~1
    row_sums = predictions.sum(axis=1)
    print(f"Mean Row Sum (should be ~1.0): {row_sums.mean():.4f}")
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # 7. Generate Submission Format
    print("\n[7] Generating Sample Submission...")

    # Create a dummy dataframe matching the predictions
    submission_df = pd.DataFrame(predictions, columns=Config.CLASS_LABELS)
    # Add image_id from the dataset used
    submission_df.insert(0, "image_id", val_dataset.df["image_id"])

    print("Submission Head:")
    print(submission_df.head())

    # Save to working directory
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Saved demo submission to {output_path}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
