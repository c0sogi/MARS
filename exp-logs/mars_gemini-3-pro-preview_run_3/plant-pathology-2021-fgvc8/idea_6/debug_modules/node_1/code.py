import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, ModelEMA
from library.dataset import AppleDataset, get_transforms, MixupCollate
from library.model import AppleDiseaseModel
from library.engine import train_one_epoch, validate, predict_tta

if __name__ == "__main__":
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print("Initializing demonstration...")
    seed_everything(Config.SEED)

    # Modify Config for a fast, self-contained demo run
    Config.PRETRAINED = False  # Avoid downloading weights for demo speed/stability
    Config.IMG_SIZE = 224  # Smaller image size for faster processing
    Config.BATCH_SIZE = 8  # Small batch size for the demo subset
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.NUM_WORKERS = 2  # Reduce worker overhead
    Config.DEBUG = True  # Logic flag (though we manually sample below)

    # Ensure working directory exists for any outputs
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Preparation (Sampling)
    # ==========================================
    print("Preparing data subsets...")

    # Load full training metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_full = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Sample a small subset for Train and Val to ensure quick execution
    # We take 20 samples for training and 10 for validation
    df_train_demo = df_full.head(20).reset_index(drop=True)
    df_val_demo = df_full.iloc[20:30].reset_index(drop=True)

    print(f"Demo Train Size: {len(df_train_demo)}")
    print(f"Demo Val Size: {len(df_val_demo)}")

    # ==========================================
    # 3. Dataset & DataLoader Instantiation
    # ==========================================
    print("Initializing Datasets and Loaders...")

    # Get transforms
    train_transforms = get_transforms("train", Config)
    val_transforms = get_transforms("val", Config)

    # Create Datasets
    # Note: Config.INPUT_DIR is "./input". Metadata file_paths are like "train_images/xyz.jpg"
    train_dataset = AppleDataset(
        df_train_demo, train_transforms, Config.INPUT_DIR, Config.LABEL2ID
    )
    val_dataset = AppleDataset(
        df_val_demo, val_transforms, Config.INPUT_DIR, Config.LABEL2ID
    )

    # Create Mixup Collate function
    mixup_fn = MixupCollate(Config)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=mixup_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
    )

    # Verify DataLoader output shapes
    print("Verifying DataLoader shapes...")
    sample_imgs, sample_targets = next(iter(train_loader))

    # Expected Image Shape: (Batch, 3, H, W)
    assert sample_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image batch shape: {sample_imgs.shape}"

    # Expected Target Shape: (Batch, Num_Classes)
    assert sample_targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect target batch shape: {sample_targets.shape}"

    print("Data loading verification successful.")

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("Initializing Model...")

    device = Config.DEVICE
    model = AppleDiseaseModel(Config)
    model.to(device)

    # Initialize EMA
    model_ema = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # Verify Model Forward Pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {dummy_output.shape}"

    print("Model verification successful.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print(f"Starting training demonstration for {Config.EPOCHS} epochs...")

    # Optimizer & Scheduler Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch, model_ema)

        # Validate
        # Using model_ema.module for validation as per best practices
        val_loss, val_f1 = validate(
            model_ema.module, val_loader, device, threshold=Config.THRESHOLD
        )

        scheduler.step()

        print(
            f"Epoch {epoch} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        # Assertions to ensure training logic is producing valid numbers
        assert not np.isnan(loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert 0.0 <= val_f1 <= 1.0, f"F1 score out of range: {val_f1}"

    print("Training loop completed successfully.")

    # ==========================================
    # 6. Inference / TTA Demonstration
    # ==========================================
    print("Demonstrating Inference with TTA...")

    # For demonstration, we use the validation loader as the test loader
    # In a real scenario, this would use the test metadata
    test_loader_demo = val_loader

    # Run TTA Prediction
    df_submission = predict_tta(
        model_ema.module, test_loader_demo, device, threshold=Config.THRESHOLD
    )

    # Verify Submission Format
    print("Verifying submission format...")
    assert isinstance(
        df_submission, pd.DataFrame
    ), "Prediction output is not a DataFrame"
    assert "image" in df_submission.columns, "Submission missing 'image' column"
    assert "labels" in df_submission.columns, "Submission missing 'labels' column"
    assert len(df_submission) == len(
        df_val_demo
    ), f"Submission row count mismatch. Expected {len(df_val_demo)}, got {len(df_submission)}"

    # Check content of labels (should be string)
    assert isinstance(
        df_submission.iloc[0]["labels"], str
    ), "Labels column should contain strings"

    print("Inference verification successful.")
    print("\nAll demonstrations completed successfully.")
