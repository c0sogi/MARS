import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_data_and_classes, DogDataset, get_transforms
from library.model import DogModel, ModelEMA
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def run_demo():
    print("Initializing Demo...")

    # 1. Override Configuration for Speed and Demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset for rapid execution
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead on tiny datasets
    Config.WORKING_DIR = "./working/demo_run"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(
        f"Configuration: Device={Config.DEVICE}, Debug={Config.DEBUG}, BatchSize={Config.BATCH_SIZE}"
    )

    # 2. Reproducibility
    seed_everything(Config.SEED)

    # 3. Data Loading
    print("Loading Data...")
    # debug=True forces loading a small sample and skips reading full parquet cache if it doesn't match
    train_df, test_df, class_to_idx, classes = get_data_and_classes(debug=Config.DEBUG)

    # Validation: Check data integrity
    assert len(train_df) > 0, "Training dataframe is empty"
    assert len(test_df) > 0, "Test dataframe is empty"
    assert (
        len(classes) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, got {len(classes)}"
    print(
        f"Data Loaded: Train={len(train_df)}, Test={len(test_df)}, Classes={len(classes)}"
    )

    # 4. Dataset & DataLoader Setup
    # Manually split the small train_df into train/val for this demo
    split_idx = len(train_df) // 2
    val_df = train_df.iloc[:split_idx].copy()
    train_subset_df = train_df.iloc[split_idx:].copy()

    train_dataset = DogDataset(
        train_subset_df,
        class_to_idx=class_to_idx,
        transform=get_transforms("train", Config.IMG_SIZE),
        mode="train",
    )
    val_dataset = DogDataset(
        val_df,
        class_to_idx=class_to_idx,
        transform=get_transforms("val", Config.IMG_SIZE),
        mode="val",
    )
    test_dataset = DogDataset(
        test_df,
        class_to_idx=None,
        transform=get_transforms("test", Config.IMG_SIZE),
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Validation: Check batch shapes
    images, targets = next(iter(train_loader))
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape: {targets.shape}"
    print("DataLoader shapes verified.")

    # 5. Model Initialization
    print("Initializing Model...")
    # Initialize model (downloads weights if not cached)
    model = DogModel(num_classes=len(classes), pretrained=True)
    model.to(Config.DEVICE)

    # Validation: Check model output shape
    with torch.no_grad():
        # Pass the batch fetched earlier
        dummy_out = model(images.to(Config.DEVICE))
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        len(classes),
    ), f"Model output shape mismatch: {dummy_out.shape}"

    # Initialize EMA (Exponential Moving Average)
    model_ema = ModelEMA(model, device=Config.DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # 6. Training Loop Demo
    print("Starting Training Demo...")

    # Train one epoch
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, Config.DEVICE, epoch=1, model_ema=model_ema
    )
    assert (
        "Loss" in train_metrics and "Accuracy" in train_metrics
    ), "Training metrics missing keys"

    # Validate one epoch (using EMA model)
    val_metrics = valid_one_epoch(model_ema.module, val_loader, Config.DEVICE, epoch=1)
    assert (
        "Loss" in val_metrics and "Accuracy" in val_metrics
    ), "Validation metrics missing keys"

    # 7. Inference Demo
    print("Starting Inference Demo...")
    preds, ids = inference_fn(model_ema.module, test_loader, Config.DEVICE)

    # Validation: Check predictions
    assert preds.shape == (
        len(test_df),
        len(classes),
    ), f"Prediction shape mismatch: {preds.shape}"
    assert len(ids) == len(test_df), "ID count mismatch"

    # 8. Submission Generation
    print("Generating Submission...")
    submission = pd.DataFrame(preds, columns=classes)
    submission.insert(0, "id", ids)

    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(output_path, index=False)

    assert os.path.exists(output_path), "Submission file not created"
    print(f"Submission saved to {output_path}")

    # Save model checkpoint demo
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_best_model.pth")
    torch.save(model_ema.module.state_dict(), checkpoint_path)
    assert os.path.exists(checkpoint_path), "Model checkpoint not saved"
    print(f"Model checkpoint saved to {checkpoint_path}")

    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
