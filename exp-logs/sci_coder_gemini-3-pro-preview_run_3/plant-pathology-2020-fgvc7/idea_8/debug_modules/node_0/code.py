import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataset, AppleDataset
from library.models import AppleEfficientNet, AppleMaxViT
from library.engine import train_one_epoch, validate, inference_with_tta
from library.utils import seed_everything, get_class_weights, calculate_metric


def run_demo():
    print("Starting Apple Disease Detection Library Demo...")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Setting up configuration for fast demonstration...")
    seed_everything(42)

    # Override Config for speed and demonstration purposes
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    # Use smaller image size for faster processing in demo
    Config.IMG_SIZE_EFFNET = 224

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Demonstrate Utility Functions
    print("\n[2] Testing Utility Functions...")

    # Test Class Weights
    print("Calculating class weights...")
    try:
        # load_cached_data=False forces re-computation to test logic
        class_weights = get_class_weights(load_cached_data=False)
        print(f"Class weights shape: {class_weights.shape}")
        print(f"Class weights: {class_weights.cpu().numpy()}")
        assert len(class_weights) == Config.NUM_CLASSES, "Class weights length mismatch"
    except Exception as e:
        print(
            f"Note: Could not calculate class weights (possibly due to env context): {e}"
        )
        # Fallback for demo continuity
        class_weights = torch.ones(Config.NUM_CLASSES).to(device)

    # Test Metric Calculation
    print("Testing metric calculation...")
    # Create dummy ground truth and predictions
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    y_pred = np.array([[0.9, 0.1, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]])
    score = calculate_metric(y_true, y_pred)
    print(f"Calculated ROC AUC Score: {score}")
    assert 0.0 <= score <= 1.0, "Score out of range"

    # 3. Demonstrate Dataset Loading
    print("\n[3] Testing Dataset Loading...")

    # Train Dataset
    print("Loading Train Dataset (Debug mode)...")
    # We use img_size=224 for the demo to be faster
    train_dataset = get_dataset("train", img_size=224, debug=True)
    print(f"Train dataset size: {len(train_dataset)}")

    if len(train_dataset) > 0:
        sample_img, sample_label = train_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample label: {sample_label}")

        assert sample_img.shape == (3, 224, 224), "Incorrect image shape"
        assert sample_label.shape == (4,), "Incorrect label shape"
        assert isinstance(sample_img, torch.Tensor), "Image is not a tensor"

    # Test Dataset (No labels)
    print("Loading Test Dataset (Debug mode)...")
    test_dataset = get_dataset("test", img_size=224, debug=True)
    if len(test_dataset) > 0:
        sample_test_img = test_dataset[0]
        assert isinstance(sample_test_img, torch.Tensor), "Test image is not a tensor"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging/demo
        drop_last=False,
    )

    # 4. Demonstrate Model Initialization
    print("\n[4] Testing Model Initialization...")

    # We use pretrained=False to avoid downloading weights during this quick demo
    # In a real training run, pretrained=True would be used.
    model = AppleEfficientNet(model_name=Config.MODEL_EFFNET, pretrained=False)
    model.to(device)

    # Test Forward Pass with dummy data
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"

    # 5. Demonstrate Training Loop (Engine)
    print("\n[5] Testing Training Loop...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Run for one epoch (on the small debug subset)
    loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_loader,
        device=device,
        epoch=0,
        class_weights=class_weights,
    )

    print(f"Epoch 0 Training Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # 6. Demonstrate Validation Loop (Engine)
    print("\n[6] Testing Validation Loop...")

    # Use val dataset
    val_dataset = get_dataset("val", img_size=224, debug=True)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    val_loss, val_score, val_preds, val_targets = validate(
        model=model, dataloader=val_loader, device=device, class_weights=class_weights
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation ROC AUC: {val_score:.4f}")
    print(f"Predictions shape: {val_preds.shape}")

    assert val_preds.shape[1] == Config.NUM_CLASSES, "Prediction classes mismatch"
    assert len(val_preds) == len(val_dataset), "Prediction count mismatch"

    # 7. Demonstrate Inference with TTA
    print("\n[7] Testing Inference with TTA...")

    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_preds = inference_with_tta(model=model, dataloader=test_loader, device=device)

    print(f"Test Predictions shape: {test_preds.shape}")
    assert test_preds.shape[0] == len(test_dataset), "Test prediction count mismatch"
    assert test_preds.shape[1] == Config.NUM_CLASSES, "Test prediction classes mismatch"

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demo()
