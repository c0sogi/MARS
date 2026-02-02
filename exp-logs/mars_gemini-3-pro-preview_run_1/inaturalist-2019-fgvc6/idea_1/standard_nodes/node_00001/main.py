import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image

from library.config import Config
from library.utils import set_seed
from library.dataset import INatDataset, get_transforms
from library.model import get_mobilenet_model
from library.train import train_one_epoch, validate
from library.predict import generate_submission


def analyze_failures(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Computes the final metric and correlates errors with image metadata.
    """
    print("\nStarting Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []

    # 1. Inference on validation set
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 2. Compute Metric (Top-1 Error)
    # Accuracy is fraction of correct predictions
    accuracy = np.mean(all_preds == all_targets)
    error_rate = 1.0 - accuracy

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {error_rate:.10f}")

    # 3. Correlation Analysis
    # Create a binary error array (1 for error, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    print("Collecting metadata for correlation analysis...")
    file_sizes = []
    aspect_ratios = []

    # Construct full paths from the dataframe
    # val_df order matches val_loader (shuffle=False)
    full_paths = [os.path.join(Config.INPUT_ROOT, p) for p in val_df["file_path"]]

    for path in full_paths:
        try:
            # Get file size
            size = os.path.getsize(path)
            file_sizes.append(size)

            # Get aspect ratio (Width / Height)
            # Image.open is lazy, so this is relatively fast
            with Image.open(path) as img:
                w, h = img.size
                aspect_ratios.append(w / h if h > 0 else 0)
        except Exception as e:
            # Fallback for any missing/corrupt files
            file_sizes.append(0)
            aspect_ratios.append(0)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"error": errors, "file_size": file_sizes, "aspect_ratio": aspect_ratios}
    )

    # Compute correlation with the 'error' column
    correlations = analysis_df.corr()["error"].drop("error")

    print("\nCorrelation between Model Error and Input Features:")
    print(correlations)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_dataset = INatDataset(
        csv_path=Config.TRAIN_CSV, mode="train", transform=get_transforms(stage="train")
    )
    val_dataset = INatDataset(
        csv_path=Config.VAL_CSV, mode="val", transform=get_transforms(stage="val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing MobileNetV3-Large...")
    model = get_mobilenet_model(
        pretrained=True, num_classes=Config.NUM_CLASSES, device=device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Fast Baseline: Train for 2 epochs
    epochs = 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # 4. Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train
        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_acc = validate(val_loader, model, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")

    # 5. Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No checkpoint found, using current model weights.")

    # Load validation metadata for analysis
    val_df = pd.read_csv(Config.VAL_CSV)

    # Run Analysis
    analyze_failures(model, val_loader, val_df, device)

    # 6. Submission
    print("\nGenerating submission for test set...")
    generate_submission(checkpoint_path=best_model_path, device=device)


if __name__ == "__main__":
    main()
