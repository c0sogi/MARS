import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import CFG
from library.utils import seed_everything
from library.transforms import get_transforms
from library.dataset import CassavaDataset
from library.model import CassavaConvNeXt
from library.engine import (
    train_one_epoch,
    validate,
    generate_submission,
    inference_fn,
    EarlyStopping,
)


def run():
    # 1. Setup and Configuration
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    print(f"Device: {device}")
    print(f"Model: {CFG.model_name}")
    print(f"Epochs: {CFG.epochs}")

    # 2. Data Loading
    # We load datasets with the appropriate transforms
    train_dataset = CassavaDataset(split="train", transform=get_transforms("train"))
    val_dataset = CassavaDataset(split="val", transform=get_transforms("val"))
    test_dataset = CassavaDataset(split="test", transform=get_transforms("test"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = CassavaConvNeXt(
        model_name=CFG.model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
        drop_path_rate=CFG.drop_path_rate,
    )
    model.to(device)

    # 4. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)

    # 5. Loss Function
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)

    # 6. Training Loop
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")
    early_stopping = EarlyStopping(patience=3, mode="max", save_path=best_model_path)

    print("Starting training...")
    for epoch in range(CFG.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device, scheduler
        )

        # Validate (Standard validation without TTA for speed during training)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{CFG.epochs} - "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Check Early Stopping
        early_stopping(val_acc, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 7. Final Evaluation & Failure Analysis
    print(f"\nLoading best model from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # Generate predictions on validation set using TTA (if enabled in CFG)
    # inference_fn returns just the predictions
    print("Running inference on validation set with TTA...")
    val_preds = inference_fn(model, val_loader, device)
    val_labels = val_dataset.labels

    # Calculate Final Validation Metric
    final_acc = (val_preds == val_labels).mean()
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis: Correlation between Error and File Size
    print("\nPerforming Failure Analysis...")

    # Calculate errors (0 for correct, 1 for incorrect)
    errors = (val_preds != val_labels).astype(int)

    # Get file sizes for validation images
    val_file_paths = [os.path.join(CFG.input_root, p) for p in val_dataset.file_paths]
    file_sizes = [os.path.getsize(p) for p in val_file_paths]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"error": errors, "file_size": file_sizes})

    # Calculate correlation
    correlation = analysis_df.corr()["error"]["file_size"]
    print(f"Correlation between model error and file size: {correlation:.4f}")

    # 8. Submission
    # Threshold defined in task
    THRESHOLD = 0.8891855808

    if final_acc > THRESHOLD:
        print(f"\nValidation metric {final_acc} exceeds threshold {THRESHOLD}.")
        print("Generating submission...")
        generate_submission(model, test_loader, device, output_dir="./submission")
    else:
        print(f"\nValidation metric {final_acc} does not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
