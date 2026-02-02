import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Import library modules
from library.config import Config
from library.data import get_dataloaders
from library.model import CassavaClassifier
from library.engine import train_one_epoch, evaluate, predict
from library.utils import seed_everything, compute_class_weights


def main():
    # --- 1. Setup & Configuration ---
    # Override Config for optimized execution
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32

    Config.setup_directories()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --- 2. Data Loading ---
    # Load full datasets initially
    # We use load_cached_data=True as requested
    # Cite solution_lesson_node_00005: Utilizing full training dataset
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --- 3. Model Initialization ---
    model = CassavaClassifier(num_classes=Config.NUM_CLASSES).to(device)

    # --- 4. Training Setup ---
    # Compute class weights based on the full training metadata (valid prior)
    class_weights = compute_class_weights(Config.TRAIN_METADATA, load_cached_data=True)
    # Cite solution_lesson_node_00008: Label Smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Cite solution_lesson_node_00008: CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # --- 5. Training Loop ---
    best_val_acc = 0.0

    for epoch in range(Config.EPOCHS):
        # Freeze backbone for the first epoch to stabilize the head
        if epoch == 0:
            model.freeze_backbone()
        elif epoch == 1:
            model.unfreeze_backbone()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # --- 6. Final Validation Evaluation ---
    # Load best model weights for final inference
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()

    # Generate predictions on the full validation set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    # Compute and Print Metric
    final_metric = accuracy_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --- 7. Failure Analysis ---
    # Analyze correlation between prediction error and image file size
    val_df = val_loader.dataset.df.copy()
    val_df["pred"] = val_preds
    val_df["target"] = val_targets
    val_df["is_error"] = (val_df["pred"] != val_df["target"]).astype(int)

    # Compute file sizes for validation images
    file_sizes = []
    for path in val_df["full_path"]:
        try:
            file_sizes.append(os.path.getsize(path))
        except OSError:
            file_sizes.append(0)
    val_df["file_size"] = file_sizes

    # Compute Correlation
    if not val_df.empty:
        corr = val_df["is_error"].corr(val_df["file_size"])
        print(f"Correlation between Error and File Size: {corr}")

    # --- 8. Submission ---
    if final_metric > 0.8571428571428571:
        # Generate predictions on the test set
        test_preds = predict(model, test_loader, device)

        # Create submission DataFrame
        test_df = test_loader.dataset.df
        submission = pd.DataFrame(
            {"image_id": test_df["image_id"], "label": test_preds}
        )

        # Save Submission
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print("Validation metric did not meet the threshold. Submission skipped.")


if __name__ == "__main__":
    main()
