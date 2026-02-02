import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import from provided library files
from library.utils import seed_everything, get_device
from library.dataset import (
    load_processed_metadata,
    get_transforms,
    DogBreedDataset,
    INPUT_DIR,
)
from library.model import ResNet50Baseline
from library.engine import (
    train_one_epoch,
    validate,
    train_model,
    predict,
    save_submission,
)

# Constants
BATCH_SIZE = 128
WARMUP_EPOCHS = 1
FINETUNE_EPOCHS = 10
WARMUP_LR = 1e-3
FINETUNE_LR = 1e-4
PATIENCE = 3
SEED = 42
MODEL_SAVE_PATH = "./working/best_model.pth"
SUBMISSION_PATH = "./submission/submission.csv"


def analyze_failures(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample loss and correlates it with input features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")  # Return loss per sample

    all_losses = []

    # 1. Calculate per-sample loss
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            losses = criterion(outputs, labels)
            all_losses.extend(losses.cpu().numpy())

    val_df["error_magnitude"] = all_losses

    # 2. Extract input features (File Size, Width, Height, Aspect Ratio)
    # We process this on the fly for the validation set
    file_sizes = []
    widths = []
    heights = []
    aspect_ratios = []

    print("Extracting metadata features for correlation analysis...")
    for idx, row in val_df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # File size
        try:
            f_size = os.path.getsize(full_path)
        except OSError:
            f_size = 0

        # Image dims
        img = cv2.imread(full_path)
        if img is not None:
            h, w = img.shape[:2]
            ar = w / h if h > 0 else 0
        else:
            h, w, ar = 0, 0, 0

        file_sizes.append(f_size)
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(ar)

    val_df["file_size"] = file_sizes
    val_df["width"] = widths
    val_df["height"] = heights
    val_df["aspect_ratio"] = aspect_ratios

    # 3. Calculate Correlations
    features = ["file_size", "width", "height", "aspect_ratio"]
    print("\nCorrelation between Error Magnitude and Input Features:")
    for feat in features:
        # Drop NaNs if any image load failed
        valid_data = val_df[[feat, "error_magnitude"]].dropna()
        if len(valid_data) > 1:
            corr, _ = pearsonr(valid_data[feat], valid_data["error_magnitude"])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: Insufficient data")


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    train_df, val_df, test_df, classes = load_processed_metadata(load_cached_data=True)
    num_classes = len(classes)
    print(f"Loaded metadata. Classes: {num_classes}")

    train_transform = get_transforms("train")
    val_transform = get_transforms("val")  # Same for test

    train_dataset = DogBreedDataset(train_df, transform=train_transform, mode="train")
    val_dataset = DogBreedDataset(val_df, transform=val_transform, mode="val")
    test_dataset = DogBreedDataset(test_df, transform=val_transform, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = ResNet50Baseline(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Cite solution_lesson_node_00002: Decouple training objectives from evaluation metrics.
    # Use Label Smoothing for training to improve generalization.
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Use standard CrossEntropyLoss for validation to measure true Log Loss.
    val_criterion = nn.CrossEntropyLoss()

    # 4. Phase 1: Warm-up (Freeze Backbone)
    print("\n=== Phase 1: Warm-up ===")
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Ensure head is trainable
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    optimizer_warmup = optim.AdamW(model.backbone.fc.parameters(), lr=WARMUP_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(WARMUP_EPOCHS):
        loss, acc = train_one_epoch(
            model, train_loader, train_criterion, optimizer_warmup, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)
        print(
            f"Warmup Epoch {epoch+1}/{WARMUP_EPOCHS} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # 5. Phase 2: Fine-tuning (Unfreeze All)
    print("\n=== Phase 2: Fine-tuning ===")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_finetune = optim.AdamW(model.parameters(), lr=FINETUNE_LR)

    # We use the engine's train_model for full loop with early stopping
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_criterion=train_criterion,
        val_criterion=val_criterion,
        optimizer=optimizer_finetune,
        num_epochs=FINETUNE_EPOCHS,
        patience=PATIENCE,
        device=device,
        save_path=MODEL_SAVE_PATH,
    )

    # 6. Final Evaluation
    print("\n=== Final Evaluation ===")
    # Load best model (train_model already does this, but ensuring consistency)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    # Calculate Final Metric (Log Loss)
    # Validate function returns Average Cross Entropy Loss, which IS Log Loss for multi-class
    final_loss, final_acc = validate(model, val_loader, val_criterion, device)
    print(f"Final Validation Metric: {final_loss}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, val_df, device)

    # 8. Submission
    print("\n=== Generating Submission ===")
    if final_loss < 0.5640784429467243:
        print(
            f"Validation score {final_loss} improved over baseline 0.564078. Generating submission."
        )
        ids, probs = predict(model, test_loader, device)
        save_submission(ids, probs, classes, output_path=SUBMISSION_PATH)
    else:
        print(
            f"Validation score {final_loss} did not improve over baseline 0.564078. Skipping submission."
        )

    print("Run complete.")


if __name__ == "__main__":
    main()
