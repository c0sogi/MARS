import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image

# Import from library
from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    EPOCHS_HEAD,
    LEARNING_RATE_HEAD,
    EPOCHS_FINETUNE,
    LEARNING_RATE_FINETUNE,
    PATIENCE,
    INPUT_DIR,
)
from library.utils import seed_everything, save_submission
from library.dataset import get_dataloaders
from library.model import get_model, freeze_backbone, unfreeze_layer4_and_head
from library.trainer import train_one_epoch, validate, predict


def get_validation_metadata_stats(val_loader):
    """
    Extracts metadata (width, height, area, aspect_ratio) for the validation set
    by reading the image files. Used for failure analysis.
    """
    print("Extracting validation metadata for failure analysis...")
    df = val_loader.dataset.df

    widths = []
    heights = []
    areas = []
    aspect_ratios = []

    # Iterate through the dataframe to get file paths
    for _, row in df.iterrows():
        # Path construction logic matches dataset.py
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            # Lazy load to get size efficiently
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                aspect_ratios.append(w / h)
        except Exception:
            # Fallback (should not be triggered given data analysis)
            widths.append(0)
            heights.append(0)
            areas.append(0)
            aspect_ratios.append(0)

    return pd.DataFrame(
        {
            "width": widths,
            "height": heights,
            "area": areas,
            "aspect_ratio": aspect_ratios,
        }
    )


def perform_failure_analysis(model, val_loader, device):
    """
    Calculates error per sample on the validation set and correlates
    it with input image features.
    """
    model.eval()
    all_probs = []
    all_targets = []

    # 1. Get Predictions and Targets for Validation Set
    # Note: We cannot use trainer.predict() here because it returns IDs,
    # but we need ground truth labels for error calculation.
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Log Loss per sample (Error Magnitude)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)

    # Select probability assigned to the true class
    true_class_probs = probs[np.arange(len(targets)), targets]

    # Error = Negative Log Likelihood
    errors = -np.log(true_class_probs)

    # 3. Get Metadata
    meta_df = get_validation_metadata_stats(val_loader)
    meta_df["error"] = errors

    # 4. Calculate Correlation
    print("\n=== Failure Analysis ===")
    print("Correlation between Error Magnitude and Input Features:")
    correlations = meta_df.corr()["error"].drop("error")
    print(correlations)

    return correlations


def main():
    # 1. Setup
    seed_everything()
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    print("Loading datasets...")
    # load_cached_data=True to use ./working cache if available
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=True
    )
    num_classes = len(classes)

    # 3. Model Initialization
    model = get_model(num_classes=num_classes, pretrained=True, device=DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0

    # 4. Training - Phase 1: Head Adaptation
    print("\n=== Phase 1: Head Adaptation ===")
    freeze_backbone(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_HEAD
    )

    for epoch in range(EPOCHS_HEAD):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        print(
            f"Epoch {epoch+1}/{EPOCHS_HEAD} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

    # 5. Training - Phase 2: Fine-Tuning
    print("\n=== Phase 2: Fine-Tuning ===")
    # Load best weights from Phase 1
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    unfreeze_layer4_and_head(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_FINETUNE
    )

    patience_counter = 0  # Reset patience for fine-tuning

    for epoch in range(EPOCHS_FINETUNE):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        print(
            f"Epoch {epoch+1}/{EPOCHS_FINETUNE} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"Saved New Best Model. Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Evaluation & Failure Analysis
    print("\n=== Final Evaluation ===")
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading best model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    final_val_loss = validate(model, val_loader, criterion, DEVICE)
    # Required output format
    print(f"Final Validation Metric: {final_val_loss}")

    perform_failure_analysis(model, val_loader, DEVICE)

    # 7. Submission
    print("\n=== Generating Submission ===")
    probs, ids = predict(model, test_loader, DEVICE)
    save_submission(probs, ids, classes, submission_path=SUBMISSION_PATH)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
