import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_device, calculate_f1_score
from library.dataset import get_loaders
from library.model import AppleClassifier
from library.engine import train_model, validate, predict_with_tta


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates them with image features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    all_probs = []
    all_targets = []

    # 1. Get Predictions and Targets
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Error Magnitude (Mean Absolute Error per sample)
    # Shape: (N_samples,)
    errors = np.mean(np.abs(all_targets - all_probs), axis=1)

    # 3. Extract Image Meta-Features
    # We access the dataframe directly to get file paths
    val_df = val_loader.dataset.df
    widths = []
    heights = []
    brightness_vals = []

    print(f"Extracting features from {len(val_df)} validation images...")

    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(full_path)

        if img is not None:
            h, w, c = img.shape
            # Calculate brightness (simple average of channels)
            b_val = np.mean(img) / 255.0

            widths.append(w)
            heights.append(h)
            brightness_vals.append(b_val)
        else:
            # Fallback for missing images (should not happen based on checks)
            widths.append(0)
            heights.append(0)
            brightness_vals.append(0)

    # 4. Calculate Correlations
    features = {"Width": widths, "Height": heights, "Brightness": brightness_vals}

    print("\nCorrelation between Error Magnitude and Input Features:")
    print(f"{'Feature':<15} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 45)

    for name, values in features.items():
        if len(set(values)) > 1:
            corr, p_val = pearsonr(errors, values)
            print(f"{name:<15} | {corr:<12.4f} | {p_val:<12.4f}")
        else:
            print(f"{name:<15} | {'N/A':<12} | {'N/A':<12}")
    print("-" * 45)


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # Adjust Config for Fast Baseline if necessary
    # We reduce epochs slightly to ensure runtime safety while maintaining performance
    Config.EPOCHS = 15

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = AppleClassifier(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)

    # 4. Training Setup
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs=Config.EPOCHS,
        scheduler=scheduler,
        patience=5,
    )

    # 6. Final Validation & Analysis
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    val_loss, val_f1 = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_f1}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.8900961996445443

    if val_f1 > THRESHOLD:
        print(
            f"Validation F1 ({val_f1:.4f}) exceeds threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        predict_with_tta(model, test_loader, device, output_path=Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation F1 ({val_f1:.4f}) did not exceed threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
