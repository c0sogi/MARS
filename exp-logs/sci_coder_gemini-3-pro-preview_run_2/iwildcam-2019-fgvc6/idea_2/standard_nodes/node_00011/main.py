import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_transforms, compute_class_weights
from library.dataset import load_dataset
from library.model import HybridResNet
from library.trainer import (
    train_stage1_warmup,
    train_stage2_finetune,
    validate,
    generate_submission,
)


def run_failure_analysis(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    # Collect predictions
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate Error Magnitude
    # Error Magnitude = 1 - Probability of the true class
    # This quantifies how "wrong" or "uncertain" the model was about the correct label
    rows = np.arange(len(all_labels))
    true_class_probs = all_probs[rows, all_labels]
    error_magnitude = 1.0 - true_class_probs

    # Create analysis dataframe
    # We must ensure alignment. DataLoader with shuffle=False preserves order.
    # val_df should match the loader's dataset.
    analysis_df = val_df.copy().reset_index(drop=True)

    # Safety check for length alignment
    if len(analysis_df) != len(error_magnitude):
        print(
            f"Warning: Metadata length ({len(analysis_df)}) matches predictions ({len(error_magnitude)})? "
            f"{len(analysis_df) == len(error_magnitude)}"
        )
        # If mismatch (unlikely with correct usage), truncate to min length
        min_len = min(len(analysis_df), len(error_magnitude))
        analysis_df = analysis_df.iloc[:min_len]
        error_magnitude = error_magnitude[:min_len]

    analysis_df["error_magnitude"] = error_magnitude

    # Select numerical features for correlation
    # Based on EDA: frame_num, seq_num_frames, location, height
    features = ["frame_num", "seq_num_frames", "location", "height"]
    available_features = [f for f in features if f in analysis_df.columns]

    print("Correlation between Error Magnitude and Input Features:")
    correlations = {}
    for feature in available_features:
        # Ensure numeric
        if pd.api.types.is_numeric_dtype(analysis_df[feature]):
            corr = analysis_df["error_magnitude"].corr(analysis_df[feature])
            correlations[feature] = corr
            print(f"  {feature}: {corr:.4f}")
        else:
            print(f"  {feature}: Skipped (Non-numeric)")

    return correlations


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Load Data
    # Fast baseline: Limit training data to 20,000 samples to ensure quick execution
    # Full validation set required for metric calculation
    print("Loading datasets...")
    train_dataset = load_dataset(
        "train", transform=get_transforms("train"), debug_size=20000
    )
    val_dataset = load_dataset("val", transform=get_transforms("val"), debug_size=None)
    test_dataset = load_dataset(
        "test", transform=get_transforms("test"), debug_size=None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    print("Initializing model and loss...")
    # Compute class weights to handle imbalance
    class_weights = compute_class_weights(train_dataset.df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = HybridResNet()

    # 4. Training
    # Stage 1: Warmup (Freeze backbone, train head with L-BFGS)
    model = train_stage1_warmup(model, train_loader, val_loader, criterion, device)

    # Stage 2: Fine-tuning (Unfreeze Layer 4, train with AdamW)
    model = train_stage2_finetune(model, train_loader, val_loader, criterion, device)

    # 5. Final Validation Assessment
    print("\n=== Final Validation Assessment ===")
    val_loss, val_f1 = validate(model, val_loader, criterion, device)
    # Print the required metric format
    print(f"Final Validation Metric: {val_f1}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, val_dataset.df, device)

    # 7. Submission
    # Threshold defined in requirements
    threshold = 0.2804134650748149

    if val_f1 > threshold:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({val_f1}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
