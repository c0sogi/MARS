import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_f1_macro
from library.dataset import get_dataloaders
from library.model import train_model, validate, generate_submission


def run_pipeline():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Fast Baseline Configuration
    # We limit training to 1 epoch to ensure the task completes within the 2-hour limit.
    # We use the full dataset to ensure the model sees all 64,500 classes.
    Config.EPOCHS = 1
    Config.USE_SWA = False

    print(f"Starting Fast Baseline Run")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(
            "Metadata not found. Please ensure metadata generation is complete."
        )

    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Pre-calculate class counts for failure analysis
    train_class_counts = train_df["category_id"].value_counts().to_dict()

    # 3. Initialize DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Train Model
    print("Starting training...")
    train_start = time.time()

    # train_model handles the training loop, saving the best model, and returning it.
    model = train_model(train_loader, val_loader)

    train_duration = time.time() - train_start
    print(f"Training completed in {train_duration:.2f} seconds.")

    # 5. Validation
    print("Performing final validation...")
    device = torch.device(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Calculate metric on the full validation set
    val_loss, val_f1 = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_f1}")

    # 6. Failure Analysis
    print("Running Failure Analysis...")
    model.eval()

    # Collect predictions and labels for analysis
    all_preds = []
    all_labels = []

    # We iterate over val_loader to get sample-level predictions
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate binary error (1 = Incorrect, 0 = Correct)
    errors = (all_preds != all_labels).astype(int)

    # Analysis 1: Correlation with Class Frequency
    # Map each validation sample's true label to its frequency in the training set
    sample_freqs = [train_class_counts.get(label, 0) for label in all_labels]

    if len(set(errors)) > 1:  # Ensure we have both errors and successes
        corr_freq, p_freq = stats.pointbiserialr(errors, sample_freqs)
        print(
            f"Correlation (Error vs. Class Frequency): {corr_freq:.6f} (p-value: {p_freq:.6e})"
        )
    else:
        print(
            "Correlation (Error vs. Class Frequency): Undefined (All samples correct or all wrong)"
        )

    # Analysis 2: Correlation with Input Feature (File Size)
    # We sample a subset to avoid excessive disk I/O overhead
    sample_indices = np.random.choice(
        len(val_df), size=min(2000, len(val_df)), replace=False
    )

    file_sizes = []
    sampled_errors = []

    for idx in sample_indices:
        rel_path = val_df.iloc[idx]["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            size = os.path.getsize(full_path)
            file_sizes.append(size)
            sampled_errors.append(errors[idx])
        except Exception:
            continue

    if len(file_sizes) > 0 and len(set(sampled_errors)) > 1:
        corr_size, p_size = stats.pointbiserialr(sampled_errors, file_sizes)
        print(
            f"Correlation (Error vs. File Size): {corr_size:.6f} (p-value: {p_size:.6e})"
        )
    else:
        print(
            "Correlation (Error vs. File Size): Skipped (Insufficient data or variance)"
        )

    # 7. Submission
    THRESHOLD = 0.3544800410153631

    if val_f1 > THRESHOLD:
        print(
            f"Validation F1 ({val_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader)
    else:
        print(
            f"Validation F1 ({val_f1}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
