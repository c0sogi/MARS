import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import (
    SEED,
    DEVICE,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    INPUT_ROOT,
    BATCH_SIZE,
    NUM_WORKERS,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    seed_everything,
    NUM_CLASSES,
)
from library.dataset import CameraTrapDataset, get_transforms
from library.model import get_model
from library.loss import ClassBalancedFocalLoss
from library.utils import get_class_weights, calculate_macro_f1
from library.engine import train_model, evaluate, predict_and_submit


def run():
    # 1. Reproducibility
    seed_everything(SEED)
    print(f"Device: {DEVICE}")

    # 2. Data Preparation
    print("Loading metadata...")
    # Load full training metadata to calculate class weights based on the true distribution
    df_train_full = pd.read_csv(TRAIN_META_PATH)

    # Subset training data using "Preserve Rare, Undersample Majority" strategy
    # Cite Lesson 00008: Prioritize exposure to rare samples over model complexity.
    # We keep ALL minority class samples and undersample the majority (empty) class
    # to fit within the compute budget while maximizing information.

    TOTAL_BUDGET = 65000

    df_minority = df_train_full[df_train_full["Category"] != 0]
    df_majority = df_train_full[df_train_full["Category"] == 0]

    n_minority = len(df_minority)
    n_majority_to_sample = TOTAL_BUDGET - n_minority

    if n_majority_to_sample > 0 and n_majority_to_sample < len(df_majority):
        print(
            f"Undersampling majority class. Keeping all {n_minority} minority samples."
        )
        df_majority_sampled = df_majority.sample(
            n=n_majority_to_sample, random_state=SEED
        )
        df_train_subset = (
            pd.concat([df_minority, df_majority_sampled])
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )
    else:
        # Fallback if budget is large enough for everything or minority is huge
        if len(df_train_full) > TOTAL_BUDGET:
            df_train_subset = df_train_full.sample(
                n=TOTAL_BUDGET, random_state=SEED
            ).reset_index(drop=True)
        else:
            df_train_subset = df_train_full

    print(f"Training data prepared: {len(df_train_subset)} samples.")

    # Calculate class weights based on the actual training subset distribution
    # This ensures the loss is calibrated to the batches the model actually sees.
    class_weights = get_class_weights(df_train_subset)
    print("Class weights calculated for subset.")

    # Load Validation Metadata
    df_val = pd.read_csv(VAL_META_PATH)

    # Create Datasets
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    train_dataset = CameraTrapDataset(
        df_train_subset, INPUT_ROOT, transform=train_transform
    )
    val_dataset = CameraTrapDataset(df_val, INPUT_ROOT, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # 4. Loss, Optimizer, Scheduler
    # Using Class Balanced Focal Loss as per strategy
    criterion = ClassBalancedFocalLoss(alpha=class_weights, gamma=2.0).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Increased epochs slightly to account for larger dataset and harder negatives
    NUM_EPOCHS = 12
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. Training
    print("Starting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=NUM_EPOCHS,
        patience=3,  # Early stopping patience
        device=DEVICE,
    )

    # 6. Final Validation & Metric
    print("Performing final validation...")
    val_loss, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_f1}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    model.eval()
    all_targets = []
    all_preds = []

    # Collect predictions (using no_grad for speed/memory)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate binary error (0 = correct, 1 = incorrect)
    errors = (all_targets != all_preds).astype(int)

    # Feature for correlation: Class Frequency (Log scale to handle skew)
    # We map each target label to its frequency in the training set
    train_class_counts = df_train_full["Category"].value_counts().sort_index()
    target_frequencies = np.array([train_class_counts.get(t, 0) for t in all_targets])

    # Calculate correlation
    if np.std(errors) > 0 and np.std(target_frequencies) > 0:
        correlation = np.corrcoef(errors, target_frequencies)[0, 1]
        print(f"Correlation between Error Magnitude and Class Frequency: {correlation}")
    else:
        print("Could not calculate correlation (zero variance in errors or features).")

    # 8. Submission
    THRESHOLD = 0.44583477715072195
    if val_f1 > THRESHOLD:
        print(
            f"Validation metric {val_f1} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        df_test = pd.read_csv(TEST_META_PATH)
        test_transform = get_transforms("test")
        test_dataset = CameraTrapDataset(
            df_test, INPUT_ROOT, transform=test_transform, is_test=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        predict_and_submit(
            model, test_loader, device=DEVICE, submission_path=SUBMISSION_FILE_PATH
        )
    else:
        print(
            f"Validation metric {val_f1} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
