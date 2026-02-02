import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.dataset import BreastCancerDataset
from library.model import TriSpectralHybridModel
from library.engine import fit, evaluate, set_seed, pf1_score


def analyze_failures(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and metadata features.
    """
    model.eval()
    all_probs = []
    all_targets = []

    # Inference on validation set
    with torch.no_grad():
        for images, tabular, labels in val_loader:
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)

            # Forward pass
            logits = model(images, tabular)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Calculate Absolute Error
    errors = np.abs(all_probs - all_targets)

    print("\n=== Failure Analysis ===")
    print(f"Analyzing {len(errors)} validation samples.")

    # Features to analyze
    features = ["age", "implant", "machine_id"]

    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 50)

    for feat in features:
        if feat in val_df.columns:
            # Get feature values
            values = val_df[feat].values

            # Handle missing values (simple mean imputation for analysis)
            if pd.isnull(values).any():
                if np.issubdtype(values.dtype, np.number):
                    values = np.nan_to_num(values, nan=np.nanmean(values))
                else:
                    continue

            # Ensure numeric types
            if not np.issubdtype(values.dtype, np.number):
                continue

            # Calculate correlation
            try:
                corr, pval = pearsonr(values, errors)
                print(f"{feat:<20} | {corr:<12.4f} | {pval:<12.4f}")
            except Exception as e:
                print(f"{feat:<20} | Error: {e}")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\n=== Generating Submission ===")

    # Load Test Data
    test_dataset = BreastCancerDataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    model.eval()
    all_probs = []

    # Inference
    with torch.no_grad():
        for images, tabular, _ in test_loader:
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)

            logits = model(images, tabular)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    # Map to Prediction IDs
    test_df = test_dataset.df.copy()
    test_df["cancer"] = all_probs

    # Aggregate: Max probability per prediction_id
    submission = test_df.groupby("prediction_id")["cancer"].max().reset_index()

    # Save
    submission.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    print(f"Rows: {len(submission)}")
    print(submission.head())


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # Load full datasets first
    full_train_dataset = BreastCancerDataset(mode="train", load_cached_data=True)
    val_dataset = BreastCancerDataset(mode="val", load_cached_data=True)

    # Subsample Training Data for Fast Baseline
    # Limit to 5000 samples to ensure execution within 2 hours
    MAX_TRAIN_SAMPLES = 5000
    if len(full_train_dataset) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(full_train_dataset)} to {MAX_TRAIN_SAMPLES}..."
        )
        indices = np.random.choice(
            len(full_train_dataset), size=MAX_TRAIN_SAMPLES, replace=False
        )
        train_dataset = Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    print("Initializing TriSpectralHybridModel...")
    model = TriSpectralHybridModel().to(device)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Reduced epochs for fast baseline
    NUM_EPOCHS = 3

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    # 5. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=NUM_EPOCHS,
        checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
    )

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    # Define criterion for evaluation (must match training weighting)
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_loss, val_score = evaluate(model, val_loader, device, criterion)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, val_dataset.df, device)

    # 8. Submission
    THRESHOLD = 0.044888656586408615
    if val_score > THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"\nValidation score ({val_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
