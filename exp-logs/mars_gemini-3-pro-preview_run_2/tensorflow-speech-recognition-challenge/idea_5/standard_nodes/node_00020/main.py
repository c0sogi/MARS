import torch
import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import accuracy_score

# Import from provided library
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import SwinAudioClassifier
from library.train import Trainer, generate_submission


def pearson_corr(x, y):
    """Calculate Pearson correlation coefficient using NumPy."""
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return 0.0
    return num / den


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use the full dataset to ensure we can reach the high accuracy threshold.
    # The dataset size (approx 46k train) is small enough for A100 to handle quickly.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    model = SwinAudioClassifier(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # 4. Training
    # We limit epochs to 10 for a fast baseline.
    # Swin Transformer with pretrained weights converges relatively quickly.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        patience=5,
    )

    print("Starting training...")
    trainer.fit(num_epochs=10)

    # 5. Validation & Failure Analysis
    print("Loading best model for validation and failure analysis...")
    try:
        load_checkpoint(Config.CHECKPOINT_PATH, model, device=Config.DEVICE)
    except FileNotFoundError:
        print("Warning: Checkpoint not found. Using current model weights.")

    model.eval()

    all_preds = []
    all_targets = []
    error_magnitudes = []
    input_means = []
    input_stds = []

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    print("Evaluating on validation set...")
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Failure Analysis Data Collection
            # Error magnitude = 1 - probability of the correct class
            # Gather probabilities of the target classes
            target_probs = probs.gather(1, targets.view(-1, 1)).squeeze()
            errors = 1.0 - target_probs
            error_magnitudes.extend(errors.cpu().numpy())

            # Input features: Mean and Std of the spectrogram (per sample)
            # inputs shape: (B, 1, 224, 224)
            # Flatten spatial dims for stats: (B, 224*224)
            flat_inputs = inputs.view(inputs.size(0), -1)
            batch_means = flat_inputs.mean(dim=1)
            batch_stds = flat_inputs.std(dim=1)

            input_means.extend(batch_means.cpu().numpy())
            input_stds.extend(batch_stds.cpu().numpy())

    # Calculate Final Metric
    final_acc = accuracy_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_acc}")

    # Calculate Correlations for Failure Analysis
    corr_mean = pearson_corr(input_means, error_magnitudes)
    corr_std = pearson_corr(input_stds, error_magnitudes)

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (Input Mean vs Error Magnitude): {corr_mean:.6f}")
    print(f"Correlation (Input Std vs Error Magnitude):  {corr_std:.6f}")
    print("-" * 30)

    # 6. Submission
    # Threshold check
    THRESHOLD = 0.9853666694539677

    if final_acc > THRESHOLD:
        print(
            f"Validation accuracy ({final_acc}) passed threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation accuracy ({final_acc}) did not pass threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
