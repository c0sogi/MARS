import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Import library components
from library.config import Config
from library.utils import set_seed, get_device
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.inference import generate_submission


def main():
    # ==========================================
    # 1. Configuration Override for Fast Baseline
    # ==========================================
    # Adjust epochs to ensure execution finishes well within limits while allowing convergence.
    # A100 GPU can handle ~1.5k steps/epoch very quickly.
    Config.epochs = 15

    # Ensure reproducibility
    set_seed(Config.seed)
    device = get_device()

    print(f"Starting run with {Config.epochs} epochs on {device}...")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    trainer = Trainer()
    trainer.fit()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nRunning rigorous validation assessment...")

    # Load the best model for evaluation
    model = trainer.model
    checkpoint_path = Config.best_model_path

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    # Get validation loader
    _, val_loader, _ = get_dataloaders()

    all_preds = []
    all_targets = []

    # Inference loop (No Gradient)
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Metric
    val_accuracy = accuracy_score(all_targets, all_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_accuracy}")

    # Failure Analysis
    # Calculate error magnitude (Binary: 1 for error, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    # Calculate correlation between Error Magnitude and Input Feature (Target Label Index)
    # This identifies if specific classes are correlated with higher error rates.
    if len(errors) > 1:
        correlation = np.corrcoef(errors, all_targets)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error Magnitude and Target Label: {correlation}")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = 0.9866209549293419

    if val_accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy ({val_accuracy}) exceeds threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        generate_submission()
    else:
        print(
            f"\nValidation accuracy ({val_accuracy}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
