import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.utils import set_seed, load_checkpoint


def main():
    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    print("Initializing DataLoaders...")
    # Load full dataset to achieve high accuracy
    # The A100 GPU is sufficient to train on the full 46k samples quickly (approx 10-15 mins)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=False,
    )

    print("Initializing Trainer...")
    trainer = Trainer(device=Config.DEVICE)

    print("Starting Training...")
    # Train the model
    # We use the epochs defined in Config (40) to ensure convergence with Mixup augmentation
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # -------------------------------------------------------------------------
    # Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Validation Assessment & Failure Analysis...")

    # Load the best model saved during training to ensure analysis matches reported best metric
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    load_checkpoint(best_model_path, trainer.model, device=Config.DEVICE)
    trainer.model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    # Features for failure analysis: Mean and Std of the input spectrogram
    feat_means = []
    feat_stds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward pass
            outputs = trainer.model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Get the probability assigned to the correct class
            # targets is shape (B), view as (B, 1) for gather
            true_class_probs = probs.gather(1, targets.view(-1, 1)).squeeze()

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(true_class_probs.cpu().numpy())

            # Extract features from input tensor (B, 1, F, T) for failure analysis
            # Flatten spatial dims to calc stats per sample
            flat_inputs = inputs.view(inputs.size(0), -1)
            feat_means.extend(flat_inputs.mean(dim=1).cpu().numpy())
            feat_stds.extend(flat_inputs.std(dim=1).cpu().numpy())

    # Calculate Final Metric
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    final_acc = (all_preds == all_targets).mean()

    # Print required metric with full precision
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    # Error Magnitude = 1 - Probability(True Class)
    # If model is confident and correct, error is close to 0.
    # If model is wrong or uncertain, error is high.
    error_magnitudes = 1.0 - np.array(all_probs)

    # Calculate correlations
    if len(error_magnitudes) > 1:
        corr_mean, _ = pearsonr(error_magnitudes, feat_means)
        corr_std, _ = pearsonr(error_magnitudes, feat_stds)
    else:
        corr_mean, corr_std = 0.0, 0.0

    print("-" * 40)
    print("Failure Analysis (Correlation with Error Magnitude)")
    print("-" * 40)
    print(f"Input Spectrogram Mean Intensity: {corr_mean:.6f}")
    print(f"Input Spectrogram Std Deviation:  {corr_std:.6f}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9665551839464883

    if final_acc > THRESHOLD:
        print(
            f"Validation accuracy ({final_acc:.6f}) exceeds threshold ({THRESHOLD:.6f})."
        )
        print("Generating submission for test set...")
        trainer.predict(test_loader)
    else:
        print(
            f"Validation accuracy ({final_acc:.6f}) did not exceed threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
