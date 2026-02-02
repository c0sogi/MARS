import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, compute_auc
from library.dataset import get_dataloaders
from library.model import MultiBandResNetCRNN
from library.trainer import Trainer


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Correlates prediction error with spectrogram statistics.
    """
    print("\n=== Starting Failure Analysis ===")
    model.eval()

    errors = []
    spec_means = []
    spec_stds = []
    spec_maxs = []

    # We need to iterate through the loader to get data and predictions
    # We will compute stats on the spectrograms directly

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            # Get predictions
            output = model(data).squeeze(1)
            probs = torch.sigmoid(output)

            # Calculate absolute error
            batch_errors = torch.abs(probs - target).cpu().numpy()
            errors.extend(batch_errors)

            # Calculate spectrogram statistics (per sample in batch)
            # data shape: (B, 1, F, T)
            # Flatten spatial dims for stats: (B, F*T)
            flat_specs = data.view(data.size(0), -1)

            batch_means = flat_specs.mean(dim=1).cpu().numpy()
            batch_stds = flat_specs.std(dim=1).cpu().numpy()
            batch_maxs = flat_specs.max(dim=1).values.cpu().numpy()

            spec_means.extend(batch_means)
            spec_stds.extend(batch_stds)
            spec_maxs.extend(batch_maxs)

    # Create DataFrame
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "spec_mean": spec_means,
            "spec_std": spec_stds,
            "spec_max": spec_maxs,
        }
    )

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")

    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # Identify systematic patterns
    print("\nSystematic Error Patterns:")
    for feature, corr in correlations.items():
        if abs(corr) > 0.1:
            direction = "increases" if corr > 0 else "decreases"
            print(
                f"- Error tends to increase as {feature} {direction} (corr: {corr:.4f})"
            )
        else:
            print(
                f"- No significant linear correlation with {feature} (corr: {corr:.4f})"
            )


def main():
    # 1. Configuration Overrides for Fast Baseline
    # We limit epochs to ensure the script completes quickly while still learning.
    Config.EPOCHS = 6

    # 2. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 3. Data Loading
    print("Initializing DataLoaders...")
    # Use cached data to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 4. Model Initialization
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = MultiBandResNetCRNN().to(device)

    # 5. Training
    print("Initializing Trainer...")
    trainer = Trainer(model, train_loader, val_loader, device)

    print("Starting Training Loop...")
    trainer.fit(epochs=Config.EPOCHS)

    # 6. Evaluation on Validation Set
    print("\n=== Final Validation Evaluation ===")
    # Load the best model saved during training
    trainer.load_best_model()

    # Perform inference on validation set
    _, val_auc = trainer.validate()

    # REQUIRED: Print the final metric in the exact format
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.9946524988681537

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(test_loader)
    else:
        print(
            f"\nValidation AUC ({val_auc}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
