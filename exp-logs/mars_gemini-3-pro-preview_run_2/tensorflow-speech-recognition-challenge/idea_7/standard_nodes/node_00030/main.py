import sys
import os
import torch
import numpy as np
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.trainer import Trainer
from library.dataset import get_dataloaders


def main():
    # Set global seed for reproducibility
    set_seed(42)

    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # Optimize hyperparameters for A100 GPU and time constraints.
    # Increasing batch size speeds up training significantly.
    Config.BATCH_SIZE = 512
    # Reducing epochs ensures the run finishes quickly (Fast Baseline requirement)
    # while 15 epochs is sufficient for this architecture to converge.
    Config.NUM_EPOCHS = 15
    # Update scheduler duration to match the new epoch count
    Config.T_MAX = 15

    # Check device availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be slow.")

    # ==========================================
    # 2. Training
    # ==========================================
    # Initialize trainer and start training loop
    trainer = Trainer()
    trainer.fit(load_cached_data=True)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    # We perform a dedicated validation pass to compute the exact metric
    # and gather data for failure analysis.

    # Load validation data (cached for speed)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Ensure model is in evaluation mode and on the correct device
    trainer.model.eval()
    device = trainer.device

    all_probs = []
    all_targets = []
    all_means = []
    all_stds = []

    # Inference loop without gradient calculation
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = trainer.model(features)
            probs = torch.softmax(outputs, dim=1)

            # Store probabilities and targets
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

            # Compute input feature statistics for failure analysis
            # features shape: (Batch, 1, F, T)
            # Flatten to (Batch, F*T) to compute mean/std per sample
            flat_features = features.view(features.size(0), -1)
            all_means.append(flat_features.mean(dim=1).cpu().numpy())
            all_stds.append(flat_features.std(dim=1).cpu().numpy())

    # Concatenate batches
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_means = np.concatenate(all_means)
    all_stds = np.concatenate(all_stds)

    # Compute Accuracy
    preds = np.argmax(all_probs, axis=1)
    correct = preds == all_targets
    accuracy = np.mean(correct)

    # Print the required metric string
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    # Calculate Error Magnitude: 1.0 - Probability assigned to the correct class
    # We use advanced indexing to select the probability of the true label for each sample
    rows = np.arange(len(all_targets))
    true_class_probs = all_probs[rows, all_targets]
    error_magnitudes = 1.0 - true_class_probs

    # Calculate Pearson correlation between Error Magnitude and Input Statistics
    # np.corrcoef returns a correlation matrix; we take the off-diagonal element [0, 1]
    corr_mean = np.corrcoef(error_magnitudes, all_means)[0, 1]
    corr_std = np.corrcoef(error_magnitudes, all_stds)[0, 1]

    print("\nFailure Analysis:")
    print(
        f"Correlation between Error Magnitude and Input Mean (Signal Volume): {corr_mean}"
    )
    print(
        f"Correlation between Error Magnitude and Input Std (Signal Contrast): {corr_std}"
    )

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.9853666694539677

    if accuracy > THRESHOLD:
        print(f"\nValidation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        trainer.predict_and_submit(load_cached_data=True)
    else:
        print(
            f"\nValidation accuracy ({accuracy}) does not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
