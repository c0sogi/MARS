import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.model import SimpleConvNet
from library.trainer import train
from library.inference import generate_submission


def set_seed(seed=Config.SEED):
    """Sets fixed random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # Ensure deterministic behavior for cuDNN if needed,
    # though usually adds overhead. For a baseline, standard seeding is often enough.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_validation_and_failure_analysis(model, val_loader, device):
    """
    Evaluates the model, calculates accuracy, and performs failure analysis.
    """
    model.eval()

    all_preds = []
    all_labels = []

    # For failure analysis
    input_means = []
    input_stds = []
    errors = []  # 0 for correct, 1 for incorrect

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Collect metrics
            preds_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)

            # Failure Analysis Data Collection
            # Calculate stats per sample in the batch
            # inputs shape: (Batch, 1, Freq, Time)
            # We flatten last 3 dims to compute stats per sample
            batch_flattened = inputs.view(inputs.size(0), -1)

            batch_means = torch.mean(batch_flattened, dim=1).cpu().numpy()
            batch_stds = torch.std(batch_flattened, dim=1).cpu().numpy()

            input_means.extend(batch_means)
            input_stds.extend(batch_stds)

            # Error: 1 if incorrect, 0 if correct
            batch_errors = (preds_np != labels_np).astype(int)
            errors.extend(batch_errors)

    # 1. Validation Metric
    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Metric: {acc}")

    # 2. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.array(errors)
    input_means = np.array(input_means)
    input_stds = np.array(input_stds)

    # Calculate correlations
    # We handle cases where std might be 0 (constant input) though unlikely with audio
    if len(errors) > 1 and np.std(errors) > 0:
        corr_mean, _ = pearsonr(errors, input_means)
        corr_std, _ = pearsonr(errors, input_stds)

        print(f"Correlation between Error and Input Mean Intensity: {corr_mean:.6f}")
        print(f"Correlation between Error and Input Standard Deviation: {corr_std:.6f}")

        if abs(corr_mean) > 0.1 or abs(corr_std) > 0.1:
            print(
                "Observation: Weak to moderate correlation detected between signal properties and error rate."
            )
        else:
            print(
                "Observation: No significant linear correlation between simple signal stats and error rate."
            )
    else:
        print("Could not calculate correlation (possibly 0 variance in errors).")


def main():
    # 1. Setup
    set_seed()
    device = torch.device(Config.DEVICE)

    # 2. Train
    print("Starting Training...")
    train(debug=False, epochs=Config.NUM_EPOCHS)

    # 3. Load Best Model for Validation
    print("\nLoading best model for validation...")
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"
        )

    model = SimpleConvNet(num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model = model.to(device)

    # Get Validation Loader
    # We ignore the train loader here
    _, val_loader = get_dataloaders(debug=False)

    # 4. Run Validation and Failure Analysis
    run_validation_and_failure_analysis(model, val_loader, device)

    # 5. Generate Submission
    # Only generate submission if metric is better than baseline
    BASELINE_METRIC = 0.9207291579563509

    if acc > BASELINE_METRIC:
        print(
            f"\nValidation metric {acc:.6f} > baseline {BASELINE_METRIC:.6f}. Generating Submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation metric {acc:.6f} <= baseline {BASELINE_METRIC:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
