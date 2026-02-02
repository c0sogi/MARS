import os
import sys
import numpy as np
import torch
import warnings
from scipy.stats import pearsonr

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(".")

# Import Library Modules
from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import EnergyGatedEfficientNet
from library.trainer import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for fast baseline execution
    Config.EPOCHS = 15

    # Set reproducibility
    set_seed(Config.SEED)
    print(f"Configuration: Epochs={Config.EPOCHS}, Device={Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(
        load_cached_data=True
    )

    num_classes = len(label_encoder)
    print(f"Data loaded. Fine-grained classes: {num_classes}")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing model...")
    model = EnergyGatedEfficientNet(num_classes=num_classes)

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    trainer = Trainer(model, train_loader, val_loader, test_loader, label_encoder)
    trainer.fit(epochs=Config.EPOCHS)

    # ---------------------------------------------------------
    # 5. Final Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\nRunning Final Validation on Best Model...")

    # Load the best model weights
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        print("Warning: Best model not found, using current weights.")

    model.eval()
    device = torch.device(Config.DEVICE)
    model.to(device)

    all_preds = []
    all_targets = []

    # Storage for Failure Analysis
    feature_energy_means = []
    feature_spec_means = []

    with torch.no_grad():
        for inputs, energy, targets in val_loader:
            inputs = inputs.to(device)
            energy_gpu = energy.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs, energy_gpu.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Collect features for failure analysis
            # Energy is (B, T), take mean per sample
            feature_energy_means.extend(energy.mean(dim=1).numpy())
            # Spectrogram inputs are (B, 1, F, T), take global mean
            feature_spec_means.extend(inputs.mean(dim=(1, 2, 3)).cpu().numpy())

    # Map predictions to competition targets
    pred_labels = label_encoder.inverse_transform(all_preds)
    target_labels = label_encoder.inverse_transform(all_targets)

    mapped_preds = [label_encoder.map_to_target(l) for l in pred_labels]
    mapped_targets = [label_encoder.map_to_target(l) for l in target_labels]

    # Calculate Accuracy
    correct_mapped = sum(1 for p, t in zip(mapped_preds, mapped_targets) if p == t)
    final_metric = correct_mapped / len(mapped_targets)

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Define Error: 1 if incorrect, 0 if correct
    error_flags = [1 if p != t else 0 for p, t in zip(mapped_preds, mapped_targets)]

    if len(error_flags) > 0 and np.sum(error_flags) > 0:
        # Correlation with Mean Energy
        corr_energy, _ = pearsonr(error_flags, feature_energy_means)
        print(f"Correlation between Error and Mean Energy: {corr_energy:.6f}")

        # Correlation with Spectrogram Intensity
        corr_spec, _ = pearsonr(error_flags, feature_spec_means)
        print(f"Correlation between Error and Spectrogram Intensity: {corr_spec:.6f}")
    else:
        print("No errors found or dataset too small for correlation analysis.")

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 0.9872909698996656

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
