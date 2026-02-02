import sys
import os
import numpy as np
import pandas as pd
import torch
import random
import warnings

# Import provided library modules
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import ParallelDCNResNet, predict_and_submit
from library.train_utils import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for Fast Baseline execution
    # Reducing epochs to 10 ensures the script completes quickly (Fast Baseline constraint)
    # while utilizing the A100's speed to process the full dataset.
    Config.EPOCHS = 10

    print(f"Initializing Fast Baseline Run (Epochs={Config.EPOCHS})...")

    # 2. Data Loading
    # We use the full dataset (debug_sample_size=None) to aim for the high validation threshold.
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    model = ParallelDCNResNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)

    # 4. Training
    trainer = Trainer(model)
    trained_model = trainer.fit(train_loader, val_loader)

    # 5. Final Validation & Failure Analysis
    print("Performing Final Validation & Failure Analysis...")
    device = Config.DEVICE
    trained_model.eval()
    trained_model.to(device)

    all_preds = []
    all_labels = []
    all_inputs = []

    correct = 0
    total = 0

    # Iterate through validation set to calculate metric and collect data for analysis
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs_dev = inputs.to(device)
            labels_dev = labels.to(device)

            outputs = trained_model(inputs_dev)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels_dev).sum().item()

            # Collect data for failure analysis (move to CPU)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    final_acc = correct / total
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # --- Failure Analysis ---
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_inputs = np.concatenate(all_inputs)

    # Binary error mask (1 if error, 0 if correct)
    errors = (all_preds != all_labels).astype(int)

    print("\nFailure Analysis: Correlation of features with Error")

    # Reconstruct feature names to map correlations back to features
    new_continuous = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Distance_To_Hydrology",
        "Hydrology_Elevation",
        "Mean_Distance_To_Amenities",
    ]
    continuous_cols = Config.CONTINUOUS_FEATURES + new_continuous
    binary_cols = Config.BINARY_FEATURES
    feature_names = continuous_cols + binary_cols

    n_features = all_inputs.shape[1]
    correlations = []

    # Ensure feature name list matches dimension
    if n_features != len(feature_names):
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    for i in range(n_features):
        feat_vals = all_inputs[:, i]
        # Skip constant features to avoid NaN
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            # Calculate correlation between feature value and error probability
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by magnitude of correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Error:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # 6. Conditional Submission
    THRESHOLD = 0.9625041666666667

    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric {final_acc} > {THRESHOLD}. Generating submission..."
        )
        predict_and_submit(trained_model, test_loader, test_ids)
    else:
        print(f"\nValidation metric {final_acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
