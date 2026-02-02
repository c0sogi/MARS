import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import ParallelDCNResNeXt, generate_submission
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration
    # Enforce reproducibility
    seed_everything(Config.SEED)
    print("Orchestration script started.")

    # 2. Data Loading
    # Load cached data to save time, or process from scratch if cache is missing
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine model input specifications dynamically from data
    # train_loader.dataset.X is a torch.Tensor
    input_dim = train_loader.dataset.X.shape[1]
    num_classes = Config.NUM_CLASSES
    print(
        f"Data loaded successfully. Input Dimension: {input_dim}, Classes: {num_classes}"
    )

    # 3. Model Initialization
    print("Initializing Parallel DCN-ResNeXt model...")
    model = ParallelDCNResNeXt(input_dim=input_dim, num_classes=num_classes)

    # 4. Training
    # Initialize Trainer to handle the training loop, scheduling, and early stopping
    print("Starting training phase...")
    trainer = Trainer(model, train_loader, val_loader)
    best_model = trainer.fit()

    # 5. Validation and Failure Analysis
    print("\nRunning final validation and failure analysis...")

    device = Config.DEVICE
    best_model.eval()
    best_model.to(device)

    all_preds = []
    all_targets = []

    # Efficient inference loop (no gradients)
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = best_model(data)
            preds = torch.argmax(output, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Concatenate batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate and print the required metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis: Correlation between features and error
    print("Analyzing systematic failures...")

    # Error vector: 1 if prediction was wrong, 0 if correct
    errors = (all_preds != all_targets).astype(int)

    # Retrieve feature matrix (convert tensor to numpy)
    val_features = val_loader.dataset.X.numpy()

    # Reconstruct feature names based on the engineering pipeline order
    # Order in features.py: Scaled(Continuous + Engineered) + Binary
    feature_names = Config.CONTINUOUS_COLS + Config.ENGINEERED_COLS + Config.BINARY_COLS

    # Verify dimensions match
    if val_features.shape[1] != len(feature_names):
        print(
            f"Warning: Feature name count ({len(feature_names)}) != Matrix width ({val_features.shape[1]}). Using generic names."
        )
        feature_names = [f"Feature_{i}" for i in range(val_features.shape[1])]

    correlations = []
    # Calculate correlation for each feature against the error vector
    for i, name in enumerate(feature_names):
        feat_col = val_features[:, i]

        # Handle constant features to avoid division by zero
        if np.std(feat_col) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]

        correlations.append((name, corr))

    # Sort by magnitude of correlation (absolute value)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features Correlated with Prediction Error:")
    print(f"{'Feature':<40} {'Correlation':<10}")
    print("-" * 52)
    for name, corr in correlations[:10]:
        print(f"{name:<40} {corr:.6f}")

    # 6. Submission Generation
    threshold = 0.9625041666666667

    if accuracy > threshold:
        print(
            f"\nValidation accuracy ({accuracy:.6f}) exceeds threshold ({threshold:.6f})."
        )
        print("Generating submission file...")
        generate_submission(best_model, test_loader, device=device)
    else:
        print(
            f"\nValidation accuracy ({accuracy:.6f}) does not exceed threshold ({threshold:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
