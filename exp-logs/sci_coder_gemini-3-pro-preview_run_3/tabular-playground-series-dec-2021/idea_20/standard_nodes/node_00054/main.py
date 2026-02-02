import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import get_dataloaders, set_seed
from library.model import ParallelVectorDCNResNet, generate_submission
from library.train_utils import run_training


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Load cached data to save time and utilize preprocessed artifacts
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = ParallelVectorDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        resnet_blocks=Config.RESNET_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
        num_cross_layers=2,  # As per experiment design
    ).to(device)

    # 4. Training
    # run_training handles the loop, scheduler, early stopping, and returns the best model
    print("Starting Training...")
    model = run_training(
        model,
        train_loader,
        val_loader,
        device,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 5. Validation & Failure Analysis
    print("Running Validation and Failure Analysis...")
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Get probability assigned to the true class
            # targets are 0-6 indices
            true_probs = probs.gather(1, targets.view(-1, 1)).squeeze()

            _, preds = outputs.max(1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(true_probs.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_inputs = np.concatenate(all_inputs)

    # Calculate Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis: Correlation between features and Error Magnitude
    # Error Magnitude = 1.0 - Probability of True Class
    error_magnitude = 1.0 - all_probs

    # Reconstruct feature names for reporting based on Config logic
    cont_cols = list(Config.CONTINUOUS_COLS)
    if Config.USE_CYCLICAL_ASPECT:
        cont_cols.extend(["Aspect_Sin", "Aspect_Cos"])
    if Config.USE_EUCLIDEAN_HYDRO:
        cont_cols.append("Hydro_Euclidean")
    if Config.USE_ABS_HYDRO_ELEV:
        cont_cols.append("Hydro_Elevation")
    if Config.USE_MEAN_AMENITIES:
        cont_cols.append("Mean_Amenities")
    bin_cols = list(Config.BINARY_COLS)
    feature_names = cont_cols + bin_cols

    # Fallback if dimensions don't match
    if len(feature_names) != all_inputs.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(all_inputs.shape[1])]

    correlations = []
    for i in range(all_inputs.shape[1]):
        feat_vals = all_inputs[:, i]

        # Calculate correlation safely
        std_feat = np.std(feat_vals)
        std_err = np.std(error_magnitude)

        if std_feat < 1e-9 or std_err < 1e-9:
            corr = 0.0
        else:
            # np.corrcoef returns matrix [[1, r], [r, 1]]
            corr = np.corrcoef(feat_vals, error_magnitude)[0, 1]
            if np.isnan(corr):
                corr = 0.0

        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.4f}")

    # 6. Submission
    threshold = 0.9625041666666667
    if accuracy > threshold:
        print(f"\nValidation metric {accuracy} > {threshold}. Generating submission...")
        generate_submission(model, test_loader, test_ids, device)
    else:
        print(f"\nValidation metric {accuracy} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
