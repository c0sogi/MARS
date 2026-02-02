import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config
from library.data_utils import get_dataloaders, FeatureEngineer
from library.model_utils import ParallelLowRankDCNResNet, predict_and_submit
from library.train_utils import run_training, validate


def main():
    # 1. Configuration Override for Fast Baseline
    # We limit epochs to 20 to ensure the run completes within the 2-hour limit
    # while using the full dataset to meet the high accuracy requirement.
    Config.EPOCHS = 20
    Config.set_seed(Config.SEED)

    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # 2. Data Loading
    print("Loading data...")
    # Use full dataset (debug=False) to maximize performance
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # Determine input dimension from a single batch
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]
    print(f"Input Feature Dimension: {input_dim}")

    # 3. Model Initialization
    print("Initializing Parallel Low-Rank DCN-ResNet...")
    model = ParallelLowRankDCNResNet(
        input_dim=input_dim, num_classes=Config.NUM_CLASSES
    )

    # 4. Training
    print("Starting training loop...")
    # run_training handles optimizer, scheduler, and early stopping, returning the best model
    best_model = run_training(model, train_loader, val_loader)

    # 5. Final Validation
    print("Computing final validation metrics...")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate(best_model, val_loader, criterion, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")

    # Reconstruct feature names for meaningful reporting
    try:
        # Load a tiny sample to get columns, applying the same feature engineering
        df_sample = pd.read_parquet(Config.TRAIN_PATH).head(100)
        fe = FeatureEngineer()
        df_sample = fe.engineer_features(df_sample)
        # Exclude ID and Target to match the X matrix
        feature_names = [
            c for c in df_sample.columns if c not in [Config.ID_COL, Config.TARGET_COL]
        ]
    except Exception as e:
        print(f"Warning: Could not reconstruct feature names ({e}). Using indices.")
        feature_names = [f"Feature_{i}" for i in range(input_dim)]

    # Collect predictions and inputs for the entire validation set
    best_model.eval()
    all_preds = []
    all_labels = []
    all_inputs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            # Move inputs back to CPU to conserve GPU memory
            all_inputs.append(inputs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_inputs = np.concatenate(all_inputs)

    # Calculate Error (1 = Incorrect, 0 = Correct)
    errors = (all_preds != all_labels).astype(int)
    error_rate = errors.mean()
    print(f"Overall Error Rate: {error_rate:.6f}")

    # Calculate Correlations between each feature and the error vector
    correlations = []
    for i in range(len(feature_names)):
        feat_vals = all_inputs[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feat_vals) < 1e-9 or np.std(errors) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation (magnitude of association)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Model Failure (Correlation with Error):")
    for name, corr in correlations[:10]:
        print(f"  {name:<35} : {corr:.6f}")

    # 7. Submission Logic
    THRESHOLD = 0.9625041666666667
    if val_acc > THRESHOLD:
        print(
            f"\nValidation accuracy ({val_acc:.8f}) exceeds threshold ({THRESHOLD:.8f})."
        )
        predict_and_submit(best_model, test_loader, test_ids)
    else:
        print(
            f"\nValidation accuracy ({val_acc:.8f}) does not exceed threshold ({THRESHOLD:.8f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
