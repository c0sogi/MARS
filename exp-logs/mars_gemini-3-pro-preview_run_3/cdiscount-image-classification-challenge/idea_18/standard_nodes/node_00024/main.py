import os
import sys
import numpy as np
import pandas as pd
import torch
import time

# Import provided libraries
from library.config import Config
from library.feature_engine import FeatureEngine
from library.core import train_ensemble, generate_submission
from library.dataset import create_dataloader
from library.model import ProjectedMultiTaskMLP
from library.utils import seed_everything, HierarchyMapper


def failure_analysis(model, val_loader, device):
    print("\n=== Failure Analysis ===")
    model.eval()

    all_preds = []
    all_targets = []
    all_features = []

    # Collect predictions and features
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            # targets is tuple (l1, l2, l3)
            target_l3 = targets[2].to(device)

            # Forward pass
            _, _, logits_l3 = model(features)
            preds = torch.argmax(logits_l3, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target_l3.cpu().numpy())
            # Store features for correlation analysis
            all_features.append(features.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_features = np.concatenate(all_features, axis=0)

    # Calculate Accuracy
    errors = (all_preds != all_targets).astype(int)
    accuracy = 1.0 - errors.mean()

    print(f"Final Validation Metric: {accuracy}")

    # Correlation Analysis
    # Calculate point-biserial correlation between each feature and the error
    # Error is binary (0/1), Features are continuous

    X = all_features
    y = errors

    # Check if we have enough variance in errors to compute correlation
    if y.std() == 0:
        print("All predictions are correct (or all wrong). Correlation undefined.")
        return accuracy

    # Centering
    X_mean = X.mean(axis=0)
    y_mean = y.mean()

    X_centered = X - X_mean
    y_centered = y - y_mean

    # Covariance
    covariance = (X_centered * y_centered[:, None]).sum(axis=0)

    # Standard Deviations
    X_std = np.sqrt((X_centered**2).sum(axis=0))
    y_std = np.sqrt(np.sum(y_centered**2))

    # Correlation
    epsilon = 1e-8
    correlation = covariance / (X_std * y_std + epsilon)

    # Summary Stats
    max_corr = np.max(correlation)
    min_corr = np.min(correlation)
    mean_abs_corr = np.mean(np.abs(correlation))
    top_feature_idx = np.argmax(np.abs(correlation))

    print(f"Error-Feature Correlation Summary:")
    print(f"  Max Correlation: {max_corr:.6f}")
    print(f"  Min Correlation: {min_corr:.6f}")
    print(f"  Mean Abs Correlation: {mean_abs_corr:.6f}")
    print(f"  Top Feature Index: {top_feature_idx}")

    return accuracy


def main():
    # 1. Configure for Fast Baseline
    # We override Config attributes to fit within the time limit
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20000  # Process 20k samples to ensure completion < 35 mins
    Config.EPOCHS = 5  # Train for 5 epochs
    Config.NUM_MODELS = 1  # Train 1 model (no ensemble)
    Config.TRAIN_BATCH_SIZE = 1024

    print(
        f"Configuration: DEBUG={Config.DEBUG}, SAMPLES={Config.DEBUG_SAMPLES}, EPOCHS={Config.EPOCHS}"
    )

    # 2. Feature Generation
    # Extracts features from BSON if not already cached
    engine = FeatureEngine()
    engine.generate_features(load_cached_data=True)

    # 3. Model Training
    # Trains the model(s) and saves checkpoints
    train_ensemble()

    # 4. Validation & Analysis
    device = torch.device(Config.DEVICE)

    # Load Hierarchy Mapper
    mapper = HierarchyMapper(Config.CATEGORY_NAMES)
    mapper.process(load_cached=True)

    # Create Validation Loader
    val_loader = create_dataloader(
        Config.VAL_FEATURES,
        Config.VAL_LABELS,
        mapper,
        mode="val",
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    # Load the trained model
    # Since NUM_MODELS=1, we look for model_0.pth
    model_path = os.path.join(Config.WORKING_DIR, "model_0.pth")
    if not os.path.exists(model_path):
        print("Model checkpoint not found. Training may have failed.")
        return

    model = ProjectedMultiTaskMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Run Failure Analysis and Get Metric
    final_metric = failure_analysis(model, val_loader, device)

    # 5. Submission Logic
    # Threshold from instructions
    THRESHOLD = 0.6239621493939094

    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")
        generate_submission()
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    seed_everything(Config.SEED)
    main()
