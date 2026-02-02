import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

import library.config as config
import library.data_loader as data_loader
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def validate_best_model(trainer, val_loader):
    """
    Evaluates the best saved model on the validation set.
    """
    device = trainer.device
    # Load best model weights
    trainer.model.load_state_dict(torch.load(config.MODEL_PATH))
    trainer.model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = trainer.model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return correct / total


def perform_failure_analysis(trainer, val_loader):
    """
    Calculates correlations between prediction error and input features.
    """
    print("\nRunning Failure Analysis...")
    device = trainer.device
    # Ensure model is loaded (should be handled by validate_best_model call prior)
    trainer.model.eval()

    all_inputs = []
    all_preds = []
    all_targets = []

    # Collect data
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = swa_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            all_inputs.append(X_batch.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(y_batch.numpy())

    # Concatenate
    X_val = np.concatenate(all_inputs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate Error (0 for correct, 1 for incorrect)
    errors = (preds != targets).astype(int)

    # Get Feature Names
    feature_names = config.FINAL_CONTINUOUS_FEATURES + config.FINAL_BINARY_FEATURES

    # Ensure shape matches
    if X_val.shape[1] != len(feature_names):
        print(
            f"Warning: Feature count mismatch. Data: {X_val.shape[1]}, Names: {len(feature_names)}"
        )
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    # Create DataFrame
    df_analysis = pd.DataFrame(X_val, columns=feature_names)
    df_analysis["Error_Magnitude"] = errors

    # Compute Correlations
    correlations = df_analysis.corr()["Error_Magnitude"].drop("Error_Magnitude")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 Features correlated with Error Magnitude:")
    for feat in top_correlations.index:
        corr_val = correlations[feat]
        print(f"  {feat}: {corr_val:.6f}")


def main():
    # 1. Setup
    # -------------------------------------------------------------------------
    # Fix seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading Data...")
    # Using cached data if available to speed up execution
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Determine input dimension from a batch
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]

    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer(device=device, input_dim=input_dim)

    # Train the model (includes SWA logic)
    # We use the full epoch count from config to ensure SWA works as intended
    trainer.train(train_loader, val_loader, epochs=config.EPOCHS)

    # 4. Validation (Best Model)
    # -------------------------------------------------------------------------
    print("Validating Best Model...")
    final_metric = validate_best_model(trainer, val_loader)

    # REQUIRED FORMAT: Print full precision metric
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    perform_failure_analysis(trainer, val_loader)

    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9625041666666667

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
