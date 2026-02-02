import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders, apply_feature_engineering
from library.train_utils import train_model, predict_and_submit, evaluate


def main():
    # 1. Setup & Configuration
    print("Initializing pipeline...")
    device = torch.device(Config.DEVICE)

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # 2. Data Loading
    # Load data using cached files if available to speed up execution
    print("Loading datasets...")
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Training
    # Train the Deep Parallel Vector-DCN-ResNet
    # We use the full 60 epochs as defined in Config to maximize performance for the high threshold
    print(f"Starting training for {Config.EPOCHS} epochs...")
    model = train_model(
        train_loader, val_loader, input_dim, num_classes=7, epochs=Config.EPOCHS
    )

    # 4. Validation Assessment
    # Evaluate the best model on the hold-out validation set
    print("Evaluating best model on validation set...")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # Print the required metric format
    print(f"Final Validation Metric: {val_acc}")

    # 5. Failure Analysis
    print("\nPerforming failure analysis...")
    model.eval()

    # Collect all validation inputs, predictions, and labels
    all_inputs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Move to CPU to avoid OOM during accumulation
            all_inputs.append(inputs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    X_val = np.vstack(all_inputs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Calculate binary error (1 if incorrect, 0 if correct)
    errors = (y_pred != y_true).astype(int)

    # Reconstruct feature names to make analysis interpretable
    # We load a small sample of the raw data and apply the same FE pipeline
    try:
        df_sample = pd.read_parquet(Config.TRAIN_DATA_PATH)
        # Apply feature engineering
        df_sample = apply_feature_engineering(df_sample)

        # Replicate the column ordering logic from data_utils.process_data
        exclude_cols = ["Id", "Cover_Type"]
        binary_cols = [
            c
            for c in df_sample.columns
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]
        continuous_cols = [
            c for c in df_sample.columns if c not in binary_cols + exclude_cols
        ]
        feature_names = continuous_cols + binary_cols

        # Verify alignment
        if len(feature_names) != X_val.shape[1]:
            print(
                f"Warning: Feature name count ({len(feature_names)}) != Input Dim ({X_val.shape[1]}). Using generic names."
            )
            feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    except Exception as e:
        print(f"Error reconstructing feature names: {e}. Using generic names.")
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    # Calculate correlation between each feature and the error vector
    correlations = []
    for i in range(X_val.shape[1]):
        feature_col = X_val[:, i]
        # Skip constant features to avoid warnings
        if np.std(feature_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Prediction Error:")
    print(f"{'Feature':<40} {'Correlation':<10}")
    print("-" * 50)
    for name, corr in correlations[:10]:
        print(f"{name[:39]:<40} {corr:.4f}")
    print("-" * 50)

    # 6. Submission Generation
    # Strict threshold check
    THRESHOLD = 0.9625041666666667

    if val_acc > THRESHOLD:
        print(
            f"\nValidation accuracy ({val_acc:.6f}) exceeds threshold ({THRESHOLD:.6f})."
        )
        predict_and_submit(model, test_loader, test_ids)
    else:
        print(
            f"\nValidation accuracy ({val_acc:.6f}) did not exceed threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
