import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config, process_data, feature_engineering
from library.data_loader import get_dataloaders
from library.model import get_model, predict
from library.trainer import run_training, set_seed, evaluate


def main():
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Load DataLoaders for training/inference
    train_loader, val_loader, test_loader, num_features, num_classes, test_ids = (
        get_dataloaders(load_cached_data=True)
    )

    # Load raw numpy arrays and label encoder for Analysis and Submission
    # We use process_data directly to access the underlying data structures
    _, _, X_val, y_val, _, _, _, _, label_encoder = process_data(load_cached_data=True)

    # 3. Model Initialization
    print(
        f"Initializing model with {num_features} features and {num_classes} classes..."
    )
    model = get_model(num_features, num_classes)
    model = model.to(device)

    # 4. Training
    # We execute the training loop. We use the config's epoch count (60) to ensure we have
    # sufficient convergence to beat the high threshold. The early stopping in run_training
    # will prevent wasting time if the model converges sooner.
    print("Starting training pipeline...")
    model = run_training(
        model, train_loader, val_loader, epochs=Config.EPOCHS, device=device
    )

    # 5. Final Validation Assessment
    print("Performing final validation assessment...")
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # Requirement: Print full precision validation metric
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Generate predictions on validation set
    val_preds_idx = predict(model, val_loader, device)

    # Calculate binary error (1 = Incorrect, 0 = Correct)
    errors = (val_preds_idx != y_val).astype(int)

    # Reconstruct feature names to make the analysis interpretable
    # We read a small sample of the metadata/train to deduce the column order
    try:
        # Read a tiny sample to get columns
        df_sample = pd.read_parquet(Config.TRAIN_DATA)
        df_sample = df_sample.drop(columns=["Id", "Cover_Type"], errors="ignore")

        # Apply the same feature engineering as the pipeline
        df_sample = feature_engineering(df_sample)

        # The pipeline stacks Continuous then Binary features
        binary_cols = [
            c
            for c in df_sample.columns
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]
        continuous_cols = [c for c in df_sample.columns if c not in binary_cols]
        feature_names = continuous_cols + binary_cols

        # Safety check
        if len(feature_names) != num_features:
            feature_names = [f"Feature_{i}" for i in range(num_features)]

    except Exception as e:
        print(f"Feature name reconstruction failed: {e}. Using generic names.")
        feature_names = [f"Feature_{i}" for i in range(num_features)]

    # Compute correlation between Features and Error
    # X_val is the processed (scaled) numpy array
    df_val_features = pd.DataFrame(X_val, columns=feature_names)
    df_val_features["Error_Flag"] = errors

    # Calculate correlations
    correlations = df_val_features.corr()["Error_Flag"].drop("Error_Flag")

    # Print top 10 features associated with errors
    print("Top 10 Features Correlated with Prediction Error:")
    print(correlations.abs().sort_values(ascending=False).head(10))

    # 7. Submission Generation
    threshold = 0.9626291666666666

    if val_acc > threshold:
        print(f"\nValidation metric {val_acc} > {threshold}. Generating submission...")

        # Generate predictions on test set
        test_preds_idx = predict(model, test_loader, device)

        # Map integer predictions back to original Class IDs (1, 2, 3, 4, 6, 7)
        final_preds = label_encoder.inverse_transform(test_preds_idx)

        # Create submission DataFrame
        submission = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

        # Save to disk
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {val_acc} <= {threshold}. Submission generation skipped."
        )


if __name__ == "__main__":
    main()
