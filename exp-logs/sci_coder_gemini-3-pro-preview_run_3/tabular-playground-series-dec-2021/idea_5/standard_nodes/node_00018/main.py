import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.model import build_model
import library.train_eval as train_eval
import library.data_utils as data_utils

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Adjust hyperparameters for a fast but effective baseline on A100
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 4096
    Config.NUM_WORKERS = 4
    Config.DEBUG = False  # Use full dataset for maximum performance

    # Initialize seeds and directories
    Config.setup()

    # ==========================================
    # 2. Control Submission Generation
    # ==========================================
    # Monkey-patch the predict function in train_eval to prevent unconditional
    # submission generation. We will handle this manually based on the metric.
    original_predict_func = train_eval.predict

    def no_op_predict(model, test_loader, config=Config):
        pass

    train_eval.predict = no_op_predict

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Starting training pipeline...")
    # train_model handles data loading, training loop, validation, and saving the best model.
    model = train_eval.train_model(load_cached_data=True)

    # ==========================================
    # 4. Final Validation & Metric Calculation
    # ==========================================
    print("Loading validation data for final assessment...")
    # Load raw arrays for efficient analysis
    train_X, train_y, val_X, val_y, test_X, test_ids = data_utils.load_data(
        load_cached_data=True
    )

    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    # Create Validation Loader
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference
    correct_preds = 0
    total_preds = 0
    all_preds_list = []
    all_targets_list = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            all_preds_list.append(predicted.cpu().numpy())
            all_targets_list.append(labels.cpu().numpy())

    final_acc = correct_preds / total_preds

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_acc}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Concatenate predictions
    all_preds = np.concatenate(all_preds_list)
    all_targets = np.concatenate(all_targets_list)

    # Calculate Error Vector (1 = Error, 0 = Correct)
    errors = (all_preds != all_targets).astype(int)

    # Reconstruct Feature Names to match the column order in X
    # We load a tiny sample to run feature engineering and get column names
    temp_df = pd.read_parquet(Config.TRAIN_METADATA_PATH).iloc[:5]
    temp_df = data_utils.feature_engineering(temp_df)

    # Identify Continuous and Binary columns
    cont_cols = data_utils.get_continuous_columns()
    # Filter out non-feature columns from the dataframe to find binary cols
    all_cols = temp_df.columns.tolist()
    exclude_cols = [Config.TARGET_COL, Config.ID_COL] + cont_cols
    bin_cols = [
        c
        for c in all_cols
        if c not in exclude_cols and c not in [Config.TARGET_COL, Config.ID_COL]
    ]

    # The preprocessing stacks continuous then binary
    feature_names = cont_cols + bin_cols

    # Calculate Correlations between Features and Errors
    # Center data for covariance calculation
    X_centered = val_X - val_X.mean(axis=0)
    y_centered = errors - errors.mean()

    # Covariance
    covariance = np.dot(X_centered.T, y_centered) / (val_X.shape[0] - 1)

    # Standard Deviations
    X_std = val_X.std(axis=0)
    y_std = errors.std()

    # Correlation Coefficient (Point-Biserial)
    # Add epsilon to avoid division by zero
    correlations = covariance / (X_std * y_std + 1e-9)

    # Pair with names and sort by absolute correlation
    feat_corr = list(zip(feature_names, correlations))
    feat_corr.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Prediction Error:")
    for name, corr_val in feat_corr[:10]:
        print(f"  {name}: {corr_val:.6f}")

    # ==========================================
    # 6. Conditional Submission
    # ==========================================
    THRESHOLD = 0.9622416666666667

    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric ({final_acc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Create Test Loader
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_X, dtype=torch.float32)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Call the original predict function
        original_predict_func(model, test_loader, Config)

    else:
        print(
            f"\nValidation metric ({final_acc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
