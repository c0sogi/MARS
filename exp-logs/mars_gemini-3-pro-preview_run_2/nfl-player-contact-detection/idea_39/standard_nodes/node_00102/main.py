import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processing import get_data
from library.model import WDPIRVModel, train_model, predict
from library.inference import predict_and_submit

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)

    # Limit epochs to ensure execution finishes within the time limit (Fast Baseline)
    Config.EPOCHS = 10

    # 2. Data Loading
    # Use debug=False to train on the full dataset for maximum performance.
    # load_cached_data=True allows skipping processing if artifacts exist.
    train_dataset, val_dataset, test_dataset, feature_dim = get_data(
        load_cached_data=True, debug=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization & Training
    print(f"Initializing WDPIRVModel with input dimension: {feature_dim}")
    model = WDPIRVModel(input_dim=feature_dim)

    print("Starting training...")
    best_threshold = train_model(model, train_loader, val_loader, Config.DEVICE)
    print(f"Optimal Threshold found: {best_threshold}")

    # 4. Validation Assessment
    print("Performing final validation evaluation...")
    # Load the best model weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    # Generate probabilities on validation set
    val_probs = predict(model, val_loader, Config.DEVICE)

    # Get ground truth labels (convert from tensor to numpy)
    y_val = val_dataset.labels.numpy()

    # Apply threshold
    val_preds = (val_probs > best_threshold).astype(int)

    # Compute Metric
    val_mcc = compute_mcc(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_mcc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_val - val_probs)

    # Retrieve feature names for interpretation
    try:
        # Attempt to read schema from cached parquet file
        feature_names = pd.read_parquet(Config.CACHE_VAL_PARQUET).columns.tolist()
    except Exception:
        # Fallback if cache reading fails
        feature_names = [f"feature_{i}" for i in range(feature_dim)]

    # Prepare data for correlation analysis
    # Convert features to numpy
    X_val = val_dataset.features.numpy()

    # To optimize memory/time, subsample if dataset is extremely large (though 800k is manageable)
    if len(errors) > 200000:
        idx = np.random.choice(len(errors), 200000, replace=False)
        X_analysis = X_val[idx]
        errors_analysis = errors[idx]
    else:
        X_analysis = X_val
        errors_analysis = errors

    # Create DataFrame for easy correlation computation
    df_analysis = pd.DataFrame(X_analysis, columns=feature_names)
    df_analysis["error_magnitude"] = errors_analysis

    # Compute correlation of features with error magnitude
    print("Computing feature correlations with error magnitude...")
    correlations = df_analysis.corrwith(df_analysis["error_magnitude"]).drop(
        "error_magnitude"
    )

    # Display top 10 features associated with errors
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)
    print("Top 10 Features correlated with Model Error:")
    print(top_correlations)

    # 6. Conditional Submission
    TARGET_SCORE = 0.6634847318478787

    if val_mcc > TARGET_SCORE:
        print(
            f"\nValidation MCC ({val_mcc}) exceeds target ({TARGET_SCORE}). Generating submission..."
        )
        predict_and_submit(
            model=model,
            test_loader=test_loader,
            device=Config.DEVICE,
            threshold=best_threshold,
            save_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation MCC ({val_mcc}) did not exceed target ({TARGET_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
