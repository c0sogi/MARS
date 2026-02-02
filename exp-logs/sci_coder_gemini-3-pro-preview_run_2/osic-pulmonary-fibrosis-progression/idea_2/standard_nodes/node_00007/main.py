import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import warnings

# Import provided library components
from library.config import Config
from library.feature_extractor import generate_features
from library.data_processor import DataProcessor
from library.model import DualElasticNet, generate_submission
from library.metrics import laplace_log_likelihood

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing Lung Function Decline Prediction Pipeline...")

    # 2. Feature Extraction
    # Extract CNN features from CT scans or load from cache if available
    print("\n[Step 1/6] Generating/Loading Image Features...")
    train_feats, val_feats, test_feats = generate_features(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 3. Data Processing
    # Merge tabular data with image features and generate interaction terms
    print("\n[Step 2/6] Processing Tabular Data...")
    processor = DataProcessor()
    (X_train, y_train), (X_val, y_val), (X_test, df_test) = processor.process_data(
        train_feats, val_feats, test_feats, load_cached_data=True
    )

    # 4. Model Training
    # Train the Dual Elastic Net (FVC Predictor + Uncertainty Estimator)
    print("\n[Step 3/6] Training Model...")
    model = DualElasticNet()
    model.fit(X_train, y_train, X_val, y_val)

    # 5. Validation Assessment
    print("\n[Step 4/6] Validating Model...")
    # Generate predictions for the validation set
    val_pred_fvc, val_pred_sigma = model.predict(X_val)

    # Compute the official metric
    final_metric = laplace_log_likelihood(y_val, val_pred_fvc, val_pred_sigma)

    # Print the metric in the required format (Full Precision)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n[Step 5/6] Performing Failure Analysis...")
    # Calculate absolute errors
    val_errors = np.abs(y_val - val_pred_fvc)

    # Load validation metadata for interpretable analysis
    # We sort by Patient and Weeks to ensure alignment with the processed y_val
    df_val_analysis = pd.read_csv(Config.VAL_META_PATH)
    df_val_analysis = df_val_analysis.sort_values(["Patient", "Weeks"])

    if len(df_val_analysis) == len(val_errors):
        df_val_analysis["Absolute_Error"] = val_errors

        # Correlate error with key clinical features
        # 'FVC' here refers to the ground truth FVC
        analysis_cols = ["Age", "Percent", "Weeks", "FVC"]
        correlations = df_val_analysis[analysis_cols].corrwith(
            df_val_analysis["Absolute_Error"]
        )

        print("Correlation between Absolute Prediction Error and Input Features:")
        print(correlations)
    else:
        print("Warning: Metadata length mismatch. Skipping correlation analysis.")

    # 7. Submission Generation
    print("\n[Step 6/6] Generating Submission...")
    THRESHOLD = -7.158702679895534

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission file."
        )
        generate_submission(model, X_test, df_test)
    else:
        print(
            f"Metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
