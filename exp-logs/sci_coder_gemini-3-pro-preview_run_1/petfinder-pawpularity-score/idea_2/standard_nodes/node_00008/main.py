import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_rmse,
)
from library.dataset import get_dataloaders
from library.model import PawpularitySwinModel
from library.engine import extract_features


def run_failure_analysis(y_true, y_pred, features_array):
    """
    Performs failure analysis on the validation set using predictions from the linear model.
    """
    # Calculate residuals (absolute error)
    errors = np.abs(y_true - y_pred)

    # Create a DataFrame for correlation analysis
    feature_names = Config.feature_cols
    # features_array is numpy array of shape (N, 12)
    analysis_df = pd.DataFrame(features_array, columns=feature_names)
    analysis_df["Error"] = errors

    # Calculate correlation
    correlations = analysis_df.corr()["Error"].drop("Error")

    print("\n=== Failure Analysis: Correlation with Error Magnitude ===")
    print(correlations.sort_values(ascending=False).to_string())
    print("========================================================\n")


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    print(f"Device: {device}")
    print(f"Model: {Config.model_name} (Backbone for Linear Probing)")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 3. Model Initialization (Feature Extractor)
    print("Initializing backbone model...")
    model = PawpularitySwinModel(pretrained=True)
    model.to(device)
    model.eval()

    # 4. Feature Extraction
    # Cite {solution_lesson_node_00001}: Decouple feature extraction from learning to accelerate baseline validation.
    print("Extracting features (Linear Probing Strategy)...")

    train_img, train_meta, train_y, _ = extract_features(model, train_loader, device)
    val_img, val_meta, val_y, _ = extract_features(model, val_loader, device)
    test_img, test_meta, _, test_ids = extract_features(model, test_loader, device)

    # Construct Feature Matrices
    # Concatenate Image Embeddings (e.g., 768 dim) and Metadata (12 dim)
    X_train = np.hstack([train_img, train_meta])
    X_val = np.hstack([val_img, val_meta])
    X_test = np.hstack([test_img, test_meta])

    # Cite {solution_lesson_node_00007}: Feature Normalization and Backbone Scaling in Multimodal Linear Probing
    # Normalize features to handle scale discrepancy between embeddings and binary flags
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Scale targets back to [0, 100] for Ridge Regression
    y_train = train_y * 100.0
    y_val = val_y * 100.0

    # 5. Training Linear Head
    # Cite {solution_lesson_node_00005}: Prioritize Linear Probing over Fine-Tuning for noisy, small-scale regression.
    # Cite {solution_lesson_node_00007}: Increase regularization strength for normalized features.
    print("Training Ridge Regression Head...")
    clf = Ridge(alpha=20.0, random_state=Config.seed)
    clf.fit(X_train, y_train)

    # 6. Evaluation
    print("Calculating final validation metric...")
    val_preds = clf.predict(X_val)
    # Clip predictions to valid range
    val_preds = np.clip(val_preds, 0, 100)

    final_rmse = calculate_rmse(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis
    run_failure_analysis(y_val, val_preds, val_meta)

    # 7. Submission
    THRESHOLD = 18.5835835849123

    if final_rmse < THRESHOLD:
        print(
            f"Validation RMSE ({final_rmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        test_preds = clf.predict(X_test)
        test_preds = np.clip(test_preds, 0, 100)

        submission_df = pd.DataFrame({"Id": test_ids, "Pawpularity": test_preds})
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"Validation RMSE ({final_rmse}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
