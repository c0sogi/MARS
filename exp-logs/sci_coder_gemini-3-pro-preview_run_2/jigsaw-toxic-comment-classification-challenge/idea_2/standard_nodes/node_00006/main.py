import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_metric
from library.data_loader import load_datasets
from library.tfidf_processor import TfidfVectorizationPipeline
from library.linear_model import LinearEnsembleTrainer
from library.transformer_data import create_dataloaders
from library.transformer_trainer import TransformerTrainer


def main():
    # 1. Initialization
    print("=== Initialization ===")
    Config.setup()
    set_seed(Config.SEED)

    # 2. Data Loading
    print("\n=== Loading Data ===")
    # Load datasets with caching enabled
    train_df, val_df, test_df = load_datasets(load_cached_data=True)

    # 3. Branch A: Linear Ensemble (TF-IDF + Logistic Regression)
    print("\n=== Branch A: Linear Ensemble Execution ===")

    # Feature Extraction
    tfidf_pipeline = TfidfVectorizationPipeline()
    X_train, X_val, X_test = tfidf_pipeline.run(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Model Training & Inference
    linear_trainer = LinearEnsembleTrainer()
    # Pass full dataframes for y (labels) as the trainer handles column extraction
    val_preds_linear, test_preds_linear = linear_trainer.train_and_predict(
        X_train, train_df, X_val, val_df, X_test
    )

    # 4. Branch B: Deep Learning (Transformer)
    print("\n=== Branch B: Transformer Execution ===")

    # Data Preparation
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df
    )

    # Model Training
    transformer_trainer = TransformerTrainer()
    transformer_trainer.train(train_loader, val_loader)

    # Inference (Using the best saved model)
    print("Generating Transformer predictions for Validation set...")
    val_preds_transformer = transformer_trainer.predict(val_loader)

    print("Generating Transformer predictions for Test set...")
    test_preds_transformer = transformer_trainer.predict(test_loader)

    # 5. Ensemble Aggregation
    print("\n=== Ensemble Aggregation ===")
    w_lin = Config.ENSEMBLE_WEIGHTS["linear"]
    w_trans = Config.ENSEMBLE_WEIGHTS["transformer"]

    print(f"Weights -> Linear: {w_lin}, Transformer: {w_trans}")

    # Weighted Average for Validation
    val_preds_ensemble = (w_lin * val_preds_linear) + (w_trans * val_preds_transformer)

    # Weighted Average for Test
    test_preds_ensemble = (w_lin * test_preds_linear) + (
        w_trans * test_preds_transformer
    )

    # 6. Validation & Metrics
    print("\n=== Final Validation ===")
    y_val_true = val_df[Config.LABEL_COLS].values

    # Compute Final Metric
    final_metric = compute_metric(y_val_true, val_preds_ensemble)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample across all labels
    # Shape: (n_samples, n_labels) -> (n_samples,)
    mae_per_sample = np.mean(np.abs(y_val_true - val_preds_ensemble), axis=1)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error": mae_per_sample,
            "char_length": val_df[Config.TEXT_COL].apply(len).values,
        }
    )

    # Calculate correlation
    error_len_corr = analysis_df["error"].corr(analysis_df["char_length"])
    print(f"Correlation between Error Magnitude and Character Length: {error_len_corr}")

    # 8. Submission Generation
    threshold = 0.9837638458604258

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission file..."
        )

        # Create submission DataFrame
        submission_df = pd.DataFrame()
        submission_df[Config.ID_COL] = test_df[Config.ID_COL]

        # Add predicted probabilities for each class
        for i, label in enumerate(Config.LABEL_COLS):
            submission_df[label] = test_preds_ensemble[:, i]

        # Save
        submission_path = Config.SUBMISSION_CSV
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
