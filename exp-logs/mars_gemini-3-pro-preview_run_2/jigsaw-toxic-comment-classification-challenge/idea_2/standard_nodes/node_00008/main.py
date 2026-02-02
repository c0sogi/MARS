import os
import sys
import gc
import torch
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

    # 4. Branch B: Deep Learning (Transformer Ensemble)
    print("\n=== Branch B: Transformer Ensemble Execution ===")

    val_preds_transformers_list = []
    test_preds_transformers_list = []

    for model_name in Config.MODEL_NAMES:
        print(f"\n--- Processing Model: {model_name} ---")

        # Data Preparation
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, model_name
        )

        # Model Training
        save_filename = f"best_model_{model_name}.bin"
        transformer_trainer = TransformerTrainer(model_name, save_filename)
        transformer_trainer.train(train_loader, val_loader)

        # Inference
        print(f"Generating {model_name} predictions for Validation set...")
        val_preds = transformer_trainer.predict(val_loader)
        val_preds_transformers_list.append(val_preds)

        print(f"Generating {model_name} predictions for Test set...")
        test_preds = transformer_trainer.predict(test_loader)
        test_preds_transformers_list.append(test_preds)

        # Cleanup to free GPU memory
        del transformer_trainer
        torch.cuda.empty_cache()
        gc.collect()

    # Average the transformer predictions (Semantic Ensemble)
    print("\nAveraging Transformer predictions...")
    val_preds_deep = np.mean(val_preds_transformers_list, axis=0)
    test_preds_deep = np.mean(test_preds_transformers_list, axis=0)

    # 5. Ensemble Aggregation (Optimized Blending)
    print("\n=== Ensemble Aggregation (Optimized Blending) ===")

    # Grid search for optimal linear weight (Cite solution_lesson_node_00007)
    best_score = -1
    best_alpha = 0.0
    y_val_true = val_df[Config.LABEL_COLS].values

    print("Performing grid search for optimal blending weight...")
    # Search alpha (weight for linear model) from 0.0 to 0.5
    # We assume deep learning is stronger, so linear weight shouldn't exceed 0.5
    for alpha in np.linspace(0.0, 0.5, 11):
        blended_preds = (alpha * val_preds_linear) + ((1 - alpha) * val_preds_deep)
        score = compute_metric(y_val_true, blended_preds)
        print(f"Alpha: {alpha:.2f}, Score: {score:.5f}")

        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(f"\nBest Alpha (Linear Weight): {best_alpha}")
    print(f"Best Validation Score: {best_score}")

    # Final Weighted Average
    val_preds_ensemble = (best_alpha * val_preds_linear) + (
        (1 - best_alpha) * val_preds_deep
    )
    test_preds_ensemble = (best_alpha * test_preds_linear) + (
        (1 - best_alpha) * test_preds_deep
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
    threshold = 0.9926490739470294

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
