import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Import provided library modules
import library.config as config
import library.dataset as dataset
import library.trainer as trainer
import library.inference as inference


def main():
    start_time = time.time()
    print("Starting pipeline execution...")

    # ==========================================
    # 1. Data Loading and Feature Engineering
    # ==========================================
    print("\n[Step 1] Loading Data...")
    # Load training and validation data
    # This triggers feature extraction if cache is not found
    X_train, y_train, X_val, y_val = dataset.get_train_data(load_cached_data=True)

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")

    # Load test data
    X_test, test_ids = dataset.get_test_data(load_cached_data=True)
    print(f"Test Data Shape: {X_test.shape}")

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\n[Step 2] Training Models...")
    # Run Cross-Validation on the training set
    # This returns a list of trained LightGBM models (one per fold)
    # We use the default parameters from config, which are tuned for MAE
    models = trainer.run_cross_validation(
        X_train, y_train, num_folds=config.NUM_FOLDS, verbose_eval=100
    )

    # ==========================================
    # 3. Validation Evaluation
    # ==========================================
    print("\n[Step 3] Evaluating on Hold-out Validation Set...")
    # Generate predictions on the hold-out validation set using the ensemble
    val_preds = trainer.predict(models, X_val)

    # Calculate Mean Absolute Error
    val_mae = mean_absolute_error(y_val, val_preds)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_mae}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n[Step 4] Performing Failure Analysis...")
    # Calculate absolute errors
    abs_errors = np.abs(y_val - val_preds)

    # Create a DataFrame to analyze correlations between features and error
    analysis_df = X_val.copy()
    analysis_df["abs_error"] = abs_errors

    # Compute correlation of features with the absolute error
    # We drop the error column itself from the correlation calculation features
    correlations = analysis_df.corrwith(analysis_df["abs_error"]).drop("abs_error")

    # Sort by absolute correlation strength (descending)
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 features correlated with prediction error:")
    for feature, corr_val in top_correlations.items():
        # Retrieve original sign
        orig_corr = correlations[feature]
        print(f"  {feature}: {orig_corr:.4f}")

    # ==========================================
    # 5. Inference and Submission
    # ==========================================
    print("\n[Step 5] Generating Submission...")
    # Generate predictions for the test set and save to CSV
    inference.predict_and_submit(models, X_test, test_ids)

    elapsed_time = time.time() - start_time
    print(f"\nPipeline completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
