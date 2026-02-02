import sys
import os
import numpy as np
import pandas as pd
import warnings

# Ensure the library module can be imported
sys.path.append(os.getcwd())

from library import config
from library import data_loader
from library import feature_engineering
from library import model_trainer
from library import inference


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    # --- 1. Setup ---
    warnings.filterwarnings("ignore")
    set_seeds(config.SEED)
    print("Initializing pipeline...")

    # --- 2. Data Loading ---
    # Loads the full dataset (Train + Val combined) and Test set
    # Utilizes caching if available to speed up subsequent runs
    df_train, df_test = data_loader.load_dataset(load_cached_data=True)

    # --- 3. Feature Engineering ---
    # Transforms data using the defined pipeline (Geometric, PCA, Group Stats)
    df_train_proc, df_test_proc = feature_engineering.process_data(
        df_train, df_test, load_cached_data=True
    )

    # --- 4. Model Training (Stratified CV) ---
    print("Starting Stratified Cross-Validation...")
    trainer = model_trainer.EnsembleTrainer(params=config.XGB_PARAMS)

    # run_stratified_cv returns the trained models and the overall OOF accuracy
    models, overall_acc = trainer.run_stratified_cv(df_train_proc)

    # --- 5. Validation Reporting ---
    # Print the required metric in the specified format
    print(f"Final Validation Metric: {overall_acc}")

    # --- 6. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # We analyze the correlation between features and the error magnitude on the training set (OOF)

    # Get the integer encoded targets used during training
    y_true_encoded = trainer.le.transform(df_train_proc[config.TARGET_COL])

    # Retrieve the probability assigned to the correct class for each sample
    # trainer.oof_preds is shape (N_samples, N_classes)
    row_indices = np.arange(len(y_true_encoded))
    prob_true_class = trainer.oof_preds[row_indices, y_true_encoded]

    # Error Magnitude: 0.0 means perfect confidence, 1.0 means completely wrong
    error_magnitude = 1.0 - prob_true_class

    # Prepare dataframe for correlation analysis (Numerical features only)
    analysis_df = df_train_proc.select_dtypes(include=[np.number]).copy()

    # Drop non-feature columns if present
    cols_to_drop = [
        c for c in [config.TARGET_COL, config.ID_COL] if c in analysis_df.columns
    ]
    if cols_to_drop:
        analysis_df.drop(columns=cols_to_drop, inplace=True)

    # Add Error Magnitude to the dataframe
    analysis_df["Error_Magnitude"] = error_magnitude

    # Compute correlations
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")

    # Display top 5 features most correlated with error (absolute value)
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)
    print("Top 5 features correlated with Error Magnitude:")
    print(correlations.loc[top_correlations.index])

    # --- 7. Submission ---
    THRESHOLD = 0.9619111111111112

    if overall_acc > THRESHOLD:
        print(
            f"\nValidation metric {overall_acc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Initialize Inference Engine
        engine = inference.InferenceEngine(
            models=models, label_encoder=trainer.le, feature_names=trainer.feature_names
        )

        # Generate averaged soft probabilities
        avg_preds = engine.predict_ensemble(df_test_proc)

        # Save final submission file
        engine.save_submission(df_test_proc, avg_preds)

    else:
        print(
            f"\nValidation metric {overall_acc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
