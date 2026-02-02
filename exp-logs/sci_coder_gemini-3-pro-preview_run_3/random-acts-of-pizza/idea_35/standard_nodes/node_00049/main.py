import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library modules
from library.config import Config
from library.utils import Timer, set_seed
from library.data_manager import get_processed_data
from library.feature_engine import FeaturePipeline
from library.training_engine import TrainingEngine
from library.model_definitions import ModelZoo


def run_failure_analysis(y_true, y_pred, feature_dict):
    """
    Analyzes the correlation between prediction errors and metadata features
    on the validation set.
    """
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - y_pred)

    # Retrieve metadata features for validation set (standardized)
    X_meta = feature_dict["X_val_metadata"]

    # Use column names from Config
    meta_cols = Config.METADATA_COLS

    correlations = []
    # Calculate correlation for each metadata feature
    for i, col_name in enumerate(meta_cols):
        if i < X_meta.shape[1]:
            # Handle constant features to avoid warning
            if np.std(X_meta[:, i]) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(errors, X_meta[:, i])[0, 1]
            correlations.append((col_name, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error and Metadata Features:")
    for name, corr in correlations:
        print(f"  {name}: {corr:.4f}")


def main():
    # Ensure reproducibility
    set_seed(Config.RANDOM_STATE)

    with Timer("Total Runtime"):

        # ---------------------------------------------------------
        # 1. Data Loading & Feature Engineering
        # ---------------------------------------------------------
        print("\n[Step 1] Loading Data and Generating Features...")
        # Load data (utilizing cache if available)
        train_df, val_df, test_df = get_processed_data(load_cached_data=True)

        # Generate all feature views
        fp = FeaturePipeline()
        feature_dict = fp.process_all(train_df, val_df, test_df, load_cached_data=True)

        # ---------------------------------------------------------
        # 2. Level 1: Cross-Validation (OOF Generation)
        # ---------------------------------------------------------
        print("\n[Step 2] Running Cross-Validation (Level 1)...")
        te = TrainingEngine()
        zoo = te.model_zoo

        # Run CV to get OOF predictions for Meta-Learner training
        oof_preds = te.run_cv_and_generate_oof(feature_dict, load_cached_data=True)

        # ---------------------------------------------------------
        # 3. Level 2: Train Meta-Learner
        # ---------------------------------------------------------
        print("\n[Step 3] Training Meta-Learner...")
        y_train = feature_dict["y_train"]
        meta_model = te.train_meta_learner(oof_preds, y_train)

        # ---------------------------------------------------------
        # 4. Validation Assessment
        # ---------------------------------------------------------
        print("\n[Step 4] Validating on Hold-Out Set...")
        # To evaluate on the hold-out set, we train base models on the Training set
        # and predict on the Validation set.

        val_preds_base = []

        for model_name in te.models_to_train:
            model = zoo.get_model(model_name)

            # Get training data
            X_train_model, y_train_model = zoo.format_data(
                model_name, feature_dict, split="train"
            )

            # Train model on full training set
            # Note: For XGBoost, we disable early stopping here (by not providing eval_set)
            # to avoid leaking the validation set into the model training process during evaluation.
            zoo.train_model(model, X_train_model, y_train_model, verbose=False)

            # Predict on Validation set
            X_val_model, _ = zoo.format_data(model_name, feature_dict, split="val")
            preds = zoo.predict_proba(model, X_val_model)
            val_preds_base.append(preds)

        # Stack Level 1 predictions
        X_meta_val = np.column_stack(val_preds_base)

        # Level 2 Prediction
        y_val_pred = meta_model.predict_proba(X_meta_val)[:, 1]
        y_val_true = feature_dict["y_val"]

        # Compute and Print Metric
        val_auc = roc_auc_score(y_val_true, y_val_pred)
        print(f"Final Validation Metric: {val_auc}")

        # ---------------------------------------------------------
        # 5. Failure Analysis
        # ---------------------------------------------------------
        run_failure_analysis(y_val_true, y_val_pred, feature_dict)

        # ---------------------------------------------------------
        # 6. Submission Generation
        # ---------------------------------------------------------
        threshold = 0.7138293787137718

        if val_auc > threshold:
            print(
                f"\n[Step 5] Metric ({val_auc}) > Threshold ({threshold}). Generating Submission..."
            )

            # Retrain base models on Train + Val for maximum performance
            # (TrainingEngine handles XGBoost early stopping using Val split correctly)
            final_models = te.retrain_base_models(feature_dict)

            # Generate predictions on Test set and save submission
            te.generate_submission(final_models, meta_model, feature_dict)

        else:
            print(
                f"\n[Step 5] Metric ({val_auc}) <= Threshold ({threshold}). Submission skipped."
            )


if __name__ == "__main__":
    main()
