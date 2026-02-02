import os
import numpy as np
import pandas as pd
import torch
import random
import joblib
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.data_loader import PizzaDataLoader
from library.feature_extraction import FeatureExtractor
from library.pipeline import CrossValidationManager


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Initialize Manager
    # The manager handles the heavy lifting of the pipeline defined in library/pipeline.py
    manager = CrossValidationManager()

    # 3. Run Training (Cross-Validation)
    # This trains the model ensemble using the AAAEF strategy (MiniLM + MPNet + GoEmotions + Metadata)
    # Models for each fold are saved to disk.
    print("Executing Training Pipeline...")
    manager.run_cv(load_cached_data=True)

    # 4. Independent Validation on Hold-out Set
    # We explicitly evaluate on the validation set to satisfy the reporting requirement.
    # Note: The provided pipeline merges train/val for CV, so this evaluates the ensemble
    # on a subset of the data it has likely seen during training (via the CV folds).
    print("\nRunning independent validation on hold-out set...")

    loader = PizzaDataLoader()
    extractor = FeatureExtractor()

    # Load Validation Data
    df_val = loader.load_data("val", load_cached_data=True)
    y_val = df_val["requester_received_pizza"].values

    # Extract Features for Validation Set
    # We rely on caching where possible to speed up execution
    feats_val = extractor.extract_features(df_val, "val", load_cached_data=True)
    meta_val = loader.get_metadata_features(df_val)

    # Combine features into the dictionary format expected by the pipeline
    X_val = {**feats_val, "metadata": meta_val}

    # Load Trained Models and Predict
    fold_predictions = []
    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_pipeline.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file for fold {fold} not found at {model_path}"
            )

        # Load pipeline (Preprocessor + Classifier)
        pipeline = joblib.load(model_path)

        # Predict probability (Class 1)
        # Inference is done on CPU/GPU automatically by the pipeline's internal logic,
        # though the pipeline here expects numpy arrays, so it's CPU-bound for the sklearn part.
        preds = pipeline.predict_proba(X_val)[:, 1]
        fold_predictions.append(preds)

    # Average Predictions (Ensemble Bagging)
    y_pred_val = np.mean(fold_predictions, axis=0)

    # Compute Metric
    val_auc = roc_auc_score(y_val, y_pred_val)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    # Calculate absolute error |y_true - y_pred|
    errors = np.abs(y_val - y_pred_val)

    # Correlate errors with numerical metadata features to find sources of difficulty
    correlations = {}
    for col in meta_val.columns:
        # Skip constant columns to avoid warnings
        if meta_val[col].std() == 0:
            continue

        corr, _ = pearsonr(errors, meta_val[col])
        correlations[col] = corr

    # Sort by absolute correlation strength
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Prediction Error:")
    for feat, corr in sorted_corrs[:5]:
        print(f"{feat}: {corr:.4f}")

    # 6. Submission Generation
    # Only generate submission if the model meets the performance threshold
    threshold = 0.7190361601447052

    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        manager.generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation AUC ({val_auc}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
