import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.feature_extractor import FeatureEngine
from library.preprocessing import StreamProcessor
from library.models import Level1Estimators, StackingMetaLearner, generate_submission


def main():
    # ==========================================
    # 1. Setup and Initialization
    # ==========================================
    set_seed(Config.SEED)
    print("Initializing Quad-Stream Ensemble Pipeline...")

    # ==========================================
    # 2. Feature Extraction
    # ==========================================
    # Extract features from Swin, EffNet, DINOv2, and CLIP.
    # Uses caching to speed up re-runs.
    feature_engine = FeatureEngine()
    raw_feats, meta_feats, targets, ids = feature_engine.extract_features(
        load_cached_data=True
    )

    # ==========================================
    # 3. Preprocessing (PCA & Scaling)
    # ==========================================
    # Compress image features using PCA (fit on train) and scale metadata.
    stream_processor = StreamProcessor()
    X_train, X_val, X_test = stream_processor.process_features(
        raw_feats, meta_feats, load_cached_data=True
    )

    y_train = targets["train"]
    y_val = targets["val"]

    print(
        f"Feature Matrices Ready: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )

    # ==========================================
    # 4. Hold-Out Validation
    # ==========================================
    print("\n=== Starting Hold-out Validation ===")

    # Initialize model classes
    l1_estimators = Level1Estimators()
    meta_learner = StackingMetaLearner()

    # Step A: Generate Out-of-Fold (OOF) predictions on the Training set
    # These are used to train the Level 2 Meta-Learner without leakage.
    print("Generating OOF predictions on Training set...")
    oof_train = l1_estimators.get_oof_predictions(X_train, y_train)

    # Step B: Train the Level 2 Meta-Learner on Training OOFs
    print("Training Meta-Learner on Training set OOF...")
    meta_learner.fit(oof_train, y_train)

    # Step C: Retrain Level 1 models on the full Training set
    # This prepares them to make predictions on the Validation set.
    print("Retraining Level 1 models on Training set...")
    l1_estimators.fit_all(X_train, y_train)

    # Step D: Inference on Validation set
    print("Predicting on Validation set...")
    base_preds_val = l1_estimators.predict(X_val)
    final_preds_val = meta_learner.predict(base_preds_val)

    # Clip predictions to valid range [1, 100]
    final_preds_val = np.clip(final_preds_val, 1.0, 100.0)

    # Step E: Calculate and Print Metric
    val_mse = mean_squared_error(y_val, final_preds_val)
    val_rmse = np.sqrt(val_mse)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_rmse}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - final_preds_val)

    # Get binary metadata flags for validation set
    val_meta_flags = meta_feats["val"]  # Shape (N, 12)
    meta_cols = Config.METADATA_COLS

    print("Correlation between Error Magnitude and Metadata Features:")
    correlations = []

    for i, col_name in enumerate(meta_cols):
        # Calculate Pearson correlation
        # Handle constant columns (std=0) to avoid runtime warnings
        if np.std(val_meta_flags[:, i]) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, val_meta_flags[:, i])

        correlations.append((col_name, corr))
        print(f"  {col_name}: {corr:.4f}")

    # Identify the feature most associated with error
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(
        f"Top factor associated with error: {correlations[0][0]} ({correlations[0][1]:.4f})"
    )

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 17.499122532793635

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation metric ({val_rmse}) is better than threshold ({THRESHOLD}). Proceeding to submission..."
        )

        # Strategy: Retrain on combined Train + Validation data for maximum performance
        print("Consolidating Train and Validation sets...")
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Re-initialize models to clear previous state
        l1_estimators_full = Level1Estimators()
        meta_learner_full = StackingMetaLearner()

        # 1. Get OOF on Full Data (Train + Val)
        print("Generating OOF predictions on Full Data...")
        oof_full = l1_estimators_full.get_oof_predictions(X_full, y_full)

        # 2. Train Meta-Learner on Full OOF
        print("Training Meta-Learner on Full OOF...")
        meta_learner_full.fit(oof_full, y_full)

        # 3. Retrain Level 1 models on Full Data
        print("Retraining Level 1 models on Full Data...")
        l1_estimators_full.fit_all(X_full, y_full)

        # 4. Predict on Test Set
        print("Predicting on Test set...")
        base_preds_test = l1_estimators_full.predict(X_test)
        final_preds_test = meta_learner_full.predict(base_preds_test)

        # 5. Generate Submission File
        test_ids = ids["test"]
        generate_submission(test_ids, final_preds_test)

    else:
        print(
            f"\nValidation metric ({val_rmse}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
