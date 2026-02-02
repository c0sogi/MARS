import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.feature_extraction import FeatureExtractor
from library.dimensionality_reduction import IndependentPCA
from library.models import get_base_models, get_meta_learner
from library.train_eval import CrossValidator, FinalTrainer, merge_data_dicts


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    print("Initializing workflow...")

    # 2. Feature Extraction
    # Extracts features from images using Swin, EfficientNet, and DINOv2.
    # Handles caching to save time on re-runs.
    extractor = FeatureExtractor()
    train_raw, val_raw, test_raw = extractor.extract_and_cache_features(
        load_cached_data=True
    )

    # 3. Hold-out Validation Logic
    print("\n=== Starting Hold-out Validation ===")

    # 3.1 Generate OOF predictions on the Training Set
    # These are required to train the Meta-Learner without leakage.
    # CrossValidator handles the internal 5-fold split and IndependentPCA fitting per fold.
    cv_validator = CrossValidator(n_folds=Config.N_FOLDS, seed=Config.SEED)
    oof_train_df, train_targets = cv_validator.run_cv(train_raw, load_cached_data=True)

    # 3.2 Train the Meta-Learner (Level 2)
    # The meta-learner learns to combine base model predictions based on OOF performance.
    meta_learner = get_meta_learner()
    meta_learner.fit(oof_train_df.values, train_targets)

    # 3.3 Train Base Models (Level 1) on the Full Training Set
    # We must fit the PCA transformation on the training set and apply it to validation.
    pca_processor = IndependentPCA(
        variance_threshold=Config.PCA_VARIANCE, seed=Config.SEED
    )
    pca_processor.fit(train_raw)

    X_train_pca = pca_processor.transform(train_raw)
    X_val_pca = pca_processor.transform(val_raw)
    y_train = train_raw["targets"]
    y_val = val_raw["targets"]

    base_models = get_base_models()
    val_base_preds = np.zeros((len(y_val), len(base_models)))

    # Train each base model and predict on validation set
    for i, (name, model) in enumerate(base_models.items()):
        # print(f"Training {name} on full training set...")
        model.fit(X_train_pca, y_train)
        val_base_preds[:, i] = model.predict(X_val_pca)

    # 3.4 Generate Ensemble Predictions
    # Feed base model predictions into the meta-learner
    val_final_preds = meta_learner.predict(val_base_preds)

    # Clip predictions to the valid range [1, 100]
    val_final_preds = np.clip(val_final_preds, 1.0, 100.0)

    # 3.5 Compute and Print Final Validation Metric
    val_rmse = np.sqrt(mean_squared_error(y_val, val_final_preds))
    print(f"Final Validation Metric: {val_rmse:.16f}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - val_final_preds)

    # Metadata feature names (matching library/data_loader.py)
    meta_cols = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    val_meta_features = val_raw["metadata"]

    print("Correlation between Error Magnitude and Metadata Features:")
    for i, col_name in enumerate(meta_cols):
        # Ensure we don't go out of bounds if metadata columns change
        if i < val_meta_features.shape[1]:
            feature_values = val_meta_features[:, i]
            # Calculate Pearson correlation
            # Handle constant features to avoid division by zero in correlation
            if np.std(feature_values) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(errors, feature_values)[0, 1]
            print(f"  {col_name}: {corr:.4f}")

    # 5. Submission Generation
    # Only generate submission if the model meets the performance threshold.
    threshold = 17.429365583625966

    if val_rmse < threshold:
        print(
            f"\nValidation RMSE ({val_rmse:.6f}) < Threshold ({threshold:.6f}). Proceeding to Submission..."
        )

        # Merge Training and Validation data to maximize performance
        full_train_raw = merge_data_dicts(train_raw, val_raw)

        # We need to regenerate OOF predictions for the merged dataset to train the final meta-learner.
        # The IDs will be different (train+val), so this will trigger a new CV run (or load if cached).
        print("Generating OOF predictions for merged dataset...")
        cv_final = CrossValidator(n_folds=Config.N_FOLDS, seed=Config.SEED)
        oof_full_df, full_targets = cv_final.run_cv(
            full_train_raw, load_cached_data=True
        )

        # Use the FinalTrainer to retrain everything on full data and predict test set
        trainer = FinalTrainer(seed=Config.SEED)
        trainer.train_and_predict(full_train_raw, test_raw, oof_full_df, full_targets)

    else:
        print(
            f"\nValidation RMSE ({val_rmse:.6f}) >= Threshold ({threshold:.6f}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
