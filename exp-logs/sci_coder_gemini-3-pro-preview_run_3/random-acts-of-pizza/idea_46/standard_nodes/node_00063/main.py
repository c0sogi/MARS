import os
import sys
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, log_info, compute_auc, timer
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.ensemble_trainer import EnsembleTrainer


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.RANDOM_SEED)
    log_info("Starting Hex-View Stacking Ensemble Pipeline...")

    # 2. Data Loading
    # Load processed dataframes for all splits
    loader = DataLoader()
    train_df = loader.load_dataset("train", load_from_cache=True)
    val_df = loader.load_dataset("val", load_from_cache=True)
    test_df = loader.load_dataset("test", load_from_cache=True)

    # Extract Targets and IDs
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values
    test_ids = test_df[Config.ID_COL].values

    # 3. Feature Engineering
    # Generate all four views: Lexical, Behavioral, Semantic, Metadata
    engineer = FeatureEngineer()

    # Lexical View (TF-IDF Text)
    X_lex_train, X_lex_val, X_lex_test = engineer.get_lexical_view(
        train_df, val_df, test_df
    )

    # Behavioral View (TF-IDF Subreddits)
    X_beh_train, X_beh_val, X_beh_test = engineer.get_behavioral_view(
        train_df, val_df, test_df
    )

    # Semantic View (Embeddings)
    X_sem_train, X_sem_val, X_sem_test = engineer.get_semantic_view(
        train_df, val_df, test_df
    )

    # Metadata View (Dense Features)
    X_meta_train, X_meta_val, X_meta_test = engineer.get_metadata_view(
        train_df, val_df, test_df
    )

    # Organize features into a dictionary for the Trainer
    feature_data = {
        "lexical": {"train": X_lex_train, "val": X_lex_val, "test": X_lex_test},
        "behavioral": {"train": X_beh_train, "val": X_beh_val, "test": X_beh_test},
        "semantic": {"train": X_sem_train, "val": X_sem_val, "test": X_sem_test},
        "metadata": {"train": X_meta_train, "val": X_meta_val, "test": X_meta_test},
    }

    # 4. Ensemble Training
    trainer = EnsembleTrainer()

    # Step 4a: Generate OOF Predictions (Level 1)
    # This uses StratifiedKFold on the training set
    oof_preds = trainer.get_oof_predictions(feature_data, y_train)

    # Step 4b: Train Meta-Learner (Level 2)
    # Trains Logistic Regression on OOF predictions
    trainer.train_meta_learner(oof_preds, y_train)

    # Step 4c: Retrain Base Models
    # Uses Validation-Guided Protocol (Early Stopping for Boosters, Full Data for others)
    trainer.retrain_final_models(feature_data, y_train, y_val)

    # 5. Validation Inference & Metric Calculation
    log_info("Performing inference on Validation Set...")

    model_names = list(trainer.base_models.keys())
    l1_val_preds = pd.DataFrame(index=np.arange(len(y_val)), columns=model_names)

    # Generate Level 1 predictions for Validation Set using retrained models
    for name in model_names:
        model = trainer.trained_base_models[name]
        views = trainer.model_requirements[name]

        # Access protected method to reuse concatenation logic
        X_val_combined = trainer._get_combined_features(feature_data, "val", views)

        if hasattr(model, "predict_proba"):
            l1_val_preds[name] = model.predict_proba(X_val_combined)[:, 1]
        else:
            l1_val_preds[name] = model.predict(X_val_combined)

    # Generate Level 2 predictions
    val_final_probs = trainer.meta_learner.predict_proba(l1_val_preds)[:, 1]

    # Compute and Print Metric
    final_score = compute_auc(y_val, val_final_probs)
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    log_info("Running Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(y_val - val_final_probs)

    # Correlate errors with numerical metadata features
    # We use the raw val_df to interpret original feature values
    analysis_cols = Config.METADATA_DENSE_FEATURES
    correlations = []

    for col in analysis_cols:
        if col in val_df.columns:
            # Simple imputation for correlation calculation
            feat_vals = val_df[col].fillna(0).values
            # Check for constant values to avoid warning
            if np.std(feat_vals) > 0:
                corr = np.corrcoef(feat_vals, errors)[0, 1]
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for col, corr in correlations[:5]:
        print(f"  {col}: {corr:.4f}")

    # 7. Submission Generation
    threshold = 0.7138293787137718
    if final_score > threshold:
        log_info(
            f"Validation score ({final_score}) exceeds threshold ({threshold}). Generating submission..."
        )
        trainer.generate_submission(feature_data, test_ids)
        log_info("Submission generation complete.")
    else:
        log_info(
            f"Validation score ({final_score}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
