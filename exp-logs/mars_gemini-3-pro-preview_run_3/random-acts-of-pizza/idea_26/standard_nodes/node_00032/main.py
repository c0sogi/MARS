import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engineering
from library import training_engine


def main():
    # 1. Configuration and Seeding
    utils.print_header("Initialization")
    utils.set_seed(config.SEED)

    # 2. Data Loading
    # Load train, val, and test datasets using the provided data loader
    # We use load_cached=True to leverage any existing preprocessed files
    train_df, val_df, test_df = data_loader.load_data(load_cached=True)

    # Extract targets
    y_train = train_df[config.TARGET_COL]
    y_val = val_df[config.TARGET_COL]

    # 3. Feature Engineering
    utils.print_header("Feature Engineering")

    # Initialize the ViewBuilder
    view_builder = feature_engineering.ViewBuilder()

    # Fit transformers on training data
    with utils.Timer("Fitting ViewBuilder"):
        view_builder.fit(train_df)

    # Transform datasets into the required views (Metadata, Lexical, Behavioral, Semantic)
    with utils.Timer("Transforming Train Data"):
        X_train_views = view_builder.transform(train_df, "train", load_cached=True)

    with utils.Timer("Transforming Validation Data"):
        X_val_views = view_builder.transform(val_df, "val", load_cached=True)

    with utils.Timer("Transforming Test Data"):
        X_test_views = view_builder.transform(test_df, "test", load_cached=True)

    # 4. Model Training
    utils.print_header("Model Training")

    # Initialize the StackingTrainer
    trainer = training_engine.StackingTrainer()

    # Execute the training pipeline (5-Fold CV + Meta-Learner + Retraining)
    with utils.Timer("Training Stacking Ensemble"):
        trainer.fit(X_train_views, y_train, X_val_views, y_val)

    # 5. Validation and Evaluation
    utils.print_header("Validation & Failure Analysis")

    # Generate predictions on the validation set
    val_preds = trainer.predict(X_val_views)

    # Calculate final metric
    final_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Calculate absolute error (since we predict probability of class 1)
    # Error is |Target - Probability|
    errors = np.abs(y_val - val_preds)

    print("\n--- Correlation between Error and Metadata Features ---")
    correlations = []

    # Calculate correlation for each numerical metadata feature
    for col in config.METADATA_FEATURES:
        if col in val_df.columns:
            # Ensure no NaNs before correlation (though ViewBuilder handles them, val_df is raw)
            # We use the raw val_df for analysis context
            feat_series = val_df[col].fillna(val_df[col].median())

            # Align indices just in case, though they should match
            if len(feat_series) == len(errors):
                corr = np.corrcoef(feat_series, errors)[0, 1]
                if not np.isnan(corr):
                    correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, corr in correlations:
        print(f"{col}: {corr:.4f}")

    # 6. Submission Generation
    utils.print_header("Submission Generation")

    threshold = 0.7085870249842536

    if final_auc > threshold:
        utils.print_info(
            f"Validation AUC ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Generate predictions on test set
        test_preds = trainer.predict(X_test_views)

        # Save submission
        trainer.save_predictions(
            test_df[config.ID_COL], test_preds, config.SUBMISSION_PATH
        )
    else:
        utils.print_info(
            f"Validation AUC ({final_auc}) does not exceed threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
