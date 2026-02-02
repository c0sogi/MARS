import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Import from provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data_processor import process_and_cache_data
from library.ensemble_model import StackingEnsemble


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data Loading & Processing
    # Loads pre-processed features from cache (idea_9) to ensure fast execution.
    # Returns PCA-reduced features concatenated with scaled metadata.
    print("Loading and processing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = process_and_cache_data(
        load_cached_data=True
    )

    # 3. Model Training
    print("Initializing Stacking Ensemble...")
    ensemble = StackingEnsemble()

    # Step 3a: Train Meta-Learner via Cross-Validation
    # This generates OOF predictions from base learners and fits the Ridge meta-learner.
    print("Running Cross-Validation to train Meta-Learner...")
    ensemble.train_cv(X_train, y_train)

    # Step 3b: Retrain Base Learners
    # Retrains SVR, KNN, ExtraTrees, and LightGBM on the full training set for final inference.
    print("Retraining Base Learners on full training set...")
    ensemble.fit_final(X_train, y_train)

    # 4. Validation Inference
    print("Evaluating on Hold-out Validation Set...")
    # Predict using the stacked ensemble (Base Models -> Meta Learner)
    val_preds = ensemble.predict(X_val)

    # Calculate RMSE
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)

    # Print the required metric
    print(f"Final Validation Metric: {rmse}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    residuals = np.abs(y_val - val_preds)

    # Load validation metadata to get original feature names and values
    if os.path.exists(Config.VAL_META_PATH):
        val_df = pd.read_csv(Config.VAL_META_PATH)

        # Metadata features to analyze for correlation with error
        # Note: "Subject Focus" is the column name used in data_handling.py
        meta_features = [
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

        print("Correlation between Absolute Error and Metadata Features:")
        for feat in meta_features:
            # Handle potential column naming variations (e.g. Focus vs Subject Focus)
            col_name = feat
            if col_name not in val_df.columns and feat == "Subject Focus":
                if "Focus" in val_df.columns:
                    col_name = "Focus"

            if col_name in val_df.columns:
                feat_values = val_df[col_name].values
                # Calculate Pearson correlation
                corr, _ = pearsonr(feat_values, residuals)
                print(f"  {feat}: {corr:.10f}")

        # Correlation with the Target value itself
        target_corr, _ = pearsonr(y_val, residuals)
        print(f"  Target (Pawpularity): {target_corr:.10f}")
    else:
        print("Validation metadata file not found. Skipping detailed failure analysis.")

    # 6. Submission Generation
    # Only submit if the validation metric is better (lower) than the threshold
    THRESHOLD = 17.361083072547856

    if rmse < THRESHOLD:
        print(
            f"\nValidation metric ({rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        ensemble.generate_submission(X_test, test_ids)
    else:
        print(
            f"\nValidation metric ({rmse}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
