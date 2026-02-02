import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, JigsawMetrics
from library.data_loader import load_data
from library.features import FeatureExtractor
from library.model import RidgeRegressor


def perform_failure_analysis(val_df, val_preds, target_col="target"):
    """
    Analyzes model errors on the validation set to identify systematic biases.
    Calculates the correlation between absolute error and identity attributes/text length.
    """
    print("\n==========================================")
    print("FAILURE ANALYSIS")
    print("==========================================")

    # Calculate Absolute Error
    # We use the raw continuous target and raw predictions
    analysis_df = val_df.copy()
    analysis_df["prediction"] = val_preds
    analysis_df["abs_error"] = (
        analysis_df[target_col] - analysis_df["prediction"]
    ).abs()

    # Calculate Text Length
    analysis_df["text_len"] = analysis_df["comment_text"].str.len()

    # Define features to correlate with error
    # We look at the identity columns defined in JigsawMetrics
    metrics_helper = JigsawMetrics()
    features_to_analyze = metrics_helper.identities + ["text_len"]

    correlations = {}

    for feature in features_to_analyze:
        if feature in analysis_df.columns:
            # Handle NaNs in identity columns (assume 0 if missing)
            feat_values = analysis_df[feature].fillna(0.0)

            # Calculate correlation with absolute error
            # We use pandas corr() which handles alignment
            corr = feat_values.corr(analysis_df["abs_error"])
            correlations[feature] = corr

    # Sort and print correlations
    # Positive correlation means higher presence of feature -> higher error
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print("Correlation between Feature and Model Error (Absolute Error):")
    print("(Positive values indicate the feature is associated with higher error)")
    print("-" * 50)
    for feat, corr in sorted_corrs:
        print(f"{feat:<30} : {corr:.6f}")

    print("==========================================\n")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing Toxicity Classification Pipeline...")

    # 2. Load Data
    # load_data handles caching and bias-corrective resampling internally
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=Config.DEBUG)

    # Extract targets from training data
    y_train = train_df["target"].values

    # 3. Feature Extraction
    # FeatureExtractor handles TF-IDF vectorization and caching
    extractor = FeatureExtractor()
    X_train, X_val, X_test = extractor.extract_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    print(f"Feature Matrix Shape (Train): {X_train.shape}")

    # 4. Model Training
    # Initialize and train Ridge Regression
    model = RidgeRegressor(
        alpha=Config.RIDGE_ALPHA, solver=Config.RIDGE_SOLVER, random_state=Config.SEED
    )

    model.train(X_train, y_train)

    # 5. Validation
    # Evaluate using the competition metrics
    # val_df contains the necessary identity columns for bias metric calculation
    results = model.evaluate(X_val, val_df, target_col="target")

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {results['score']}")

    # 6. Failure Analysis
    # Get raw predictions for analysis
    val_preds = model.predict(X_val)
    perform_failure_analysis(val_df, val_preds)

    # 7. Submission
    # Generate predictions for test set
    print("Generating predictions for Test set...")
    test_preds = model.predict(X_test)

    # Save submission file
    model.save_submission(
        test_ids=test_df["id"],
        test_preds=test_preds,
        output_path=Config.SUBMISSION_PATH,
    )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
