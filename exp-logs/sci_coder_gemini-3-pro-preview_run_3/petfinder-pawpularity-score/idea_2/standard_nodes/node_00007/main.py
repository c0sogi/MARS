import pandas as pd
import numpy as np
import sys
import os

# Import from the provided library
from library import config, utils, data_loader, feature_extractor, svr_model


def main():
    # ==========================================
    # 1. Setup
    # ==========================================
    utils.seed_everything()

    # ==========================================
    # 2. Data Loading & Feature Extraction
    # ==========================================
    # Initialize DataLoaders
    # We use the default batch size and workers from config
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()

    # Extract Features
    # The feature_extractor module handles caching logic internally.
    # It will check for .npy files in ./working/idea_2/ and load them if present.
    # Otherwise, it uses the GPU to extract features from the images.

    print("Processing Training Data...")
    train_X, train_y = feature_extractor.get_features(
        train_loader, mode="train", load_cached_data=True
    )

    print("Processing Validation Data...")
    val_X, val_y = feature_extractor.get_features(
        val_loader, mode="val", load_cached_data=True
    )

    print("Processing Test Data...")
    test_X, test_ids = feature_extractor.get_features(
        test_loader, mode="test", load_cached_data=True
    )

    # ==========================================
    # 3. Model Training
    # ==========================================
    # Initialize the SVR wrapper
    regressor = svr_model.PetPawpularityRegressor()

    # Fit the model
    # The fit method performs GridSearchCV to find the best C and epsilon
    # We use the full training set as SVR scales reasonably well to ~7k samples
    # and we want the best possible score.
    regressor.fit(train_X, train_y)

    # ==========================================
    # 4. Validation
    # ==========================================
    print("Evaluating on Validation Set...")
    val_preds = regressor.predict(val_X)

    # Calculate RMSE
    rmse = utils.calculate_rmse(val_y, val_preds)

    # REQUIRED: Print the final validation metric in the exact format
    print(f"Final Validation Metric: {rmse}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Calculate absolute errors (residuals)
    errors = np.abs(val_y - val_preds)

    # Load validation metadata to get the explicit feature values for correlation
    # We reload from CSV to ensure we have the named columns easily accessible
    val_df = pd.read_csv(config.VAL_META_PATH)

    print("Correlation between Absolute Prediction Error and Metadata Features:")
    print(f"{'Feature':<15} | {'Correlation':<10}")
    print("-" * 30)

    for feature in config.META_FEATURES:
        if feature in val_df.columns:
            feat_values = val_df[feature].values
            # Calculate Pearson correlation
            if len(feat_values) == len(errors):
                corr = np.corrcoef(feat_values, errors)[0, 1]
                print(f"{feature:<15} | {corr:.6f}")
            else:
                print(f"{feature:<15} | Shape Mismatch")

    # ==========================================
    # 6. Submission
    # ==========================================
    # Threshold defined in the task description
    THRESHOLD = 18.44368820663551

    if rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({rmse}) meets the threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        test_preds = regressor.predict(test_X)

        # Save submission
        svr_model.generate_submission(test_ids, test_preds)
    else:
        print(
            f"\nValidation RMSE ({rmse}) does NOT meet the threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
