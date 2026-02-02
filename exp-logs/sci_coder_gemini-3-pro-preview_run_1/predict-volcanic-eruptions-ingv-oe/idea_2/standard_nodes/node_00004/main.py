import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Import provided library components
from library.utils import seed_everything
from library.config import RANDOM_SEED
from library.data import VolcanoDataLoader
from library.model import CrossValidator, InferenceModel


def main():
    # 1. Setup and Initialization
    seed_everything(RANDOM_SEED)

    # 2. Data Loading
    # We load the full training and validation sets.
    # The dataset size (~3200 train, ~800 val) is small enough that we don't need
    # to subsample for the model training phase, and feature extraction is parallelized.
    print("Initializing Data Loader...")
    loader = VolcanoDataLoader()

    print("Loading Training Data...")
    X_train, y_train = loader.get_train_data(load_cached_data=True)

    print("Loading Validation Data...")
    X_val, y_val = loader.get_val_data(load_cached_data=True)

    # 3. Model Training
    print("Starting 5-Fold Cross-Validation Training...")
    # Initialize validator with default LightGBM config
    validator = CrossValidator()

    # Train the model (returns scores and overall CV MAE)
    # Models are saved to disk during this process
    cv_scores, cv_mae = validator.train(X_train, y_train)

    # 4. Validation Inference
    print("Performing Inference on Hold-out Validation Set...")
    # Load the trained models from disk
    inference_model = InferenceModel()

    # Generate predictions
    val_preds = inference_model.predict(X_val)

    # Compute Final Metric
    final_mae = mean_absolute_error(y_val, val_preds)
    print(f"Final Validation Metric: {final_mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)

    # Create a DataFrame to correlate features with error magnitude
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors.values

    # Compute correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Identify top 5 features most correlated with error
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 features correlated with prediction error:")
    for feature, corr_value in top_correlations.items():
        # Print feature name and the actual correlation (preserving sign)
        print(f"{feature}: {correlations[feature]:.4f}")

    # 6. Submission Generation
    THRESHOLD = 4534068.74

    if final_mae < THRESHOLD:
        print(
            f"\nValidation MAE ({final_mae}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test data
        print("Loading Test Data...")
        X_test, segment_ids = loader.get_test_data(load_cached_data=True)

        # Generate and save submission
        inference_model.generate_submission(X_test, segment_ids)
    else:
        print(
            f"\nValidation MAE ({final_mae}) exceeds threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
