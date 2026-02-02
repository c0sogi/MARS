import os
import sys
import numpy as np
import pandas as pd
import warnings
import importlib
import library.config

# Cite debug_lesson_1: Force reload of config to ensure new variables (XGB_PARAMS) are picked up
importlib.reload(library.config)

# Import from provided library files
from library.config import RANDOM_STATE
from library.data_handler import FeatureEngineer
from library.model_architecture import train_model, generate_submission
from library.metrics import laplace_log_likelihood

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_STATE)
    print("Starting pipeline execution...")

    # 2. Data Loading
    # Initialize FeatureEngineer and load datasets
    # load_cached_data=True ensures we use pre-computed features if available
    fe = FeatureEngineer()
    X_train, y_train, X_val, y_val, X_test, test_df = fe.load_datasets(
        load_cached_data=True
    )

    # 3. Model Training
    # The train_model function handles the initialization and fitting of the DualModel
    # (ElasticNet for FVC and ElasticNet for Confidence)
    model = train_model(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    # We perform inference again to ensure we capture the exact metric value for the required print statement
    val_fvc_pred, val_sigma_pred = model.predict(X_val)
    final_metric = laplace_log_likelihood(y_val, val_fvc_pred, val_sigma_pred)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error magnitude
    abs_errors = np.abs(y_val - val_fvc_pred)

    # Retrieve feature names from the preprocessor to make analysis readable
    try:
        # The preprocessor is a ColumnTransformer
        feature_names = fe.preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback if scikit-learn version is old (though 1.7.2 is specified)
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    # Create a DataFrame for correlation analysis
    analysis_df = pd.DataFrame(X_val, columns=feature_names)
    analysis_df["Error_Magnitude"] = abs_errors

    # Compute correlation between features and error magnitude
    # We are interested in features that highly correlate (positively or negatively) with error
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")

    # Sort by absolute correlation
    top_correlations = correlations.iloc[correlations.abs().argsort()[::-1]].head(5)

    print("Top 5 features correlated with prediction error magnitude:")
    print(top_correlations)

    # 6. Submission Generation
    # Generate predictions for the test set and save to submission.csv
    BASELINE_METRIC = -7.158702679895534

    if final_metric > BASELINE_METRIC:
        print(
            f"Metric improved over baseline ({BASELINE_METRIC}). Generating submission..."
        )
        generate_submission(model, X_test, test_df)
    else:
        print(
            f"Metric {final_metric} did not improve over baseline {BASELINE_METRIC}. Skipping submission."
        )

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
