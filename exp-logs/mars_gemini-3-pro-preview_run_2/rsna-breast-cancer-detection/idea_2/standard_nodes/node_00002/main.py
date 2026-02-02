import sys
import os
import numpy as np
import pandas as pd
import warnings

# Ensure library modules are accessible
sys.path.append(os.getcwd())

from library.config import setup_system
from library.data_handler import get_processed_data
from library.tabular_model import CancerClassifier
from library.utils import probabilistic_f1


def perform_failure_analysis(val_df, y_true, y_pred):
    """
    Analyzes the correlation between prediction errors and input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error magnitude
    errors = np.abs(y_true - y_pred)

    # Prepare analysis dataframe
    df_analyze = val_df.copy()
    df_analyze["error"] = errors

    # Map ordinal density to numeric
    density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    if "density" in df_analyze.columns:
        df_analyze["density_numeric"] = df_analyze["density"].map(density_map)

    features_to_check = {
        "Age": "age",
        "Density": "density_numeric",
        "Implant": "implant",
    }

    print("Correlation between Error Magnitude and Features:")
    for name, col in features_to_check.items():
        if col in df_analyze.columns:
            # Drop NaNs for correlation calculation
            subset = df_analyze[[col, "error"]].dropna()

            if len(subset) > 1:
                # Calculate Pearson correlation
                # We use numpy to avoid extra dependency issues, though scipy is likely available
                try:
                    corr = np.corrcoef(
                        subset[col].astype(float), subset["error"].astype(float)
                    )[0, 1]
                    print(f"  {name}: {corr:.6f}")
                except Exception as e:
                    print(f"  {name}: Could not calculate (Error: {e})")
            else:
                print(f"  {name}: Not enough data")


def main():
    # 1. Setup Environment
    setup_system()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 2. Data Loading & Feature Extraction
    # This step loads metadata and generates/loads ResNet18 embeddings
    print("Retrieving processed data...")
    train_df, val_df, test_df = get_processed_data(load_cached_data=True)

    # 3. Model Training
    # Initialize and train the LightGBM classifier
    print("Initializing classifier...")
    classifier = CancerClassifier()

    print("Starting training...")
    classifier.fit(train_df, val_df)

    # 4. Validation Assessment
    print("Performing final validation assessment...")
    # Preprocess validation data to match training format
    X_val, y_val = classifier._preprocess(val_df, is_train=False)

    # Generate probabilities using the best iteration of the trained model
    val_probs = classifier.model.predict(
        X_val, num_iteration=classifier.model.best_iteration
    )

    # Compute Metric
    pf1_score = probabilistic_f1(y_val, val_probs)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {pf1_score}")

    # 5. Failure Analysis
    perform_failure_analysis(val_df, y_val, val_probs)

    # 6. Submission Generation
    print("\nGenerating submission...")
    classifier.predict_and_submit(test_df)
    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
