import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, save_submission
from library.preprocessor import load_processed_data
from library.model import PizzaSuccessModel


def main():
    # 1. Setup and Configuration
    # Set random seeds for reproducibility
    set_seed(Config.RANDOM_SEED)

    # 2. Data Loading
    # Load cached, preprocessed data (Sparse Matrices for X, Arrays for y)
    # This utilizes the pipeline defined in library/preprocessor.py and library/data_loader.py
    print("Loading preprocessed datasets...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_processed_data(
        load_cached_data=False
    )

    # 3. Model Training
    # Initialize the Random Forest model wrapper
    print("Initializing model...")
    model = PizzaSuccessModel()

    # Train the model on the training set and evaluate on validation set during training
    # The model.train method prints the initial AUCs
    model.train(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    # Generate probabilities for the positive class (received pizza)
    print("Running validation inference...")
    val_probs = model.predict_proba(X_val)

    # Calculate final validation metric (AUC)
    val_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nPerforming failure analysis on validation set...")
    # Calculate absolute error: |True Label - Predicted Probability|
    # High error means confident wrong prediction or low confidence right prediction
    errors = np.abs(y_val - val_probs)

    # Extract numerical features for correlation analysis
    # The preprocessor stacks Text Features (TF-IDF) followed by Numerical Features.
    # We retrieve the numerical features from the end of the sparse matrix.
    num_numerical_cols = len(Config.NUMERICAL_COLS)

    # Slice the sparse matrix to get the last N columns corresponding to numerical features
    # Convert to dense array for correlation calculation
    # Convert to CSR format because COO format (default from hstack) does not support slicing
    X_val_numerical = X_val.tocsr()[:, -num_numerical_cols:].toarray()

    print(
        f"Correlating prediction errors with {num_numerical_cols} numerical features:"
    )
    correlations = []

    for i, feature_name in enumerate(Config.NUMERICAL_COLS):
        feature_values = X_val_numerical[:, i]

        # Calculate Pearson correlation between the feature value and the error magnitude
        # Handle constant features to prevent division by zero/NaNs
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, feature_values)

        correlations.append((feature_name, corr))

    # Sort features by the magnitude of their correlation with error (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top correlations
    for name, corr in correlations:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    if val_auc > 0.6565997512879729:
        print("\nValidation metric improved. Generating test set predictions...")
        test_probs = model.predict_proba(X_test)

        # Save predictions to the submission file
        save_submission(test_probs, test_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {val_auc} did not improve upon baseline 0.6556. Skipping submission."
        )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
