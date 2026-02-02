import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import set_seed, SEED, FEATURE_PREFIXES, WORKING_DIR
from library.data_loader import load_and_process_data
from library.factorized_lda import GlobalOASDiscriminant
from library.utils import calculate_log_loss, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_pipeline():
    # 1. Initialization and Reproducibility
    set_seed(SEED)

    # 2. Data Loading
    # Using load_cached_data=True as requested to utilize preprocessed artifacts
    data, class_names = load_and_process_data(load_cached_data=True)

    # 3. Model Training
    # The GlobalOASDiscriminant uses sklearn/numpy (CPU) which is highly efficient for this dataset size.
    # GPU utilization is not applicable for this specific analytical solution provided in the library.
    model = GlobalOASDiscriminant()
    model.fit(data["train"]["X"], data["train"]["y"])

    # 4. Validation Inference
    # Predict probabilities on the validation set
    val_probs = model.predict_proba(data["val"]["X"])

    # Calculate Metric
    val_loss = calculate_log_loss(data["val"]["y"], val_probs)

    # Print required metric format
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    # Calculate error magnitude: -log(p_true)
    y_val = data["val"]["y"]
    # Select probability assigned to the true class
    true_class_probs = val_probs[np.arange(len(y_val)), y_val]
    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1.0)
    error_magnitudes = -np.log(true_class_probs)

    # Flatten validation features for correlation analysis
    # We iterate through FEATURE_PREFIXES to maintain consistent order
    feature_arrays = []
    feature_names = []

    for group in FEATURE_PREFIXES:
        arr = data["val"]["X"][group]
        feature_arrays.append(arr)
        # Generate generic names since exact column headers aren't stored in the numpy arrays
        # The preprocessor sorts them numerically (e.g., margin_1, margin_2...)
        for i in range(arr.shape[1]):
            feature_names.append(f"{group}_{i+1}")

    X_val_flat = np.hstack(feature_arrays)

    # Calculate correlations
    correlations = []
    for i in range(X_val_flat.shape[1]):
        feature_vec = X_val_flat[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feature_vec) > epsilon:
            corr, _ = pearsonr(feature_vec, error_magnitudes)
            # Check for NaN result just in case
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation strength (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nFailure Analysis - Top 5 Features Correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # 6. Submission Generation
    # Strict threshold check
    threshold = 1.2136771218566717e-09

    if val_loss < threshold:
        test_probs = model.predict_proba(data["test"]["X"])
        save_submission(data["test"]["ids"], test_probs, class_names)
    else:
        print(
            f"Validation metric {val_loss} is not lower than threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
